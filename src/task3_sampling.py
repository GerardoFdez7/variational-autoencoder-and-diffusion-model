"""Task 3.3 y 3.4(a-b): muestreo ancestral y guia sin clasificador (classifier-free guidance).

Requiere el checkpoint de la Task 3.2 (results/checkpoints/unet_cond.pt).
Las muestras de 3.4 se guardan en results/samples/ para que la evaluacion (3.4 c-d,
3.5 y 4.1) no tenga que regenerarlas; si un archivo ya existe no se vuelve a generar
(usar --fresh para forzarlo).
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from task2_diffusion import T, BETA_START, BETA_END, linear_schedule, extract
from task3_train import CKPT_PATH
from task3_unet import UNet, NULL_CLASS, NUM_CLASSES

SAMPLES_DIR = os.path.join(common.RESULTS_DIR, "samples")
GUIDANCE_WS = [1, 3, 7]
N_GRID_PER_CLASS = 10          # Task 3.3: 10 imagenes por clase
N_EVAL_PER_CLASS = 50          # Task 3.4: 50 imagenes por clase y por w
CHUNK = 250                    # imagenes por lote durante el muestreo


def samples_path(w):
    return os.path.join(SAMPLES_DIR, f"task3_4_w{w}.pt")


def load_model(path=CKPT_PATH, device="cpu"):
    """Carga la U-Net entrenada en la Task 3.2 en modo evaluacion."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontro el checkpoint {path}. Ejecutar primero src/task3_train.py.")
    ck = torch.load(path, map_location=device, weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def guided_eps(model, x, t, c, w):
    """Prediccion de ruido con guia sin clasificador:

        eps_w = eps(x_t, t, nulo) + w * (eps(x_t, t, c) - eps(x_t, t, nulo))

    Con w = 1 se reduce a la prediccion condicional, por lo que se evalua la red una
    sola vez. Con w != 1 las dos ramas van en un solo lote (2B imagenes).
    """
    if w == 1:
        return model(x, t, c)
    out = model(torch.cat([x, x]), torch.cat([t, t]),
                torch.cat([c, torch.full_like(c, NULL_CLASS)]))
    eps_c, eps_u = out.chunk(2)
    return eps_u + w * (eps_c - eps_u)


@torch.no_grad()
def sample(model, labels, w=1, generator=None, device="cpu"):
    """Muestreo ancestral (DDPM) desde x_T ~ N(0, I):

        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_w) + sigma_t z

    con sigma_t^2 = beta_t y z ~ N(0, I) para t > 1, z = 0 en t = 1.
    Devuelve x_0 recortado a [-1, 1] con forma (B, 1, 28, 28).
    """
    betas, alphas, alpha_bar = linear_schedule(T, BETA_START, BETA_END)
    B = labels.shape[0]
    c = labels.to(device)
    x = torch.randn(B, 1, 28, 28, generator=generator).to(device)
    for step in range(T, 0, -1):
        t = torch.full((B,), step, dtype=torch.long)
        eps = guided_eps(model, x, t.to(device), c, w)
        coef = extract(betas, t) / torch.sqrt(1.0 - extract(alpha_bar, t))
        mean = (x - coef.to(device) * eps) / torch.sqrt(extract(alphas, t)).to(device)
        if step > 1:
            z = torch.randn(B, 1, 28, 28, generator=generator).to(device)
            x = mean + torch.sqrt(extract(betas, t)).to(device) * z
        else:
            x = mean
    return x.clamp(-1.0, 1.0).cpu()


def sample_many(model, labels, w, generator, device="cpu", chunk=CHUNK):
    """Muestrea en lotes de `chunk` imagenes. Devuelve (imagenes, segundos)."""
    out = []
    start = time.perf_counter()
    for i in range(0, labels.shape[0], chunk):
        out.append(sample(model, labels[i:i + chunk], w, generator, device))
    return torch.cat(out), time.perf_counter() - start


def class_labels(n_per_class):
    """Etiquetas 0,0,...,1,1,...,9 con n_per_class repeticiones de cada clase."""
    return torch.arange(NUM_CLASSES).repeat_interleave(n_per_class)


def network_evals(n_images, w):
    """Evaluaciones de la red (por imagen y paso) necesarias para n_images."""
    return n_images * T * (1 if w == 1 else 2)


def pixel_stats(images, labels):
    """Estadisticas de saturacion y dispersion para describir el efecto de w.

    Las imagenes estan en [-1, 1]. Un pixel se considera saturado si |x| >= 0.99.
    La dispersion intraclase es la desviacion estandar por pixel entre imagenes de
    la misma clase, promediada sobre pixeles y clases.
    """
    sat = images.abs() >= 0.99
    disp = torch.stack([images[labels == c].std(0).mean() for c in range(NUM_CLASSES)])
    return {"frac_pixeles_saturados": float(sat.float().mean()),
            "frac_pixeles_blancos": float((images >= 0.99).float().mean()),
            "media_pixel": float(images.mean()),
            "dispersion_intraclase": float(disp.mean())}


def save_grid(images, n_per_class, title, name):
    """Cuadricula con una fila por clase y n_per_class columnas."""
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class,
                             figsize=(n_per_class * 0.9 + 1.6, NUM_CLASSES * 0.9 + 0.6))
    for r in range(NUM_CLASSES):
        for col in range(n_per_class):
            ax = axes[r, col]
            ax.imshow(images[r * n_per_class + col, 0], cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(common.CLASS_NAMES[r], rotation=0, ha="right",
                              va="center", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path = common.figure_path(name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    common.set_seed()
    common.ensure_dirs()
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fresh = "--fresh" in sys.argv
    model = load_model(device=device)
    print(f"Dispositivo: {device} | hilos: {torch.get_num_threads()}")
    # Calentamiento para que la inicializacion de CUDA no cuente en los tiempos.
    guided_eps(model, torch.zeros(2, 1, 28, 28, device=device),
               torch.ones(2, dtype=torch.long, device=device),
               torch.zeros(2, dtype=torch.long, device=device), 3)

    # --------------------------------------------- Task 3.3: cuadricula con w = 1
    # Cada bloque de muestras usa su propio generador para que sea reproducible
    # aunque los demas bloques se omitan por estar ya guardados.
    g = torch.Generator().manual_seed(common.SEED)
    grid, secs_grid = sample_many(model, class_labels(N_GRID_PER_CLASS), 1, g, device)
    grid_path = save_grid(grid, N_GRID_PER_CLASS,
                          "Muestreo ancestral condicional (w = 1)", "task3_3_grid_w1.png")
    print(f"Task 3.3: 100 imagenes en {secs_grid:.1f}s -> {grid_path}")

    # ------------------------------ Task 3.4: tiempo por 100 imagenes para cada w
    tiempos = {}
    for w in GUIDANCE_WS:
        if w == 1:
            secs = secs_grid          # la cuadricula de 3.3 ya son 100 imagenes con w = 1
        else:
            g = torch.Generator().manual_seed(common.SEED + 100 + w)
            _, secs = sample_many(model, class_labels(N_GRID_PER_CLASS), w, g, device)
        tiempos[w] = {"segundos_100_imagenes": secs,
                      "evaluaciones_red_100_imagenes": network_evals(100, w)}
        print(f"  w={w}: 100 imagenes en {secs:.1f}s")

    # --------------------- Task 3.4(a-b): 50 imagenes por clase para cada valor de w
    muestras = {}
    for w in GUIDANCE_WS:
        path = samples_path(w)
        if os.path.exists(path) and not fresh:
            print(f"  w={w}: ya existe {path}, se reutiliza")
            data = torch.load(path, weights_only=True)
        else:
            g = torch.Generator().manual_seed(common.SEED + w)
            labels = class_labels(N_EVAL_PER_CLASS)
            images, secs = sample_many(model, labels, w, g, device)
            data = {"images": images, "labels": labels, "w": w, "seed": common.SEED + w,
                    "segundos": secs, "rango": "[-1, 1]"}
            torch.save(data, path)
            print(f"  w={w}: {images.shape[0]} imagenes en {secs:.1f}s -> {path}")
        save_grid(data["images"].view(NUM_CLASSES, N_EVAL_PER_CLASS, 1, 28, 28)
                  [:, :N_GRID_PER_CLASS].reshape(-1, 1, 28, 28),
                  N_GRID_PER_CLASS, f"Guia sin clasificador (w = {w})",
                  f"task3_4_grid_w{w}.png")
        muestras[w] = {"archivo": os.path.relpath(path, common.ROOT),
                       "n_imagenes": int(data["images"].shape[0]),
                       "segundos": data["segundos"],
                       **pixel_stats(data["images"], data["labels"])}

    # Referencia: las mismas estadisticas sobre 500 imagenes reales de validacion
    # (las primeras 50 de cada clase).
    _, _, x_val, y_val, *_ = common.load_fashion_mnist(scale="symmetric")
    idx = torch.cat([torch.nonzero(y_val == c)[:N_EVAL_PER_CLASS, 0] for c in range(NUM_CLASSES)])
    reales = pixel_stats(x_val[idx], y_val[idx])

    results = {"seed": common.SEED, "dispositivo": device, "T": T,
               "sampler": "DDPM ancestral, sigma_t^2 = beta_t",
               "guia": "eps_w = eps_nulo + w (eps_cond - eps_nulo)",
               "task3_3": {"n_por_clase": N_GRID_PER_CLASS, "segundos": secs_grid,
                           "figura": "results/figures/task3_3_grid_w1.png"},
               "task3_4": {"ws": GUIDANCE_WS, "n_por_clase": N_EVAL_PER_CLASS,
                           "muestras": {str(w): m for w, m in muestras.items()},
                           "tiempo_por_100_imagenes": {str(w): v for w, v in tiempos.items()},
                           "estadisticas_imagenes_reales": reales}}
    path = common.save_results("task3_3_4_results.json", results)
    print("JSON:", path)


if __name__ == "__main__":
    main()
