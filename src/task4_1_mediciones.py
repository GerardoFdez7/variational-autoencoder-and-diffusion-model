"""Task 4.1: mediciones de nitidez y tiempo de generacion, VAE (beta = 1) vs. difusion (w = 1).

Este script solo produce las mediciones; el analisis lo escribe quien hace la 4.1.
Todo se mide en la misma maquina para que la razon de tiempos sea valida.

- El VAE de la Task 1 no se guardo, asi que se reentrena con beta = 1 usando
  task1_vae.entrenar (misma semilla e hiperparametros) y se compara su reconstruccion
  y KL finales con results/task1_results.json. En otra maquina o version de PyTorch
  el resultado no es identico bit a bit; la diferencia queda registrada en el JSON.
- Las muestras de difusion (w = 1) se reutilizan de results/samples/task3_4_w1.pt.
- La nitidez se calcula con task1_vae.nitidez sobre imagenes en [0, 1], igual que en la 1.4.
"""
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import task1_vae
import task3_sampling as ts

N_NITIDEZ = 500
N_TIEMPO = 100
REPS_VAE = 20            # el VAE tarda milisegundos: se toma la mediana de varias corridas
REPS_DIFUSION = 3
VAE_CKPT = os.path.join(common.RESULTS_DIR, "checkpoints", "vae_beta1.pt")
VAE_SAMPLES = os.path.join(ts.SAMPLES_DIR, "task4_1_vae_beta1.pt")


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def obtener_vae(x_train, x_val):
    """Carga el VAE beta = 1 guardado o lo reentrena y verifica contra la Task 1."""
    model = task1_vae.VAE()
    if os.path.exists(VAE_CKPT):
        ck = torch.load(VAE_CKPT, weights_only=True)
        model.load_state_dict(ck["model"])
        model.eval()
        return model, ck["comparacion_task1"]
    model, hist = task1_vae.entrenar(1.0, x_train, x_val)
    ref = json.load(open(os.path.join(common.RESULTS_DIR, "task1_results.json")))["tabla_1_4a"]["1.0"]
    comp = {"recon_final": hist["recon"][-1], "kl_final": hist["kl"][-1],
            "recon_task1": ref["recon_final"], "kl_task1": ref["kl_nats_final"]}
    comp["error_rel_recon"] = (comp["recon_final"] - comp["recon_task1"]) / comp["recon_task1"]
    comp["error_rel_kl"] = (comp["kl_final"] - comp["kl_task1"]) / comp["kl_task1"]
    os.makedirs(os.path.dirname(VAE_CKPT), exist_ok=True)
    torch.save({"model": model.state_dict(), "comparacion_task1": comp}, VAE_CKPT)
    model.eval()
    return model, comp


@torch.no_grad()
def tiempo_vae(model, device, n=N_TIEMPO, reps=REPS_VAE):
    """Mediana de segundos para generar n imagenes: z ~ N(0, I) y una pasada del decoder."""
    model = model.to(device)
    model.decode(torch.randn(n, task1_vae.LATENT_DIM, device=device))     # calentamiento
    tiempos = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        model.decode(torch.randn(n, task1_vae.LATENT_DIM, device=device)).cpu()
        sync(device)
        tiempos.append(time.perf_counter() - t0)
    return statistics.median(tiempos)


def tiempo_difusion(unet, device, n=N_TIEMPO, reps=REPS_DIFUSION):
    """Mediana de segundos para generar n imagenes con muestreo ancestral y w = 1."""
    labels = ts.class_labels(n // ts.NUM_CLASSES)
    ts.sample(unet, labels[:2], 1, torch.Generator().manual_seed(0), device)  # calentamiento
    tiempos = []
    for r in range(reps):
        _, secs = ts.sample_many(unet, labels, 1, torch.Generator().manual_seed(common.SEED + r),
                                 device, chunk=n)
        tiempos.append(secs)
    return statistics.median(tiempos)


def main():
    common.set_seed()
    os.makedirs(ts.SAMPLES_DIR, exist_ok=True)
    x_train, _, x_val, _, *_ = common.load_fashion_mnist(scale="unit")

    # ------------------------------------------------------------ modelo VAE
    vae, comp = obtener_vae(x_train, x_val)
    print("VAE beta = 1 frente a la Task 1:", comp)

    # --------------------------------------------- muestras para la nitidez
    if os.path.exists(VAE_SAMPLES):
        x_vae = torch.load(VAE_SAMPLES, weights_only=True)["images"]
    else:
        g = torch.Generator().manual_seed(common.SEED)
        with torch.no_grad():
            x_vae = vae.decode(torch.randn(N_NITIDEZ, task1_vae.LATENT_DIM, generator=g))
        torch.save({"images": x_vae, "beta": 1.0, "seed": common.SEED, "rango": "[0, 1]"},
                   VAE_SAMPLES)
    difusion = torch.load(ts.samples_path(1), weights_only=True)
    x_dif = (difusion["images"] + 1.0) / 2.0               # [-1, 1] -> [0, 1]
    x_real = x_val[:N_NITIDEZ]
    nitidez = {"real": task1_vae.nitidez(x_real),
               "vae_beta1": task1_vae.nitidez(x_vae),
               "difusion_w1": task1_vae.nitidez(x_dif)}
    print("Nitidez:", nitidez)

    # ------------------------------------------- tiempos en el mismo equipo
    unet_cpu = ts.load_model(device="cpu")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    tiempos = {}
    for dev in devices:
        unet = unet_cpu.to(dev)
        t_vae = tiempo_vae(vae, dev)
        t_dif = tiempo_difusion(unet, dev)
        tiempos[dev] = {"vae_segundos_100": t_vae, "difusion_segundos_100": t_dif,
                        "razon_difusion_sobre_vae": t_dif / t_vae}
        print(f"{dev}: VAE {t_vae * 1e3:.2f} ms | difusion {t_dif:.2f} s | "
              f"razon {t_dif / t_vae:,.0f}x")
    vae.to("cpu")

    results = {
        "seed": common.SEED,
        "equipo": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "hilos_cpu": torch.get_num_threads()},
        "vae_vs_task1": comp,
        "nitidez": {"n_imagenes": N_NITIDEZ, "rango_pixeles": "[0, 1]",
                    "definicion": "media de |x[:, :, :, 1:] - x[:, :, :, :-1]| (igual que Task 1.4)",
                    "reales": "primeras 500 de validacion", **nitidez},
        "tiempo_100_imagenes": {
            "evaluaciones_red": {"vae": N_TIEMPO, "difusion_w1": N_TIEMPO * ts.T},
            "repeticiones": {"vae": REPS_VAE, "difusion": REPS_DIFUSION, "estadistico": "mediana"},
            **tiempos},
        "archivos": {"vae_checkpoint": os.path.relpath(VAE_CKPT, common.ROOT),
                     "vae_muestras": os.path.relpath(VAE_SAMPLES, common.ROOT),
                     "difusion_muestras": os.path.relpath(ts.samples_path(1), common.ROOT)},
    }
    print("JSON:", common.save_results("task4_1_results.json", results))


if __name__ == "__main__":
    main()
