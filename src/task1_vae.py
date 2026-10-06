"""Task 1: VAE sobre Fashion-MNIST con espacio latente de dimension 2.

Entrena un VAE por cada beta, genera las figuras 1.2 y 1.3 y calcula los
numeros de 1.4. Resultados en results/task1_results.json.
"""
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import CLASS_NAMES, figure_path, save_results, set_seed

BATCH_SIZE = 128
LATENT_DIM = 2
EPOCHS = 20
LR = 1e-3
BETA_VALUES = [0.1, 1.0, 10.0]
ARQUITECTURA = (
    "MLP. Encoder: 784-512-256 (ReLU) y dos cabezas lineales 256->2 (mu, logvar). "
    "Decoder: 2-256-512-784 (ReLU) con sigmoide final. Misma arquitectura para todos los beta."
)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, LATENT_DIM)
        self.fc_logvar = nn.Linear(256, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)


def reparametrizar(mu, logvar):
    """z = mu + sigma * eps, con eps ~ N(0, I)."""
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn_like(sigma)
    return mu + sigma * eps


def kl_cerrada(mu, logvar):
    """KL(N(mu, diag(sigma^2)) || N(0, I)) por imagen, sumada sobre dimensiones latentes."""
    return 0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - 1.0 - logvar, dim=1)


def error_reconstruccion(x, x_hat):
    """Error cuadratico por imagen, sumado sobre los 784 pixeles."""
    return torch.sum((x - x_hat) ** 2, dim=(1, 2, 3))


def perdida_por_imagen(x, x_hat, mu, logvar, beta):
    recon = error_reconstruccion(x, x_hat)
    kl = kl_cerrada(mu, logvar)
    return recon + beta * kl, recon, kl


@torch.no_grad()
def evaluar(model, x_val, beta):
    """Reconstruccion y KL medias por imagen sobre validacion."""
    model.eval()
    mu, logvar = model.encode(x_val)
    z = reparametrizar(mu, logvar)
    _, recon, kl = perdida_por_imagen(x_val, model.decode(z), mu, logvar, beta)
    return recon.mean().item(), kl.mean().item()


def entrenar(beta, x_train, x_val):
    set_seed()
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist = {"recon": [], "kl": []}
    n = x_train.shape[0]
    for epoca in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            x = x_train[perm[i:i + BATCH_SIZE]]
            mu, logvar = model.encode(x)
            z = reparametrizar(mu, logvar)
            loss, _, _ = perdida_por_imagen(x, model.decode(z), mu, logvar, beta)
            opt.zero_grad()
            loss.mean().backward()
            opt.step()
        recon, kl = evaluar(model, x_val, beta)
        hist["recon"].append(recon)
        hist["kl"].append(kl)
        print(f"beta={beta:<5} epoca {epoca:2d}/{EPOCHS}  val recon={recon:8.3f}  KL={kl:7.3f}", flush=True)
    return model, hist


def nitidez(x):
    """Media de |diferencia| entre pixeles horizontalmente adyacentes."""
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean().item()


def log_normal(z, mu, sigma):
    """Log-densidad de una gaussiana diagonal, sumada sobre dimensiones."""
    return torch.sum(-0.5 * math.log(2 * math.pi) - torch.log(sigma)
                     - 0.5 * ((z - mu) / sigma) ** 2, dim=1)


def verificar_kl_mc(model, x_val, y_val, indice, n_muestras=100_000):
    """Compara la KL cerrada con la estimacion Monte Carlo (en float64)."""
    with torch.no_grad():
        mu, logvar = model.encode(x_val[indice:indice + 1])
    mu, logvar = mu.double(), logvar.double()
    sigma = torch.exp(0.5 * logvar)
    cerrada = kl_cerrada(mu, logvar).item()
    z = mu + sigma * torch.randn(n_muestras, LATENT_DIM, dtype=torch.float64)
    muestras = log_normal(z, mu, sigma) - log_normal(z, torch.zeros_like(mu), torch.ones_like(sigma))
    mc = muestras.mean().item()
    ee = (muestras.std() / math.sqrt(n_muestras)).item()
    return {
        "indice_imagen_validacion": int(indice),
        "clase": CLASS_NAMES[int(y_val[indice])],
        "mu": mu.squeeze().tolist(),
        "sigma": sigma.squeeze().tolist(),
        "n_muestras": n_muestras,
        "kl_cerrada": cerrada,
        "kl_montecarlo": mc,
        "error_relativo": abs(mc - cerrada) / abs(cerrada),
        "error_estandar_mc": ee,
        "errores_estandar_de_distancia": abs(mc - cerrada) / ee,
    }


def mostrar_cuadricula(ax, imgs, filas, cols):
    """Pega imgs (N,1,28,28) en una cuadricula filas x cols."""
    a = imgs.squeeze(1).reshape(filas, cols, 28, 28).permute(0, 2, 1, 3).reshape(filas * 28, cols * 28)
    ax.imshow(a.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def main():
    t0 = time.time()
    set_seed()
    x_train, y_train, x_val, y_val, _, _ = common.load_fashion_mnist("unit")
    print(f"train={tuple(x_train.shape)} val={tuple(x_val.shape)}")

    modelos, curvas = {}, {}
    t_ent = time.time()
    for beta in BETA_VALUES:
        modelos[beta], curvas[beta] = entrenar(beta, x_train, x_val)
    tiempo_entrenamiento = time.time() - t_ent

    # Fase de generacion/evaluacion con semilla propia y reproducible
    set_seed()
    epocas = np.arange(1, EPOCHS + 1)

    # Figura 1.2: curvas de validacion
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for j, beta in enumerate(BETA_VALUES):
        axs[0, j].plot(epocas, curvas[beta]["recon"], "o-", color="tab:blue")
        axs[0, j].set_title(f"beta = {beta}")
        axs[1, j].plot(epocas, curvas[beta]["kl"], "s-", color="tab:red")
        axs[1, j].set_xlabel("Epoca")
    axs[0, 0].set_ylabel("Reconstruccion (error cuadratico, suma 784 px)")
    axs[1, 0].set_ylabel("KL (nats por imagen)")
    fig.suptitle("Metricas de validacion por epoca (fila superior: reconstruccion; inferior: KL)")
    fig.tight_layout()
    fig.savefig(figure_path("task1_2_curvas_validacion.png"), dpi=130)
    plt.close(fig)

    # Codificacion de validacion (medias)
    mus = {}
    with torch.no_grad():
        for beta, m in modelos.items():
            m.eval()
            mus[beta] = m.encode(x_val)[0]

    # Figura 1.3a: scatter latente
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8))
    cmap = plt.get_cmap("tab10")
    for ax, beta in zip(axs, BETA_VALUES):
        mu = mus[beta].numpy()
        for c in range(10):
            sel = (y_val == c).numpy()
            ax.scatter(mu[sel, 0], mu[sel, 1], s=5, color=cmap(c), label=CLASS_NAMES[c], alpha=0.7)
        ax.set_title(f"beta = {beta}")
        ax.set_xlabel("mu_1")
        ax.set_ylabel("mu_2")
        ax.grid(alpha=0.3)
    axs[-1].legend(markerscale=3, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Medias mu del encoder sobre las 5000 imagenes de validacion")
    fig.tight_layout()
    fig.savefig(figure_path("task1_3a_scatter_latente.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Figura 1.3b: cuadricula 15x15 (beta = 1); z2 crece hacia arriba
    m1 = modelos[1.0]
    eje = torch.linspace(-3, 3, 15)
    z1, z2 = torch.meshgrid(eje, eje.flip(0), indexing="xy")
    zg = torch.stack([z1.reshape(-1), z2.reshape(-1)], dim=1)
    with torch.no_grad():
        grid = m1.decode(zg)
    fig, ax = plt.subplots(figsize=(10, 10))
    mostrar_cuadricula(ax, grid, 15, 15)
    ax.set_title("beta = 1: decodificacion en rejilla 15x15, z1 (izq->der) y z2 (abajo->arriba) en [-3, 3]")
    fig.tight_layout()
    fig.savefig(figure_path("task1_3b_grid_beta1.png"), dpi=130)
    plt.close(fig)

    # Figura 1.3c: 64 muestras z ~ N(0, I) por beta
    for beta in BETA_VALUES:
        with torch.no_grad():
            s = modelos[beta].decode(torch.randn(64, LATENT_DIM))
        fig, ax = plt.subplots(figsize=(7, 7.4))
        mostrar_cuadricula(ax, s, 8, 8)
        ax.set_title(f"beta = {beta}: 64 muestras con z ~ N(0, I)")
        fig.tight_layout()
        fig.savefig(figure_path(f"task1_3c_muestras_beta{beta}.png"), dpi=130)
        plt.close(fig)

    # Figura 1.3d: interpolacion entre dos clases distintas (beta = 1)
    clase_a, clase_b = 0, 9
    ia = int((y_val == clase_a).nonzero()[0])
    ib = int((y_val == clase_b).nonzero()[0])
    xa, xb = x_val[ia:ia + 1], x_val[ib:ib + 1]
    alphas = torch.linspace(0, 1, 10)  # extremos + 8 puntos intermedios

    with torch.no_grad():
        mu_a, mu_b = m1.encode(xa)[0], m1.encode(xb)[0]

        def interp_pixeles(a):
            return (1 - a) * xa + a * xb

        def interp_latente(a):
            return m1.decode((1 - a) * mu_a + a * mu_b)

        fila_pix = torch.cat([interp_pixeles(a) for a in alphas])
        fila_lat = torch.cat([interp_latente(a) for a in alphas])
        medio_pix = interp_pixeles(0.5)
        medio_lat = interp_latente(0.5)
    fig, axs = plt.subplots(2, 1, figsize=(13, 3.6))
    for ax, fila, titulo in zip(
        axs, (fila_pix, fila_lat),
        ("Interpolacion en espacio de pixeles", "Interpolacion en espacio latente (sobre mu, decodificada)")):
        mostrar_cuadricula(ax, fila, 1, 10)
        ax.set_title(titulo, fontsize=10)
    fig.suptitle(f"beta = 1: {CLASS_NAMES[clase_a]} (izq., idx {ia}) -> {CLASS_NAMES[clase_b]} (der., idx {ib}); "
                 f"alpha = 0 ... 1 en 10 columnas (8 intermedios)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figure_path("task1_3d_interpolacion_beta1.png"), dpi=130)
    plt.close(fig)

    # 1.4a: tabla final
    tabla = {}
    for beta in BETA_VALUES:
        r, k = curvas[beta]["recon"][-1], curvas[beta]["kl"][-1]
        tabla[str(beta)] = {"recon_final": r, "kl_nats_final": k, "kl_bits_final": k / math.log(2)}

    # 1.4b: verificacion de la KL
    verif = verificar_kl_mc(m1, x_val, y_val, ia)

    # 1.4c: nitidez
    nit = {"nitidez_real": nitidez(x_val[:1000])}
    for beta in BETA_VALUES:
        with torch.no_grad():
            gen = modelos[beta].decode(torch.randn(1000, LATENT_DIM))
        nit[f"nitidez_gen_{beta}"] = nitidez(gen)

    # 1.4d: extras
    extras = {"norma_media_mu": {}, "fraccion_esquinas": {}, "fraccion_fuera_de_rejilla": {},
              "definicion_esquinas": "puntos con |mu1|>2 y |mu2|>2 (cuatro cuadrados de 1x1 en las esquinas de [-3,3]^2)"}
    for beta in BETA_VALUES:
        mu = mus[beta]
        extras["norma_media_mu"][str(beta)] = mu.norm(dim=1).mean().item()
        extras["fraccion_esquinas"][str(beta)] = ((mu.abs() > 2).all(dim=1)).float().mean().item()
        extras["fraccion_fuera_de_rejilla"][str(beta)] = ((mu.abs() > 3).any(dim=1)).float().mean().item()
    extras["nitidez_medio_pixeles"] = nitidez(medio_pix)
    extras["nitidez_medio_latente"] = nitidez(medio_lat)
    extras["interpolacion"] = {"idx_a": ia, "clase_a": CLASS_NAMES[clase_a],
                               "idx_b": ib, "clase_b": CLASS_NAMES[clase_b], "alpha_medio": 0.5}

    payload = {
        "seed": common.SEED,
        "config": {"batch_size": BATCH_SIZE, "latent_dim": LATENT_DIM, "epochs": EPOCHS, "lr": LR,
                   "beta_values": BETA_VALUES, "arquitectura": ARQUITECTURA,
                   "nota_validacion": "recon y KL de validacion medias por imagen, con z muestreado por reparametrizacion"},
        "curvas": {str(b): curvas[b] for b in BETA_VALUES},
        "tabla_1_4a": tabla,
        "verificacion_kl_1_4b": verif,
        "nitidez_1_4c": nit,
        "extras_1_4d": extras,
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
    }
    ruta = save_results("task1_results.json", payload)

    print("\n=== RESUMEN ===")
    print("1.4a beta | recon | KL nats | KL bits")
    for b, v in tabla.items():
        print(f"  {b:>5} | {v['recon_final']:.4f} | {v['kl_nats_final']:.4f} | {v['kl_bits_final']:.4f}")
    print("1.4b", verif)
    print("1.4c", nit)
    print("1.4d", extras)
    print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.1f} s; total {time.time() - t0:.1f} s")
    print("JSON:", ruta)


if __name__ == "__main__":
    main()
