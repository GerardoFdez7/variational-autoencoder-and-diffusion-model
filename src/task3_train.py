"""Task 3.2: entrenamiento del modelo de difusion condicional (prediccion de ruido)."""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from task2_diffusion import T, BETA_START, BETA_END, linear_schedule, q_sample
from task3_unet import UNet, NULL_CLASS, count_params

EPOCHS = 30
BATCH_SIZE = 128
LR = 2e-4
P_UNCOND = 0.1
N_INTERVALS = 10                       # intervalos de t de 100 pasos
CKPT_PATH = os.path.join(common.RESULTS_DIR, "checkpoints", "unet_cond.pt")


def train_step(model, opt, x0, y, alpha_bar):
    """Un paso de entrenamiento.

    Para cada imagen: t ~ U{1..T}, eps ~ N(0,I), x_t = q_sample(x0, t, eps) (Task 2.1),
    clase -> condicion nula (10) con probabilidad p_uncond, y se minimiza
    L = E ||eps - eps_theta(x_t, t, c)||^2 (MSE promedio sobre pixeles y lote).
    """
    B = x0.shape[0]
    t = torch.randint(1, T + 1, (B,))
    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, eps, alpha_bar)
    c = torch.where(torch.rand(B) < P_UNCOND, torch.full_like(y, NULL_CLASS), y)
    loss = F.mse_loss(model(xt, t, c), eps)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def val_loss_per_interval(model, x_val, y_val, alpha_bar, seed=0, chunk=1000):
    """Perdida MSE promedio en cada intervalo de t de 100 pasos sobre la validacion.

    Para cada imagen se sortea un t uniforme dentro del intervalo y un eps nuevo
    (generador fijo => los dos modelos/condiciones son comparables). Se reporta
    con la clase real (condicional) y con la condicion nula.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    cond, uncond = [], []
    N = x_val.shape[0]
    for k in range(N_INTERVALS):
        lo, hi = k * (T // N_INTERVALS) + 1, (k + 1) * (T // N_INTERVALS)
        se_c = se_u = 0.0
        for i in range(0, N, chunk):
            x0, y = x_val[i:i + chunk], y_val[i:i + chunk]
            t = torch.randint(lo, hi + 1, (x0.shape[0],), generator=g)
            eps = torch.randn(x0.shape, generator=g)
            xt = q_sample(x0, t, eps, alpha_bar)
            se_c += F.mse_loss(model(xt, t, y), eps, reduction="sum").item()
            se_u += F.mse_loss(model(xt, t, torch.full_like(y, NULL_CLASS)), eps,
                               reduction="sum").item()
        cond.append(se_c / (N * 784))
        uncond.append(se_u / (N * 784))
    return cond, uncond


def main():
    common.set_seed()
    common.ensure_dirs()
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    x_train, y_train, x_val, y_val, *_ = common.load_fashion_mnist(scale="symmetric")
    _, _, alpha_bar = linear_schedule(T, BETA_START, BETA_END)

    model = UNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history, start = [], 0
    if os.path.exists(CKPT_PATH) and "--fresh" not in sys.argv:   # reanuda si se interrumpio
        ck = torch.load(CKPT_PATH)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        history, start = ck["history"], ck["epoch"]
        torch.set_rng_state(ck["rng"])
        print(f"Reanudando desde la epoca {start}")

    N = x_train.shape[0]
    print(f"Parametros: {count_params(model)} | train: {N} | hilos: {torch.get_num_threads()}")
    for epoch in range(start, EPOCHS):
        model.train()
        t0 = time.time()
        perm = torch.randperm(N)
        total, n = 0.0, 0
        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            b = idx.shape[0]
            total += train_step(model, opt, x_train[idx], y_train[idx], alpha_bar) * b
            n += b
        history.append(total / n)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                    "history": history, "rng": torch.get_rng_state()}, CKPT_PATH)
        print(f"epoca {epoch + 1:2d}/{EPOCHS}  loss={history[-1]:.5f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------ curva de entrenamiento
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(1, EPOCHS + 1), history, marker="o", ms=3)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE de entrenamiento (eps)")
    ax.set_title("Perdida de entrenamiento por epoca")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(common.figure_path("task3_2_loss_entrenamiento.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------- perdida de validacion por intervalo de t
    cond, uncond = val_loss_per_interval(model, x_val, y_val, alpha_bar)
    labels = [f"{k * 100 + 1}-{(k + 1) * 100}" for k in range(N_INTERVALS)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(labels, cond, marker="o", label="condicional (clase real)")
    ax.plot(labels, uncond, marker="s", ls="--", label="condicion nula")
    ax.set_yscale("log")
    ax.set_xlabel("Intervalo de t")
    ax.set_ylabel("MSE de validacion (eps)")
    ax.set_title("Perdida de validacion por intervalo de t (5000 imagenes)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    fig.savefig(common.figure_path("task3_2_loss_por_intervalo_t.png"), dpi=130)
    plt.close(fig)

    results = {"seed": common.SEED, "parametros": count_params(model),
               "config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                          "p_uncond": P_UNCOND, "T": T, "beta_start": BETA_START,
                          "beta_end": BETA_END, "optimizador": "Adam"},
               "loss_entrenamiento_por_epoca": history,
               "val_loss_por_intervalo_t": {"intervalos": labels, "condicional": cond,
                                            "condicion_nula": uncond,
                                            "n_imagenes_val": int(x_val.shape[0])},
               "intervalo_mayor_perdida": labels[max(range(N_INTERVALS), key=cond.__getitem__)],
               "intervalo_menor_perdida": labels[min(range(N_INTERVALS), key=cond.__getitem__)]}
    path = common.save_results("task3_2_results.json", results)
    print("Loss final de entrenamiento:", history[-1])
    for lab, c, u in zip(labels, cond, uncond):
        print(f"  t in [{lab:>9}]  cond={c:.5f}  nula={u:.5f}")
    print("JSON:", path)


if __name__ == "__main__":
    main()
