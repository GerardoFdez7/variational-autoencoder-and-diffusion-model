"""Task 2: proceso forward de difusion sobre Fashion-MNIST (sin entrenar redes)."""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

T = 1000
BETA_START = 1e-4
BETA_END = 0.02


# ---------------------------------------------------------------- calendarios
def linear_schedule(T, beta_start, beta_end):
    """Calendario lineal. Devuelve betas, alphas y alpha_bar con indice 0 <-> t=1."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


# Calendario propio.
# Base: calendario coseno de Nichol & Dhariwal (2021), "Improved Denoising
# Diffusion Probabilistic Models", ICML: alpha_bar_t = f(t)/f(0),
# f(t) = cos^2(((t/T + s)/(1 + s)) * pi/2), con s = 0.008.
# Modificaciones: (1) offset s = 0.02 en lugar de 0.008, lo que suaviza el
# inicio y reduce el beta inicial; (2) exponente p = 3 en lugar de 2, es decir
# f(t) = cos^3(...), que acelera la caida de alpha_bar y garantiza
# alpha_bar_T < 1e-3 sin depender solo del recorte de betas; (3) betas
# recortados a [1e-5, 0.999] y alpha_bar recalculado como cumprod de los betas
# recortados, de modo que beta_t en (0,1) y la relacion alpha_bar = prod alpha_s
# se cumplen exactamente.
COS_OFFSET = 0.02
COS_POWER = 3.0
BETA_MIN_CLIP = 1e-5
BETA_MAX_CLIP = 0.999


def modified_cosine_schedule(T, s=COS_OFFSET, power=COS_POWER):
    steps = torch.arange(0, T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** power
    alpha_bar_raw = f / f[0]
    betas = (1.0 - alpha_bar_raw[1:] / alpha_bar_raw[:-1]).clamp(BETA_MIN_CLIP, BETA_MAX_CLIP)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


def snr_from_alpha_bar(alpha_bar):
    return alpha_bar / (1.0 - alpha_bar)


def first_t(mask):
    """Primer t (base 1) donde mask es True, o None."""
    idx = torch.nonzero(mask)
    return int(idx[0].item()) + 1 if len(idx) else None


# ------------------------------------------------------------------ forward
def extract(coef, t):
    """Toma coef[t-1] por elemento del lote y lo reforma a (B,1,1,1). t en 1..T."""
    return coef[t - 1].float().reshape(-1, 1, 1, 1)


def q_sample(x0, t, eps, alpha_bar):
    """x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps (t: vector de indices 1..T)."""
    ab = extract(alpha_bar, t)
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps


def main():
    common.set_seed()
    common.ensure_dirs()
    x_train, y_train, *_ = common.load_fashion_mnist(scale="symmetric")
    results = {"seed": common.SEED,
               "config": {"T": T, "beta_start": BETA_START, "beta_end": BETA_END}}

    betas, alphas, alpha_bar = linear_schedule(T, BETA_START, BETA_END)
    snr = snr_from_alpha_bar(alpha_bar)
    ts = torch.arange(1, T + 1)

    # ------------------------------------------------------------- Task 2.2
    t_check, n_copies = 300, 5000
    x0 = x_train[0:1]                                  # imagen fija (1,1,28,28)
    x = x0.repeat(n_copies, 1, 1, 1)
    for s in range(t_check):
        eps_s = torch.randn_like(x)
        x = math.sqrt(alphas[s].item()) * x + math.sqrt(betas[s].item()) * eps_s
    emp_mean = x.mean(dim=0)[0]
    emp_var = x.var(dim=0, unbiased=True)[0]
    ab300 = alpha_bar[t_check - 1].item()
    theo_mean = math.sqrt(ab300) * x0[0, 0]
    theo_var = 1.0 - ab300
    err_mean = (emp_mean - theo_mean).abs()
    std_err = math.sqrt(1.0 - ab300) / math.sqrt(n_copies)
    n_pix = x0[0, 0].numel()
    results["verificacion_2_2"] = {
        "indice_imagen_x0": 0, "clase_x0": common.CLASS_NAMES[int(y_train[0])],
        "t": t_check, "n_copias": n_copies,
        "alpha_bar_300": ab300,
        "media_teorica_formula": "sqrt(alpha_bar_300) * x0",
        "varianza_teorica": theo_var,
        "error_max_abs_media": err_mean.max().item(),
        "error_medio_abs_media": err_mean.mean().item(),
        "varianza_empirica_promedio": emp_var.mean().item(),
        "varianza_empirica_min": emp_var.min().item(),
        "varianza_empirica_max": emp_var.max().item(),
        "error_abs_varianza_promedio": emp_var.mean().item() - theo_var,
        "error_max_abs_varianza": (emp_var - theo_var).abs().max().item(),
        "error_estandar_esperado": std_err,
        "cociente_error_max_sobre_error_estandar": err_mean.max().item() / std_err,
        "cociente_error_medio_sobre_error_estandar": err_mean.mean().item() / std_err,
        "n_pixeles": n_pix,
        "max_esperado_gaussiano_desv_std": math.sqrt(2 * math.log(n_pix)),
    }

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    paneles = [(emp_mean, "Media empirica (t=300)", "gray", (-1, 1)),
               (theo_mean, "Media teorica", "gray", (-1, 1)),
               (err_mean, "|error de la media|", "magma", (0, err_mean.max().item())),
               (emp_var, "Varianza empirica por pixel", "viridis", (0, 1.2 * theo_var)),
               (torch.full_like(emp_var, theo_var), "Varianza teorica (1 - alpha_bar)",
                "viridis", (0, 1.2 * theo_var))]
    for a, (img, titulo, cmap, (lo, hi)) in zip(ax.flat[:5], paneles):
        im = a.imshow(img.numpy(), cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(titulo, fontsize=10)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    ax.flat[5].hist((emp_mean - theo_mean).flatten().numpy() / std_err, bins=40, density=True,
                    color="tab:blue", alpha=0.7, label="error / error estandar")
    zz = torch.linspace(-4, 4, 200)
    ax.flat[5].plot(zz, torch.exp(-zz ** 2 / 2) / math.sqrt(2 * math.pi), "k", label="N(0,1)")
    ax.flat[5].set_title("Error de la media en unidades de error estandar", fontsize=10)
    ax.flat[5].legend()
    fig.suptitle("Verificacion empirica de q(x_t | x_0) en t=300 (5000 copias)")
    fig.tight_layout()
    fig.savefig(common.figure_path("task2_2_verificacion.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------ Task 2.3a
    t_show = [1, 50, 100, 250, 500, 750, 1000]
    class_ids = [1, 3, 9]                              # Trouser, Dress, Ankle boot
    fig, ax = plt.subplots(3, 7, figsize=(14, 6.4))
    for r, c in enumerate(class_ids):
        xi = x_train[(y_train == c).nonzero()[0, 0]].unsqueeze(0)
        eps = torch.randn(1, 1, 28, 28)
        for k, tt in enumerate(t_show):
            xt = q_sample(xi, torch.tensor([tt]), eps, alpha_bar)
            ax[r, k].imshow(xt[0, 0].clamp(-1, 1).numpy(), cmap="gray", vmin=-1, vmax=1)
            ax[r, k].set_xticks([])
            ax[r, k].set_yticks([])
            if r == 0:
                ax[r, k].set_title(f"t = {tt}")
        ax[r, 0].set_ylabel(common.CLASS_NAMES[c], fontsize=11)
    fig.suptitle("Proceso forward (calendario lineal)")
    fig.tight_layout()
    fig.savefig(common.figure_path("task2_3a_forward_visual.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------ Task 2.3b/c
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(ts, alpha_bar)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$\bar\alpha_t$")
    ax[0].set_title(r"$\bar\alpha_t$ (lineal)")
    ax[0].grid(alpha=0.3)
    ax[1].semilogy(ts, snr)
    ax[1].axhline(1, color="gray", ls="--", lw=0.8)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("SNR(t)")
    ax[1].set_title(r"SNR$(t)=\bar\alpha_t/(1-\bar\alpha_t)$ (escala log)")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(common.figure_path("task2_3b_alphabar_snr.png"), dpi=130)
    plt.close(fig)

    t_snr1_lin = first_t(snr < 1)
    results["hitos_2_3c"] = {
        "primer_t_snr_menor_1": t_snr1_lin,
        "primer_t_alphabar_menor_0.01": first_t(alpha_bar < 0.01),
        "alpha_bar_en_hito_snr": alpha_bar[t_snr1_lin - 1].item(),
        "alpha_bar_T": alpha_bar[-1].item(),
        "snr_T": snr[-1].item(),
    }

    # --------------------------------------------------------------- Task 2.4
    betas2, alphas2, alpha_bar2 = modified_cosine_schedule(T)
    snr2 = snr_from_alpha_bar(alpha_bar2)
    cond_ab = bool(alpha_bar2[-1].item() < 1e-3)
    cond_beta = bool(((betas2 > 0) & (betas2 < 1)).all().item())
    cond_beta_lin = bool(((betas > 0) & (betas < 1)).all().item())
    results["calendario_propio_2_4"] = {
        "nombre": "Coseno modificado (offset 0.02, exponente 3)",
        "formula": "alpha_bar_t = f(t)/f(0), f(t) = cos^3(((t/T + s)/(1+s)) * pi/2), s=0.02; "
                   "beta_t = clip(1 - alpha_bar_t/alpha_bar_{t-1}, 1e-5, 0.999); "
                   "alpha_bar final = cumprod(1 - beta_t)",
        "cita": "Nichol & Dhariwal (2021), Improved Denoising Diffusion Probabilistic Models, ICML",
        "modificacion": "Offset s 0.008 -> 0.02; exponente 2 -> 3 (cos^3); betas recortados a "
                        "[1e-5, 0.999] con alpha_bar recalculado por cumprod.",
        "parametros": {"s": COS_OFFSET, "exponente": COS_POWER,
                       "beta_min_clip": BETA_MIN_CLIP, "beta_max_clip": BETA_MAX_CLIP},
        "verificacion_condiciones": {
            "alpha_bar_T_menor_1e-3": cond_ab,
            "beta_en_(0,1)_para_todo_t": cond_beta,
            "beta_min": betas2.min().item(), "beta_max": betas2.max().item(),
            "alpha_bar_T": alpha_bar2[-1].item()},
        "lineal": {"alpha_bar_T": alpha_bar[-1].item(), "primer_t_snr_menor_1": t_snr1_lin,
                   "beta_en_(0,1)": cond_beta_lin},
        "propio": {"alpha_bar_T": alpha_bar2[-1].item(),
                   "primer_t_snr_menor_1": first_t(snr2 < 1),
                   "primer_t_alphabar_menor_0.01": first_t(alpha_bar2 < 0.01)},
    }
    assert cond_ab and cond_beta, "El calendario propio no cumple las condiciones"

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogy(ts, snr, label="Lineal")
    ax.semilogy(ts, snr2, label="Coseno modificado (propio)")
    ax.axhline(1, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("SNR(t)")
    ax.set_title("Comparacion de SNR (escala log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(common.figure_path("task2_4_comparacion_snr.png"), dpi=130)
    plt.close(fig)

    # -------------------------------------------------------------- Task 2.5a
    suma_betas = T * (BETA_START + BETA_END) / 2
    suma_cuad = T * (BETA_END ** 3 - BETA_START ** 3) / (3 * (BETA_END - BETA_START))
    log_ab = -suma_betas - 0.5 * suma_cuad
    ab_aprox = math.exp(log_ab)
    ab_num = alpha_bar[-1].item()
    results["calculo_2_5a"] = {
        "T": T, "beta_start": BETA_START, "beta_end": BETA_END,
        "suma_betas": {"formula": "T*(beta_start+beta_end)/2",
                       "beta_start_mas_beta_end": BETA_START + BETA_END,
                       "T_por_suma": T * (BETA_START + BETA_END),
                       "valor": suma_betas,
                       "valor_numerico_torch": betas.sum().item()},
        "suma_betas_cuadrado": {"formula": "T*(beta_end^3 - beta_start^3)/(3*(beta_end - beta_start))",
                                "beta_end_cubo": BETA_END ** 3,
                                "beta_start_cubo": BETA_START ** 3,
                                "numerador_T_por_diferencia_cubos": T * (BETA_END ** 3 - BETA_START ** 3),
                                "denominador_3_por_diferencia": 3 * (BETA_END - BETA_START),
                                "valor": suma_cuad,
                                "valor_numerico_torch": (betas ** 2).sum().item()},
        "mitad_suma_cuadrados": 0.5 * suma_cuad,
        "log_alphabar_T_aprox": log_ab,
        "log_alphabar_T_numerico": torch.log(alpha_bar[-1]).item(),
        "alphabar_T_aprox": ab_aprox,
        "alphabar_T_numerico": ab_num,
        "error_relativo": abs(ab_aprox - ab_num) / ab_num,
    }

    # -------------------------------------------------------------- Task 2.5b
    def contar(snr_vec):
        return {"en_rango_0.01_100": int(((snr_vec >= 0.01) & (snr_vec <= 100)).sum()),
                "snr_menor_0.01": int((snr_vec < 0.01).sum()),
                "snr_mayor_100": int((snr_vec > 100).sum())}
    results["conteo_snr_2_5b"] = {"lineal": contar(snr), "propio": contar(snr2), "T": T}

    # -------------------------------------------------------------- Task 2.5c
    var_x0 = x_train.var(unbiased=False).item()
    subconjunto = x_train[torch.randperm(x_train.shape[0])[:10000]]
    filas = []
    for tt in [1, 250, 500, 1000]:
        xt = q_sample(subconjunto, torch.full((subconjunto.shape[0],), tt),
                      torch.randn_like(subconjunto), alpha_bar)
        v_emp = xt.var(unbiased=False).item()
        ab_t = alpha_bar[tt - 1].item()
        v_pred = ab_t * var_x0 + (1 - ab_t)
        filas.append({"t": tt, "varianza_empirica": v_emp, "varianza_predicha": v_pred,
                      "error_abs": abs(v_emp - v_pred),
                      "error_relativo": abs(v_emp - v_pred) / v_pred})
    results["varianza_2_5c"] = {"var_x0": var_x0, "n_imagenes_subconjunto": 10000,
                                "var_x_T": filas[-1]["varianza_empirica"], "tabla": filas}

    path = common.save_results("task2_results.json", results)

    # ---------------------------------------------------------------- resumen
    v = results["verificacion_2_2"]
    print("=== 2.2 ===")
    for k in ["alpha_bar_300", "varianza_teorica", "error_max_abs_media", "error_medio_abs_media",
              "varianza_empirica_promedio", "error_estandar_esperado",
              "cociente_error_max_sobre_error_estandar", "cociente_error_medio_sobre_error_estandar",
              "max_esperado_gaussiano_desv_std"]:
        print(f"  {k}: {v[k]:.8g}")
    print("=== 2.3c ===", results["hitos_2_3c"])
    print("=== 2.4 ===", results["calendario_propio_2_4"]["verificacion_condiciones"])
    print("  lineal:", results["calendario_propio_2_4"]["lineal"])
    print("  propio:", results["calendario_propio_2_4"]["propio"])
    print("=== 2.5a ===")
    for k, val in results["calculo_2_5a"].items():
        print(f"  {k}: {val}")
    print("=== 2.5b ===", results["conteo_snr_2_5b"])
    print("=== 2.5c === var_x0 =", var_x0)
    for f in filas:
        print("  ", f)
    print("JSON:", path)


if __name__ == "__main__":
    main()
