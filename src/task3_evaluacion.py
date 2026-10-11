"""Task 3.4(c-d) y mediciones de la 3.5: clasificador, fidelidad, diversidad y confusion.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import task3_sampling as ts
from task2_diffusion import T, BETA_START, BETA_END, linear_schedule, snr_from_alpha_bar

NUM_CLASSES = ts.NUM_CLASSES
N_POR_CLASE = ts.N_EVAL_PER_CLASS              # 50
CLF_CKPT = os.path.join(common.RESULTS_DIR, "checkpoints", "clasificador.pt")
CLF_EPOCHS = 5
CLF_BATCH = 128
CLF_LR = 1e-3
CLF_OBJETIVO = 0.88
CLASE_FIGURA = 0                               # T-shirt/top: clase elegida para la figura 3.4(d)
N_FIGURA = 10


class Clasificador(nn.Module):
    """CNN sencilla: dos bloques conv-ReLU-maxpool y dos capas lineales."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))    # 14 -> 7
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def predecir(model, x, chunk=1000):
    """Probabilidades softmax (N, 10) para imagenes en [-1, 1]."""
    model.eval()
    return torch.cat([F.softmax(model(x[i:i + chunk]), dim=1) for i in range(0, len(x), chunk)])


def exactitud(model, x, y):
    return float((predecir(model, x).argmax(1) == y).float().mean())


def obtener_clasificador(x_train, y_train, x_val, y_val):
    """Carga el clasificador guardado o lo entrena. Devuelve (modelo, historial)."""
    model = Clasificador()
    if os.path.exists(CLF_CKPT):
        ck = torch.load(CLF_CKPT, weights_only=True)
        model.load_state_dict(ck["model"])
        model.eval()
        return model, ck["historial"]
    common.set_seed()
    opt = torch.optim.Adam(model.parameters(), lr=CLF_LR)
    hist = {"loss_entrenamiento": [], "exactitud_validacion": []}
    n = x_train.shape[0]
    for epoca in range(1, CLF_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, CLF_BATCH):
            idx = perm[i:i + CLF_BATCH]
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.shape[0]
        hist["loss_entrenamiento"].append(total / n)
        hist["exactitud_validacion"].append(exactitud(model, x_val, y_val))
        print(f"clasificador epoca {epoca}/{CLF_EPOCHS}  loss={total / n:.4f}  "
              f"val acc={hist['exactitud_validacion'][-1]:.4f}", flush=True)
    if hist["exactitud_validacion"][-1] < CLF_OBJETIVO:
        raise RuntimeError(f"El clasificador no alcanzo {CLF_OBJETIVO:.0%} en validacion.")
    os.makedirs(os.path.dirname(CLF_CKPT), exist_ok=True)
    torch.save({"model": model.state_dict(), "historial": hist}, CLF_CKPT)
    model.eval()
    return model, hist


def diversidad_por_clase(images, labels):
    """Distancia euclidiana promedio entre todos los pares de imagenes de cada clase."""
    return [float(torch.pdist(images[labels == c].flatten(1).double()).mean())
            for c in range(NUM_CLASSES)]


def matriz_confusion(y_true, y_pred):
    """Filas: clase pedida/real; columnas: clase predicha. Conteos."""
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for a, b in zip(y_true.tolist(), y_pred.tolist()):
        cm[a, b] += 1
    return cm


def par_mas_confundido(cm):
    """Par (i, j), i < j, con mayor cm[i, j] + cm[j, i] (normalizado por fila)."""
    tasa = cm.double() / cm.sum(1, keepdim=True).clamp(min=1)
    mejor, par = -1.0, None
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            s = float(tasa[i, j] + tasa[j, i])
            if s > mejor:
                mejor, par = s, (i, j)
    return par, mejor


def fila_imagenes(axes, imgs, titulo):
    for k, ax in enumerate(axes):
        if k < len(imgs):
            ax.imshow(imgs[k, 0], cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(k < len(imgs))
    axes[0].set_ylabel(titulo, rotation=0, ha="right", va="center", fontsize=8)


def main():
    common.set_seed()
    common.ensure_dirs()
    x_train, y_train, x_val, y_val, *_ = common.load_fashion_mnist(scale="symmetric")
    names = common.CLASS_NAMES

    # ---------------------------------------------------- 3.4(c): clasificador
    clf, hist = obtener_clasificador(x_train, y_train, x_val, y_val)
    probs_val = predecir(clf, x_val)
    acc_val = float((probs_val.argmax(1) == y_val).float().mean())
    print(f"Exactitud del clasificador en validacion: {acc_val:.4f}")

    # Referencia real: primeras 50 imagenes de cada clase en validacion (igual que 3.4).
    idx_real = torch.cat([torch.nonzero(y_val == c)[:N_POR_CLASE, 0] for c in range(NUM_CLASSES)])
    x_real, y_real = x_val[idx_real], y_val[idx_real]
    div_real = diversidad_por_clase(x_real, y_real)
    fid_real = float((predecir(clf, x_real).argmax(1) == y_real).float().mean())

    tiempos = json.load(open(os.path.join(common.RESULTS_DIR, "task3_3_4_results.json"),
                             encoding="utf-8"))
    dispositivo_tiempos = tiempos["dispositivo"]
    tiempos = tiempos["task3_4"]["tiempo_por_100_imagenes"]

    muestras, por_w = {}, {}
    for w in ts.GUIDANCE_WS:
        data = torch.load(ts.samples_path(w), weights_only=True)
        imgs, labs = data["images"], data["labels"]
        probs = predecir(clf, imgs)
        pred = probs.argmax(1)
        div = diversidad_por_clase(imgs, labs)
        muestras[w] = (imgs, labs, pred)
        por_w[w] = {
            "fidelidad": float((pred == labs).float().mean()),
            "diversidad": float(np.mean(div)),
            "segundos_100_imagenes": tiempos[str(w)]["segundos_100_imagenes"],
            "evaluaciones_red_100_imagenes": tiempos[str(w)]["evaluaciones_red_100_imagenes"],
            "fidelidad_por_clase": {names[c]: float((pred[labs == c] == c).float().mean())
                                    for c in range(NUM_CLASSES)},
            "diversidad_por_clase": dict(zip(names, div)),
            "prob_media_clase_pedida": {names[c]: float(probs[labs == c, c].mean())
                                        for c in range(NUM_CLASSES)},
            "frac_pixeles_blancos_por_clase": {
                names[c]: float((imgs[labs == c] >= 0.99).float().mean())
                for c in range(NUM_CLASSES)},
            "intensidad_media_por_clase": {names[c]: float(imgs[labs == c].mean())
                                           for c in range(NUM_CLASSES)},
        }

    print(f"\n{'w':>4} {'fidelidad':>10} {'diversidad':>11} {'s/100 img':>10}")
    for w, r in por_w.items():
        print(f"{w:>4} {r['fidelidad']:>10.3f} {r['diversidad']:>11.3f} "
              f"{r['segundos_100_imagenes']:>10.2f}")
    print(f"real {fid_real:>10.3f} {np.mean(div_real):>11.3f}")

    # --------------------------- 3.4(d): figura de una clase para cada w (+ reales)
    fig, axes = plt.subplots(len(ts.GUIDANCE_WS) + 1, N_FIGURA,
                             figsize=(N_FIGURA * 0.9 + 1.4, 4.4))
    for r, w in enumerate(ts.GUIDANCE_WS):
        imgs, labs, _ = muestras[w]
        fila_imagenes(axes[r], imgs[labs == CLASE_FIGURA][:N_FIGURA], f"w = {w}")
    fila_imagenes(axes[-1], x_real[y_real == CLASE_FIGURA][:N_FIGURA], "reales")
    fig.suptitle(f"Clase {names[CLASE_FIGURA]}: 10 muestras por valor de w")
    fig.tight_layout()
    fig.savefig(common.figure_path("task3_4d_clase_por_w.png"), dpi=130)
    plt.close(fig)

    # ---------------------------------- 3.5(a): diversidad por clase y su caida
    div_w = {w: [por_w[w]["diversidad_por_clase"][n] for n in names] for w in ts.GUIDANCE_WS}
    caida_abs = [div_w[1][c] - div_w[7][c] for c in range(NUM_CLASSES)]
    caida_rel = [caida_abs[c] / div_w[1][c] for c in range(NUM_CLASSES)]
    c_abs = int(np.argmax(caida_abs))
    c_rel = int(np.argmax(caida_rel))
    print(f"\nMayor caida absoluta de diversidad (w=1 -> 7): {names[c_abs]} "
          f"({caida_abs[c_abs]:.3f}); relativa: {names[c_rel]} ({caida_rel[c_rel]:.1%})")

    x_pos = np.arange(NUM_CLASSES)
    ancho = 0.2
    colores_w = {1: "#9ecae1", 3: "#4292c6", 7: "#08519c"}       # un solo tono, claro -> oscuro
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x_pos - 1.5 * ancho, div_real, ancho, color="#969696", label="reales")
    for k, w in enumerate(ts.GUIDANCE_WS):
        ax.bar(x_pos + (k - 0.5) * ancho, div_w[w], ancho, color=colores_w[w], label=f"w = {w}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Distancia euclidiana media entre pares")
    ax.set_title("Diversidad intraclase por clase (50 imagenes por clase, pixeles en [-1, 1])")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(common.figure_path("task3_5a_diversidad_por_clase.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------- 3.5(b): matriz de confusion w = 1
    imgs1, labs1, pred1 = muestras[1]
    cm_gen = matriz_confusion(labs1, pred1)
    cm_val = matriz_confusion(y_val, probs_val.argmax(1))
    (ci, cj), tasa_par = par_mas_confundido(cm_gen)
    tasa = lambda cm, a, b: float(cm[a, b] / cm[a].sum())
    print(f"Par mas confundido (w=1): {names[ci]} <-> {names[cj]} | "
          f"generadas {tasa(cm_gen, ci, cj):.2%} / {tasa(cm_gen, cj, ci):.2%} | "
          f"validacion real {tasa(cm_val, ci, cj):.2%} / {tasa(cm_val, cj, ci):.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, cm, titulo in ((axes[0], cm_gen, "Muestras generadas con w = 1 (50 por clase)"),
                           (axes[1], cm_val, "Validacion real (5000 imagenes)")):
        pct = 100 * cm.double() / cm.sum(1, keepdim=True)
        im = ax.imshow(pct.numpy(), cmap="Blues", vmin=0, vmax=100)
        for a in range(NUM_CLASSES):
            for b in range(NUM_CLASSES):
                if cm[a, b] > 0:
                    ax.text(b, a, f"{pct[a, b]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if pct[a, b] > 50 else "black")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Clase predicha por el clasificador")
        ax.set_ylabel("Clase pedida / real")
        ax.set_title(titulo, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.025, label="% de la fila")
    fig.suptitle("Matrices de confusion del clasificador (porcentaje por fila)")
    fig.savefig(common.figure_path("task3_5b_matriz_confusion.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    pred_val = probs_val.argmax(1)
    filas = [
        (imgs1[(labs1 == ci) & (pred1 == cj)], f"gen. {names[ci]}\n-> pred. {names[cj]}"),
        (imgs1[(labs1 == cj) & (pred1 == ci)], f"gen. {names[cj]}\n-> pred. {names[ci]}"),
        (x_val[(y_val == ci) & (pred_val == cj)], f"real {names[ci]}\n-> pred. {names[cj]}"),
        (x_val[(y_val == cj) & (pred_val == ci)], f"real {names[cj]}\n-> pred. {names[ci]}"),
        (x_val[(y_val == ci) & (pred_val == ci)], f"real {names[ci]}\n(correcta)"),
        (x_val[(y_val == cj) & (pred_val == cj)], f"real {names[cj]}\n(correcta)"),
    ]
    fig, axes = plt.subplots(len(filas), N_FIGURA, figsize=(N_FIGURA * 0.9 + 2.0, 6.6))
    for r, (imgs, titulo) in enumerate(filas):
        fila_imagenes(axes[r], imgs[:N_FIGURA], titulo)
    fig.suptitle(f"Par mas confundido con w = 1: {names[ci]} <-> {names[cj]}")
    fig.tight_layout()
    fig.savefig(common.figure_path("task3_5b_par_confundido.png"), dpi=130)
    plt.close(fig)

    # ----------------------------------- 3.5(d): SNR en los intervalos de la 3.2
    _, _, alpha_bar = linear_schedule(T, BETA_START, BETA_END)
    snr = snr_from_alpha_bar(alpha_bar)
    perdida = json.load(open(os.path.join(common.RESULTS_DIR, "task3_2_results.json"),
                             encoding="utf-8"))["val_loss_por_intervalo_t"]
    intervalos = []
    for k, (lab, loss) in enumerate(zip(perdida["intervalos"], perdida["condicional"])):
        lo, hi = k * 100 + 1, (k + 1) * 100
        s = snr[lo - 1:hi]
        intervalos.append({"intervalo": lab, "perdida_condicional": loss,
                           "snr_inicio": float(snr[lo - 1]), "snr_fin": float(snr[hi - 1]),
                           "pasos_snr_mayor_100": int((s > 100).sum()),
                           "pasos_snr_menor_0.01": int((s < 0.01).sum())})

    results = {
        "seed": common.SEED,
        "clasificador": {
            "arquitectura": "Conv(1->32,3x3)-ReLU-MaxPool, Conv(32->64,3x3)-ReLU-MaxPool, "
                            "Linear(3136->128)-ReLU-Dropout(0.3), Linear(128->10)",
            "entrada": "imagenes reales en [-1, 1] (misma escala que las muestras)",
            "epochs": CLF_EPOCHS, "batch_size": CLF_BATCH, "lr": CLF_LR, "optimizador": "Adam",
            "historial": hist, "exactitud_validacion": acc_val,
            "checkpoint": os.path.relpath(CLF_CKPT, common.ROOT)},
        "definiciones": {
            "fidelidad": "fraccion de muestras cuya clase predicha coincide con la pedida",
            "diversidad": "media de ||x_i - x_j||_2 sobre los 1225 pares de cada clase "
                          "(pixeles en [-1, 1]), promediada sobre las 10 clases"},
        "tabla_3_4d": {
            "dispositivo_tiempos": dispositivo_tiempos,
            "filas": [{"w": w, "fidelidad": r["fidelidad"], "diversidad": r["diversidad"],
                       "segundos_100_imagenes": r["segundos_100_imagenes"],
                       "evaluaciones_red_100_imagenes": r["evaluaciones_red_100_imagenes"]}
                      for w, r in por_w.items()],
            "referencia_real": {"fidelidad": fid_real, "diversidad": float(np.mean(div_real)),
                                "imagenes": "primeras 50 por clase de validacion"}},
        "por_w": {str(w): r for w, r in por_w.items()},
        "diversidad_real_por_clase": dict(zip(names, div_real)),
        "analisis_3_5a": {
            "caida_abs_w1_a_w7": dict(zip(names, caida_abs)),
            "caida_rel_w1_a_w7": dict(zip(names, caida_rel)),
            "clase_mayor_caida_abs": names[c_abs],
            "clase_mayor_caida_rel": names[c_rel]},
        "analisis_3_5b": {
            "filas": "clase pedida (generadas) o real (validacion)",
            "columnas": "clase predicha",
            "clases": names,
            "matriz_generadas_w1": cm_gen.tolist(),
            "matriz_validacion_real": cm_val.tolist(),
            "par_mas_confundido": [names[ci], names[cj]],
            "suma_tasas_par": tasa_par,
            "tasas_generadas": {f"{names[ci]}->{names[cj]}": tasa(cm_gen, ci, cj),
                                f"{names[cj]}->{names[ci]}": tasa(cm_gen, cj, ci)},
            "tasas_validacion_real": {f"{names[ci]}->{names[cj]}": tasa(cm_val, ci, cj),
                                      f"{names[cj]}->{names[ci]}": tasa(cm_val, cj, ci)}},
        "analisis_3_5d": intervalos,
        "figuras": ["results/figures/task3_4d_clase_por_w.png",
                    "results/figures/task3_5a_diversidad_por_clase.png",
                    "results/figures/task3_5b_matriz_confusion.png",
                    "results/figures/task3_5b_par_confundido.png"],
    }
    print("JSON:", common.save_results("task3_4cd_5_results.json", results))


if __name__ == "__main__":
    main()
