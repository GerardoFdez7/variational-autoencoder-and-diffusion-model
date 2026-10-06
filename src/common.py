"""Utilidades compartidas: semilla, carga de datos y guardado de resultados.

Este modulo es estable: los scripts de cada task lo importan pero no lo modifican.
"""
import json
import os
import random

import numpy as np
import torch
from torchvision import datasets

# Semilla global del laboratorio. Se reporta en REPORTE.md.
SEED = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

N_VAL = 5000  # ultimos 5000 ejemplos de train como validacion


def set_seed(seed: int = SEED) -> None:
    """Fija la semilla de random, numpy y torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_fashion_mnist(scale: str = "unit"):
    """Devuelve (x_train, y_train, x_val, y_val, x_test, y_test) como tensores.

    Las imagenes salen con forma (N, 1, 28, 28). `scale` controla el rango:
    "unit" -> [0, 1] (Task 1), "symmetric" -> [-1, 1] (Task 2).
    """
    train = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True)

    x_all = train.data.float().div(255.0).unsqueeze(1)
    y_all = train.targets.clone()
    x_test = test.data.float().div(255.0).unsqueeze(1)
    y_test = test.targets.clone()

    if scale == "symmetric":
        x_all = x_all * 2.0 - 1.0
        x_test = x_test * 2.0 - 1.0
    elif scale != "unit":
        raise ValueError(f"scale desconocido: {scale}")

    # Los ultimos N_VAL ejemplos del conjunto de entrenamiento son validacion.
    x_train, x_val = x_all[:-N_VAL], x_all[-N_VAL:]
    y_train, y_val = y_all[:-N_VAL], y_all[-N_VAL:]
    return x_train, y_train, x_val, y_val, x_test, y_test


def save_results(name: str, payload: dict) -> str:
    """Guarda un diccionario de resultados como JSON en results/."""
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def figure_path(name: str) -> str:
    ensure_dirs()
    return os.path.join(FIGURES_DIR, name)
