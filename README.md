# Modelos generativos desde cero: VAE y difusión sobre Fashion-MNIST

Este proyecto implementa, usando únicamente tensores de PyTorch y sin librerías de alto nivel para modelos
generativos, dos de las familias de modelos generativos más representativas, evaluadas sobre Fashion-MNIST.
La primera parte entrena un autoencoder variacional con un espacio latente de dos dimensiones para tres
valores del peso β de la divergencia KL, y usa las figuras y métricas resultantes para estudiar el
compromiso entre fidelidad de reconstrucción y estructura del espacio latente. La segunda parte construye el
proceso forward de difusión con su calendario lineal de ruido, verifica numéricamente la forma cerrada
de `x_t` contra la simulación paso a paso, y compara el calendario lineal contra un calendario propio en
términos de relación señal a ruido. Todas las fórmulas (reparametrización, KL en forma cerrada, muestreo
forward y calendarios) están escritas a mano. Cada script deja sus resultados numéricos en un archivo JSON y
sus figuras en `results/figures/`.

## Requisitos

- Python 3.10 o superior
- Las dependencias de [requirements.txt](requirements.txt)

No se necesita GPU: todo corre en CPU en pocos minutos.

## Instalación

```bash
git clone <url-del-repositorio>
cd variational-autoencoder-and-diffusion-model

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecución

Ambos scripts se ejecutan desde la raíz del proyecto. La primera ejecución descarga Fashion-MNIST en `data/`
automáticamente.

```bash
# Task 1 — Autoencoder variacional (entrena 3 modelos, ~15 min en CPU)
python src/task1_vae.py

# Task 2 — Proceso forward de difusión (~1 min, no entrena ninguna red)
python src/task2_diffusion.py
```

## Salidas

| Ruta | Contenido |
|------|-----------|
| `results/figures/` | Todas las figuras de ambos tasks en PNG |
| `results/task1_results.json` | Métricas del VAE: curvas por época, tabla de β, verificación de la KL, nitidez |
| `results/task2_results.json` | Métricas de difusión: verificación de la forma cerrada, hitos de SNR, calendarios, varianzas |

La semilla aleatoria está fijada en `42` (definida en `src/common.py`), por lo que las ejecuciones son
reproducibles.

## Estructura

```
src/
  common.py            # semilla, carga de datos y utilidades de guardado
  task1_vae.py         # Task 1: VAE, barrido de β, figuras y análisis
  task2_diffusion.py   # Task 2: proceso forward, calendarios y verificaciones
results/
  figures/             # figuras generadas
  task1_results.json
  task2_results.json
```
