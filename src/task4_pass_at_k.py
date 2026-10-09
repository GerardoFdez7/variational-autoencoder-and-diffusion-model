"""Task 4.2: pass@k con el estimador insesgado y el estimador ingenuo.

Para un problema con n intentos de los cuales c son correctos:

    insesgado:  pass@k = 1 - C(n - c, k) / C(n, k)
    ingenuo:    pass@k = 1 - (1 - c/n)^k

Los calculos se hacen con fracciones exactas (math.comb + Fraction) y solo se pasan a
float para reportarlos.
"""
import os
import sys
from fractions import Fraction
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

# (n, c) por problema, tomados del enunciado del laboratorio.
# TODO: llenar con los valores de P1-P6 del enunciado.
PROBLEMS = {
    # "P1": (n, c),
}
KS = [1, 5, 10]
K_DETALLE = 5          # k para el desarrollo de P2 y la comparacion de estimadores
OBJETIVO = Fraction(9, 10)
MAX_INTENTOS = 10      # presupuesto del agente


def pass_at_k(n, c, k):
    """Estimador insesgado (fraccion exacta). Requiere 0 <= c <= n y 1 <= k <= n."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"valores invalidos: n={n}, c={c}, k={k}")
    return 1 - Fraction(comb(n - c, k), comb(n, k))


def pass_at_k_naive(n, c, k):
    """Estimador ingenuo 1 - (1 - c/n)^k (fraccion exacta)."""
    return 1 - (1 - Fraction(c, n)) ** k


def min_k(n, c, objetivo=OBJETIVO):
    """Menor k <= n con pass@k > objetivo, o None si no existe."""
    return next((k for k in range(1, n + 1) if pass_at_k(n, c, k) > objetivo), None)


def detalle(n, c, k):
    """Desarrollo paso a paso de pass@k para un problema, como lineas de texto."""
    num, den = comb(n - c, k), comb(n, k)
    val = pass_at_k(n, c, k)
    return [f"n = {n}, c = {c}, k = {k}",
            f"C(n - c, k) = C({n - c}, {k}) = {num}",
            f"C(n, k)     = C({n}, {k}) = {den}",
            f"pass@{k} = 1 - {num}/{den} = {val} = {float(val):.6f}"]


def main():
    if not PROBLEMS:
        sys.exit("Falta llenar PROBLEMS con los valores (n, c) de P1-P6 del enunciado.")

    tabla = {}
    for name, (n, c) in PROBLEMS.items():
        tabla[name] = {"n": n, "c": c,
                       **{f"pass@{k}": float(pass_at_k(n, c, k)) for k in KS if k <= n}}
    promedios = {f"pass@{k}": sum(r[f"pass@{k}"] for r in tabla.values()) / len(tabla)
                 for k in KS if all(f"pass@{k}" in r for r in tabla.values())}

    print(f"{'':4} {'n':>3} {'c':>3} " + " ".join(f"{'pass@' + str(k):>9}" for k in KS))
    for name, r in tabla.items():
        print(f"{name:4} {r['n']:>3} {r['c']:>3} "
              + " ".join(f"{r.get(f'pass@{k}', float('nan')):>9.4f}" for k in KS))
    print("prom.    " + " ".join(f"{promedios.get(f'pass@{k}', float('nan')):>9.4f}"
                                  for k in KS))

    resultados = {"tabla": tabla, "promedios": promedios}

    if "P2" in PROBLEMS:
        n, c = PROBLEMS["P2"]
        lineas = detalle(n, c, K_DETALLE)
        k_min = min_k(n, c)
        print("\nDesarrollo de P2:\n  " + "\n  ".join(lineas))
        print(f"k minimo con pass@k > {float(OBJETIVO)} para P2: {k_min}")
        resultados["P2"] = {"desarrollo_k5": lineas, "k_minimo_pass_mayor_0.9": k_min}

    comparacion = {}
    print(f"\nInsesgado vs. ingenuo (k = {K_DETALLE}):")
    for name, (n, c) in PROBLEMS.items():
        if K_DETALLE > n:
            continue
        u, v = pass_at_k(n, c, K_DETALLE), pass_at_k_naive(n, c, K_DETALLE)
        comparacion[name] = {"insesgado": float(u), "ingenuo": float(v),
                             "diferencia_ingenuo_menos_insesgado": float(v - u)}
        print(f"  {name}: insesgado={float(u):.4f}  ingenuo={float(v):.4f}  "
              f"diferencia={float(v - u):+.4f}")
    resultados["comparacion_k5"] = comparacion

    # Con hasta MAX_INTENTOS intentos, un problema se resuelve si al menos uno de
    # ellos es correcto; la probabilidad estimada es pass@MAX_INTENTOS.
    agente = {name: float(pass_at_k(n, c, MAX_INTENTOS))
              for name, (n, c) in PROBLEMS.items() if MAX_INTENTOS <= n}
    print(f"\nProbabilidad de resolver con hasta {MAX_INTENTOS} intentos:")
    for name, p in agente.items():
        print(f"  {name}: {p:.4f}")
    resultados[f"agente_{MAX_INTENTOS}_intentos"] = agente

    print("JSON:", common.save_results("task4_2_results.json", resultados))


if __name__ == "__main__":
    main()
