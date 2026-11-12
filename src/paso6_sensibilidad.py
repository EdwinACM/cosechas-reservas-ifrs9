"""Paso 6. Sensibilidad de la reserva IFRS 9.

(a) A los escenarios macro: cada uno por separado, las ponderaciones y un
    desplazamiento paralelo de la trayectoria base.
(b) A la definición de SICR: umbral de mora y umbrales de deterioro relativo
    de PD. Se reporta también cuánto cambia solo la reserva de etapas 1 y 2,
    porque la etapa 3 (que no depende del SICR) domina el total."""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import PESOS_ESCENARIOS, SALIDA, escribir_csv, guardar_figura
from reserva import ESCENARIOS, H, SICR_BASE, Cartera, trayectoria


def reserva_ruta(c: Cartera, U: np.ndarray, sicr=None) -> dict:
    r = c.evaluar_ruta(U)
    et = c.etapas(sicr)
    vig = float(np.where(et == 1, r["ecl12"], r["ecl_vida"]).sum())
    s3 = float(c.reserva_etapa3(r["ecl3_factor"]).sum())
    return {"reserva_etapas12": vig, "reserva_etapa3": s3, "reserva_total": vig + s3}


def main() -> None:
    c = Cartera()
    saldo12 = float(c.lib["saldo"].sum())

    # (a1) Escenarios por separado y ponderado
    filas = []
    for esc in ESCENARIOS:
        r = c.reserva(escenario=esc)
        filas.append({"caso": esc, "desempleo_medio_12m": c.u_medio[esc],
                      "reserva_etapas12": r[r["etapa"] < 3]["reserva"].sum(), "reserva_etapa3": r[r["etapa"] == 3]["reserva"].sum(),
                      "reserva_total": r["reserva"].sum()})
    r = c.reserva()
    filas.append({"caso": "ponderado (50/30/20)",
                  "desempleo_medio_12m": sum(PESOS_ESCENARIOS[e] * c.u_medio[e] for e in ESCENARIOS),
                  "reserva_etapas12": r[r["etapa"] < 3]["reserva"].sum(), "reserva_etapa3": r[r["etapa"] == 3]["reserva"].sum(),
                  "reserva_total": r["reserva"].sum()})
    esc = pd.DataFrame(filas)
    base_tot, base_12 = esc.loc[esc["caso"] == "base", ["reserva_total", "reserva_etapas12"]].iloc[0]
    esc["variacion_total_vs_base"] = esc["reserva_total"] / base_tot - 1
    esc["variacion_etapas12_vs_base"] = esc["reserva_etapas12"] / base_12 - 1
    escribir_csv(esc.round(4), SALIDA / "sensibilidad_escenarios.csv")

    # (a2) Peso del escenario adverso (el optimista queda en 20%)
    filas = []
    for w_adv in np.arange(0, 0.65, 0.1):
        pesos = {"base": 1 - 0.2 - w_adv, "adverso": w_adv, "optimista": 0.2}
        r = c.reserva(pesos=pesos)
        filas.append({"peso_adverso": round(w_adv, 2), "peso_base": round(pesos["base"], 2),
                      "reserva_etapas12": r[r["etapa"] < 3]["reserva"].sum(), "reserva_total": r["reserva"].sum()})
    pesos_t = pd.DataFrame(filas)
    escribir_csv(pesos_t.round(2), SALIDA / "sensibilidad_pesos.csv")

    # (a3) Desplazamiento paralelo del desempleo base (puntos porcentuales)
    filas = []
    base_U = trayectoria(c.u, 47, "base")
    for dp in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]:
        U = base_U.copy()
        U[1:] += dp
        filas.append({"desplazamiento_pp": dp, "desempleo_medio_12m": float(U[:12].mean()), **reserva_ruta(c, U)})
    desp = pd.DataFrame(filas)
    escribir_csv(desp.round(2), SALIDA / "sensibilidad_desempleo.csv")

    # (b) Definición de SICR
    filas = []
    ref = c.reserva()
    ref_12 = ref[ref["etapa"] < 3]["reserva"].sum()
    for bucket, razon, delta in itertools.product([1, 2], [1.25, 1.5, 2.0, 3.0, np.inf], [0.002, 0.005, 0.01]):
        if np.isinf(razon) and delta != 0.005:
            continue            # sin criterio relativo el delta no importa
        sicr = {"bucket_min": bucket, "razon_pd": razon, "delta_pd": delta}
        r = c.reserva(sicr=sicr)
        v = r[r["etapa"] < 3]
        e2 = v[v["etapa"] == 2]
        filas.append({
            "mora_minima": {1: "1 día o más", 2: "30 días o más"}[bucket],
            "razon_pd_minima": "sin criterio relativo" if np.isinf(razon) else razon,
            "aumento_pd_minimo_pp": None if np.isinf(razon) else delta * 100,
            "creditos_etapa2": len(e2), "saldo_etapa2": e2["saldo"].sum(),
            "pct_saldo_etapa2": e2["saldo"].sum() / v["saldo"].sum(),
            "reserva_etapas12": v["reserva"].sum(), "reserva_total": r["reserva"].sum(),
            "variacion_etapas12_vs_base": v["reserva"].sum() / ref_12 - 1,
            "es_base": bool(bucket == SICR_BASE["bucket_min"] and razon == SICR_BASE["razon_pd"]
                            and delta == SICR_BASE["delta_pd"])})
    sicr_t = pd.DataFrame(filas)
    escribir_csv(sicr_t.round(5), SALIDA / "sensibilidad_sicr.csv")

    # Cotas: todo a 12 meses y todo a vida (efecto máximo de la clasificación)
    w = PESOS_ESCENARIOS
    e12 = sum(w[e] * c.ecl12[e] for e in ESCENARIOS).sum()
    evida = sum(w[e] * c.ecl_vida[e] for e in ESCENARIOS).sum()
    cotas = pd.DataFrame([{"supuesto": "toda la cartera vigente a 12 meses", "reserva_etapas12": e12},
                          {"supuesto": "toda la cartera vigente a vida", "reserva_etapas12": evida},
                          {"supuesto": "clasificación base", "reserva_etapas12": ref_12}])
    escribir_csv(cotas.round(2), SALIDA / "sensibilidad_cotas_clasificacion.csv")

    # Gráfica
    fig, ejes = plt.subplots(2, 3, figsize=(17, 9))
    ax = ejes[0, 0]
    x = np.arange(len(esc))
    ax.bar(x, esc["reserva_etapas12"] / 1e6, color="#1F3864", label="etapas 1 y 2")
    ax.bar(x, esc["reserva_etapa3"] / 1e6, bottom=esc["reserva_etapas12"] / 1e6, color="#BFBFBF", label="etapa 3")
    ax.set_xticks(x, esc["caso"], rotation=12, fontsize=8)
    ax.set_ylabel("Reserva (millones)")
    ax.set_title("Reserva por escenario")
    ax.legend(fontsize=8)
    ax = ejes[0, 1]
    ax.plot(pesos_t["peso_adverso"] * 100, pesos_t["reserva_etapas12"] / 1e6, marker="o", color="#C00000")
    ax.set_xlabel("Peso del escenario adverso (%), optimista fijo en 20%")
    ax.set_ylabel("Reserva etapas 1 y 2 (millones)")
    ax.set_title("Sensibilidad a las ponderaciones")
    ax.grid(alpha=0.25)
    ax = ejes[0, 2]
    ax.plot(desp["desplazamiento_pp"], desp["reserva_etapas12"] / 1e6, marker="o", color="#1F3864")
    ax.set_xlabel("Desplazamiento del desempleo base (puntos porcentuales)")
    ax.set_ylabel("Reserva etapas 1 y 2 (millones)")
    ax.set_title("Sensibilidad a la trayectoria de desempleo")
    ax.grid(alpha=0.25)
    s = sicr_t[sicr_t["mora_minima"] == "30 días o más"].copy()
    s["etiqueta"] = [("solo mora" if isinstance(r, str) else f"{r}x / {d}pp") for r, d in
                     zip(s["razon_pd_minima"], s["aumento_pd_minimo_pp"])]
    colores = ["#C00000" if b else "#7F7F7F" for b in s["es_base"]]
    ax = ejes[1, 0]
    ax.bar(range(len(s)), s["saldo_etapa2"] / 1e6, color=colores)
    ax.set_xticks(range(len(s)), s["etiqueta"], rotation=70, fontsize=7)
    ax.set_ylabel("Saldo en etapa 2 (millones)")
    ax.set_title("SICR: saldo en etapa 2 (rojo = base)")
    ax = ejes[1, 1]
    ax.bar(range(len(s)), s["variacion_etapas12_vs_base"] * 100, color=colores)
    ax.set_xticks(range(len(s)), s["etiqueta"], rotation=70, fontsize=7)
    ax.set_ylabel("Cambio en reserva de etapas 1 y 2 (%)")
    ax.set_title("SICR: efecto en la reserva de etapas 1 y 2")
    ax = ejes[1, 2]
    ax.bar(range(3), cotas["reserva_etapas12"] / 1e6, color=["#7F7F7F", "#7F7F7F", "#C00000"])
    ax.set_xticks(range(3), ["todo a 12 meses", "todo a vida", "clasificación base"], fontsize=8)
    ax.set_ylim(cotas["reserva_etapas12"].min() / 1e6 * 0.9, cotas["reserva_etapas12"].max() / 1e6 * 1.05)
    ax.set_ylabel("Reserva etapas 1 y 2 (millones)")
    ax.set_title("Cotas de la clasificación")
    fig.tight_layout()
    guardar_figura(fig, SALIDA / "07_sensibilidad.png")

    pd.set_option("display.width", 220)
    print(esc.round(3).to_string(index=False))
    print(pesos_t.round(1).to_string(index=False))
    print(desp.round(1).to_string(index=False))
    print(sicr_t.round(4).to_string(index=False))
    print(cotas.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
