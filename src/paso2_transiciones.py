"""Paso 2. Matrices de transición de buckets de mora.

Se mide de un mes al siguiente (y a 12 meses) cómo se mueven los créditos entre
al corriente, 1-29, 30-59, 60-89 y 90+ días, con una columna extra para los que
se cierran (prepago o vencimiento). Se comparan las cosechas en alerta del
paso 1 contra el resto, y los meses de estrés de desempleo contra los demás."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import BUCKETS, SALIDA, cargar, escribir_csv, guardar_figura

DESTINOS = BUCKETS + ["cerrado"]


def pares(panel: pd.DataFrame, rezago: int) -> pd.DataFrame:
    """Une cada fila con la del mismo crédito `rezago` meses después."""
    a = panel[["id", "mes", "estado"]]
    b = panel[["id", "mes", "estado"]].copy()
    b["mes"] = b["mes"] - rezago
    p = a.merge(b, on=["id", "mes"], suffixes=("", "_sig"))
    return p[p["estado"] <= 4]


def matriz(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cuenta = pd.crosstab(p["estado"], p["estado_sig"]).reindex(index=range(5), columns=range(6), fill_value=0)
    tasa = cuenta.div(cuenta.sum(axis=1).replace(0, np.nan), axis=0)
    cuenta.index, cuenta.columns = BUCKETS, DESTINOS
    tasa.index, tasa.columns = BUCKETS, DESTINOS
    return tasa, cuenta


def main() -> None:
    pres, panel, macro = cargar()
    resumen = pd.read_csv(SALIDA / "cosechas_resumen.csv")
    alertas = set(resumen.loc[resumen["alerta"], "cosecha"])
    panel = panel.merge(pres[["id", "mes_origen"]], on="id")

    p1 = pares(panel, 1)
    p1 = p1.merge(pres[["id", "mes_origen"]], on="id").merge(
        macro.rename(columns={"desempleo": "u"}), on="mes")
    p1["grupo_cosecha"] = np.where(p1["mes_origen"].isin(alertas), "cosechas en alerta", "resto")
    p1["periodo"] = np.where(p1["u"] >= 4.6, "desempleo alto (4.6% o más)", "desempleo normal")

    salidas = {}
    tasa, cuenta = matriz(p1)
    salidas["global"] = tasa
    escribir_csv(tasa.round(5), SALIDA / "transicion_buckets_1m.csv", indice=True)
    escribir_csv(cuenta, SALIDA / "transicion_buckets_1m_conteo.csv", indice=True)

    largas = []
    for columna in ["grupo_cosecha", "periodo"]:
        for nombre, g in p1.groupby(columna):
            t, c = matriz(g)
            salidas[nombre] = t
            largas.append(t.assign(grupo=nombre, desde=t.index).reset_index(drop=True))
    escribir_csv(pd.concat(largas)[["grupo", "desde"] + DESTINOS].round(5), SALIDA / "transicion_buckets_1m_por_grupo.csv")

    t12, c12 = matriz(pares(panel, 12))
    escribir_csv(t12.round(5), SALIDA / "transicion_buckets_12m.csv", indice=True)

    # Indicadores de flujo: roll rate de cada bucket al siguiente y tasa de cura
    filas = []
    for nombre, t in salidas.items():
        filas.append({"grupo": nombre,
                      "roll_0_a_1": t.loc["al corriente", "1-29"],
                      "roll_1_a_2": t.loc["1-29", "30-59"],
                      "roll_2_a_3": t.loc["30-59", "60-89"],
                      "roll_3_a_4": t.loc["60-89", "90+"],
                      "cura_desde_30_59": t.loc["30-59", "al corriente"],
                      "cadena_0_a_90": t.loc["al corriente", "1-29"] * t.loc["1-29", "30-59"]
                      * t.loc["30-59", "60-89"] * t.loc["60-89", "90+"]})
    flujo = pd.DataFrame(filas)
    escribir_csv(flujo.round(5), SALIDA / "roll_rates.csv")

    # Gráfica
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, nombre in [(ejes[0], "global"), (ejes[1], "12m")]:
        m = salidas["global"] if nombre == "global" else t12
        im = ax.imshow(m.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6), DESTINOS, rotation=30)
        ax.set_yticks(range(5), BUCKETS)
        for i in range(5):
            for j in range(6):
                v = m.iloc[i, j]
                if pd.notna(v) and v >= 0.0005:
                    ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.5 else "black")
        ax.set_title("Transición a 1 mes" if nombre == "global" else "Transición a 12 meses")
        ax.set_xlabel("Bucket destino")
        ax.set_ylabel("Bucket origen")
    ax = ejes[2]
    claves = ["resto", "cosechas en alerta", "desempleo normal", "desempleo alto (4.6% o más)"]
    cols = ["roll_0_a_1", "roll_1_a_2", "roll_2_a_3", "roll_3_a_4"]
    for nombre in claves:
        fila = flujo[flujo["grupo"] == nombre].iloc[0]
        ax.plot(range(4), [fila[c] * 100 for c in cols], marker="o", label=nombre)
    ax.set_yscale("log")
    ax.set_xticks(range(4), ["0 a 1-29", "1-29 a 30-59", "30-59 a 60-89", "60-89 a 90+"], rotation=15)
    ax.set_ylabel("% por mes (escala logarítmica)")
    ax.set_title("Roll rates mensuales por grupo")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    guardar_figura(fig, SALIDA / "02_transiciones_buckets.png")

    print(salidas["global"].round(4).to_string())
    print(t12.round(4).to_string())
    print(flujo.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
