"""Paso 1. Cosechas (vintages): mora 30+ y 90+ por mes en libros (MOB).

Una cosecha es el grupo de créditos originados en el mismo mes. Para cada una
se sigue qué porcentaje de lo originado ha llegado a 30+ y a 90+ días de mora
después de 1, 2, 3... meses en libros. Las curvas se comparan entre cosechas
y contra la mediana para detectar las que se originaron peor."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from comun import (MES_CORTE, MESES_ORIGEN, SALIDA, cargar, escribir_csv,
                   etiqueta_mes, guardar_figura)

UMBRAL_RAZON = 1.50      # la cosecha debe superar la mediana en al menos 50%
NIVEL_P = 0.001          # prueba binomial unilateral contra la tasa mediana
MOB_COMPARACION = [3, 6, 12, 18]


def primeros_eventos(panel: pd.DataFrame) -> pd.DataFrame:
    """Antigüedad (MOB) en la que cada crédito llega por primera vez a 30+ y a 90+."""
    p30 = panel[panel["estado"].isin([2, 3, 4])].groupby("id")["edad"].min().rename("mob_30")
    p90 = panel[panel["estado"] == 4].groupby("id")["edad"].min().rename("mob_90")
    return pd.concat([p30, p90], axis=1)


def curvas_acumuladas(pres: pd.DataFrame, ev: pd.DataFrame, columna: str, por_monto: bool = False) -> pd.DataFrame:
    """Tasa acumulada por cosecha (filas) y MOB (columnas). NaN donde aún no se observa."""
    d = pres.join(ev, on="id")
    out = {}
    max_mob = MES_CORTE - 0
    for v in range(MESES_ORIGEN):
        g = d[d["mes_origen"] == v]
        base = g["monto"].sum() if por_monto else len(g)
        fila = {}
        for m in range(0, MES_CORTE - v + 1):
            hit = g[g[columna] <= m]
            fila[m] = (hit["monto"].sum() if por_monto else len(hit)) / base
        out[v] = pd.Series(fila)
    tabla = pd.DataFrame(out).T
    tabla.index.name = "cosecha"
    return tabla.reindex(columns=range(0, MES_CORTE + 1))


def mora30_en_libros(pres: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de lo originado que está en 30+ días al cierre de cada MOB (no acumulado)."""
    p = panel.merge(pres[["id", "mes_origen"]], on="id")
    n_v = pres.groupby("mes_origen").size()
    c = p[p["estado"].isin([2, 3, 4])].groupby(["mes_origen", "edad"]).size().unstack(fill_value=0)
    c = c.reindex(index=range(MESES_ORIGEN), columns=range(0, MES_CORTE + 1), fill_value=0)
    t = c.div(n_v, axis=0)
    for v in range(MESES_ORIGEN):
        t.loc[v, t.columns > MES_CORTE - v] = np.nan
    t.index.name = "cosecha"
    return t


def comparar(curva: pd.DataFrame, n_v: pd.Series, nombre: str) -> pd.DataFrame:
    """Compara cada cosecha con la mediana de las cosechas observadas a ese MOB."""
    filas = []
    for m in MOB_COMPARACION:
        col = curva[m].dropna()
        mediana = col.median()
        for v, tasa in col.items():
            x = int(round(tasa * n_v[v]))
            p = binomtest(x, int(n_v[v]), max(mediana, 1e-6), alternative="greater").pvalue
            razon = tasa / mediana if mediana > 0 else np.nan
            filas.append({"medida": nombre, "mob": m, "cosecha": v, "mes": etiqueta_mes(v),
                          "originados": int(n_v[v]), "tasa": tasa, "mediana": mediana,
                          "razon_vs_mediana": razon, "p_valor": p,
                          "alerta": bool(razon >= UMBRAL_RAZON and p < NIVEL_P)})
    return pd.DataFrame(filas)


def main() -> None:
    pres, panel, _ = cargar()
    ev = primeros_eventos(panel)
    n_v = pres.groupby("mes_origen").size()

    c30 = curvas_acumuladas(pres, ev, "mob_30")
    c90 = curvas_acumuladas(pres, ev, "mob_90")
    c90m = curvas_acumuladas(pres, ev, "mob_90", por_monto=True)
    libros = mora30_en_libros(pres, panel)
    for nombre, t in [("cosechas_30_acumulada", c30), ("cosechas_90_acumulada", c90),
                      ("cosechas_90_acumulada_monto", c90m), ("cosechas_30_en_libros", libros)]:
        escribir_csv(t.round(5), SALIDA / f"{nombre}.csv", indice=True)

    comp = pd.concat([comparar(c30, n_v, "30+ acumulada"), comparar(c90, n_v, "90+ acumulada")])
    escribir_csv(comp.round(5), SALIDA / "cosechas_comparacion.csv")

    # Resumen por cosecha: alertas y primer MOB en que cada medida enciende la alerta
    resumen = pres.groupby("mes_origen").agg(
        originados=("id", "size"), monto=("monto", "sum"), score_medio=("score", "mean"),
        dti_medio=("dti", "mean"), pct_digital=("canal", lambda s: (s == "digital").mean())
    ).reset_index().rename(columns={"mes_origen": "cosecha"})
    resumen.insert(1, "mes", resumen["cosecha"].map(etiqueta_mes))
    for m in (6, 12):
        resumen[f"mora30_mob{m}"] = resumen["cosecha"].map(c30[m])
        resumen[f"mora90_mob{m}"] = resumen["cosecha"].map(c90[m])
    a30 = comp[(comp["medida"] == "30+ acumulada") & comp["alerta"]].groupby("cosecha")["mob"].min()
    a90 = comp[(comp["medida"] == "90+ acumulada") & comp["alerta"]].groupby("cosecha")["mob"].min()
    resumen["primer_mob_alerta_30"] = resumen["cosecha"].map(a30)
    resumen["primer_mob_alerta_90"] = resumen["cosecha"].map(a90)
    resumen["alerta"] = resumen["primer_mob_alerta_30"].notna() | resumen["primer_mob_alerta_90"].notna()
    escribir_csv(resumen.round(5), SALIDA / "cosechas_resumen.csv")

    # Gráfica
    fig, ejes = plt.subplots(2, 2, figsize=(13, 9))
    marca = resumen.set_index("cosecha")["alerta"]
    cmap = plt.get_cmap("Blues")
    for ax, curva, titulo in [(ejes[0, 0], c30, "Mora 30+ acumulada"), (ejes[0, 1], c90, "Mora 90+ acumulada")]:
        for v in range(MESES_ORIGEN):
            fila = curva.loc[v].dropna()
            if marca[v]:
                ax.plot(fila.index, fila.values * 100, color="#C00000", lw=1.4, alpha=0.9)
            else:
                ax.plot(fila.index, fila.values * 100, color=cmap(0.35 + 0.6 * v / MESES_ORIGEN), lw=0.9, alpha=0.8)
        ax.set_title(f"{titulo} por cosecha (en rojo, las que disparan alerta)")
        ax.set_xlabel("Meses en libros (MOB)")
        ax.set_ylabel("% de los créditos originados")
        ax.grid(alpha=0.25)
    # Mapa de calor de 90+ acumulada
    ax = ejes[1, 0]
    datos = c90.iloc[:, :37].to_numpy() * 100
    im = ax.imshow(datos, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_title("Mora 90+ acumulada, cosecha por MOB (%)")
    ax.set_xlabel("MOB")
    ax.set_ylabel("Cosecha (0 = ene 2022)")
    fig.colorbar(im, ax=ax)
    # Razón contra la mediana a MOB 12
    ax = ejes[1, 1]
    c12 = comp[(comp["medida"] == "90+ acumulada") & (comp["mob"] == 12)].set_index("cosecha")
    colores = ["#C00000" if a else "#7F7F7F" for a in c12["alerta"]]
    ax.bar(c12.index, c12["razon_vs_mediana"], color=colores)
    ax.axhline(1, color="black", lw=0.8)
    ax.axhline(UMBRAL_RAZON, color="#C00000", lw=0.8, ls="--")
    ax.set_title("Mora 90+ a MOB 12 contra la mediana de cosechas")
    ax.set_xlabel("Cosecha (0 = ene 2022)")
    ax.set_ylabel("Razón contra la mediana")
    fig.tight_layout()
    guardar_figura(fig, SALIDA / "01_cosechas.png")

    print(resumen[["cosecha", "mes", "originados", "score_medio", "mora30_mob6", "mora90_mob12",
                   "primer_mob_alerta_30", "primer_mob_alerta_90", "alerta"]].round(3).to_string(index=False))
    print("Cosechas en alerta:", sorted(resumen.loc[resumen["alerta"], "cosecha"].tolist()))


if __name__ == "__main__":
    main()
