"""Paso 5. Reserva de pérdida esperada IFRS 9 por etapas a la fecha de corte
(diciembre de 2025) y comparación con el cálculo simplificado por buckets.

Etapa 1: sin aumento significativo de riesgo (SICR). Reserva a 12 meses.
Etapa 2: con SICR. Reserva a vida. SICR = mora de 30 días o más, o PD de vida
         remanente al menos 2 veces la que se esperaba al origen y 0.5 puntos
         porcentuales más alta.
Etapa 3: incumplimiento (90+). Saldo neto de lo recuperado menos el valor
         presente de lo que falta por recuperar según la curva del paso 4.
La reserva es el promedio ponderado de tres escenarios de desempleo
(base 50%, adverso 30%, optimista 20%)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import (MES_CORTE, PESOS_ESCENARIOS, SALIDA, escribir_csv, etiqueta_mes,
                   guardar_figura)
from reserva import ESCENARIOS, H, SICR_BASE, Cartera, etapas_en

SNAPSHOTS = [11, 17, 23, 29, 35]     # cierres de mes desde los que se mide la transición a 12 meses
NOMBRE_ETAPA = {1: "etapa 1", 2: "etapa 2", 3: "etapa 3"}


def tabla_etapas(c: Cartera) -> pd.DataFrame:
    filas = []
    base = c.reserva()
    for etapa, g in base.groupby("etapa"):
        fila = {"etapa": NOMBRE_ETAPA[etapa], "creditos": len(g), "saldo": g["saldo"].sum()}
        for esc in ESCENARIOS:
            fila[f"reserva_{esc}"] = c.reserva(escenario=esc).query("etapa == @etapa")["reserva"].sum()
        fila["reserva_ponderada"] = g["reserva"].sum()
        fila["cobertura_ponderada"] = fila["reserva_ponderada"] / fila["saldo"]
        filas.append(fila)
    t = pd.DataFrame(filas)
    total = {"etapa": "total", "creditos": t["creditos"].sum(), "saldo": t["saldo"].sum()}
    for col in [f"reserva_{e}" for e in ESCENARIOS] + ["reserva_ponderada"]:
        total[col] = t[col].sum()
    total["cobertura_ponderada"] = total["reserva_ponderada"] / total["saldo"]
    return pd.concat([t, pd.DataFrame([total])], ignore_index=True)


def pd_por_etapa(c: Cartera) -> tuple[pd.DataFrame, pd.DataFrame]:
    et = c.etapas()
    filas, curvas = [], []
    for esc in ESCENARIOS:
        for e in (1, 2):
            m = et == e
            filas.append({"etapa": NOMBRE_ETAPA[e], "escenario": esc, "creditos": int(m.sum()),
                          "pd_12m": c.pd12[esc][m].mean(), "pd_vida": c.pdv[esc][m].mean(),
                          "pd_12m_ponderada_saldo": np.average(c.pd12[esc][m], weights=c.lib["saldo"][m]),
                          "pd_origen_vida": c.pd_orig[m].mean()})
            cum = c.dD[esc][m].mean(axis=0).cumsum()
            curvas += [{"etapa": NOMBRE_ETAPA[e], "escenario": esc, "mes_adelante": k + 1, "pd_acumulada": cum[k]}
                       for k in range(H)]
    for e in (1, 2):
        m = et == e
        filas.append({"etapa": NOMBRE_ETAPA[e], "escenario": "ponderado", "creditos": int(m.sum()),
                      "pd_12m": sum(PESOS_ESCENARIOS[s] * c.pd12[s][m].mean() for s in ESCENARIOS),
                      "pd_vida": sum(PESOS_ESCENARIOS[s] * c.pdv[s][m].mean() for s in ESCENARIOS),
                      "pd_12m_ponderada_saldo": np.nan, "pd_origen_vida": c.pd_orig[m].mean()})
    filas.append({"etapa": "etapa 3", "escenario": "todos", "creditos": len(c.s3), "pd_12m": 1.0, "pd_vida": 1.0,
                  "pd_12m_ponderada_saldo": 1.0, "pd_origen_vida": np.nan})
    return pd.DataFrame(filas), pd.DataFrame(curvas)


def matriz_etapas(c: Cartera, alertas: set) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Etapa al cierre de t0 contra etapa 12 meses después, con los cierres de SNAPSHOTS agrupados."""
    cache = {}

    def etapa_libro(t):
        if t not in cache:
            lib = etapas_en(c.panel, c.pres, c.modelo, c.macro, t)
            cache[t] = lib.set_index("id")["etapa"]
        return cache[t]

    piezas = []
    for t0 in SNAPSHOTS:
        e0 = etapa_libro(t0).rename("e0")
        inc0 = c.panel[(c.panel["mes"] == t0) & (c.panel["estado"] == 4)].set_index("id").assign(e0=3)["e0"]
        origen = pd.concat([e0, inc0])
        fin = c.panel[c.panel["mes"] == t0 + 12].set_index("id")["estado"]
        e12 = etapa_libro(t0 + 12)
        destino = pd.Series("cerrado", index=origen.index)
        destino = destino.where(~origen.index.isin(fin[fin == 4].index), "etapa 3")
        destino.loc[origen.index.intersection(e12.index)] = e12.reindex(origen.index.intersection(e12.index)).map(NOMBRE_ETAPA)
        piezas.append(pd.DataFrame({"id": origen.index, "origen": origen.map(NOMBRE_ETAPA).to_numpy(),
                                    "destino": destino.to_numpy(), "t0": t0}))
    p = pd.concat(piezas).merge(c.pres[["id", "mes_origen"]], on="id")
    cols = ["etapa 1", "etapa 2", "etapa 3", "cerrado"]

    def mat(d):
        cuenta = pd.crosstab(d["origen"], d["destino"]).reindex(index=["etapa 1", "etapa 2", "etapa 3"],
                                                                columns=cols, fill_value=0)
        return cuenta.div(cuenta.sum(axis=1), axis=0), cuenta
    tasa, cuenta = mat(p)
    por_grupo = []
    for nombre, g in [("cosechas en alerta", p[p["mes_origen"].isin(alertas)]),
                      ("resto", p[~p["mes_origen"].isin(alertas)])]:
        t, _ = mat(g)
        por_grupo.append(t.assign(grupo=nombre, desde=t.index).reset_index(drop=True))
    return tasa, cuenta, pd.concat(por_grupo)[["grupo", "desde"] + cols]


def matriz_provisiones(c: Cartera) -> pd.DataFrame:
    """Tasas por bucket del cálculo simplificado: incumplimiento a 12 meses observado
    en toda la historia (hasta 12 meses antes del corte) por la LGD promedio."""
    primero = c.panel[c.panel["estado"] == 4].groupby("id")["mes"].min().rename("mes_inc")
    x = c.panel[(c.panel["estado"] <= 3) & (c.panel["mes"] <= MES_CORTE - 12)].join(primero, on="id")
    x["inc12"] = ((x["mes_inc"] > x["mes"]) & (x["mes_inc"] <= x["mes"] + 12)).astype(float)
    lgd = c.par["lgd_descontada"]["global"]
    filas = []
    for b, nombre in enumerate(["al corriente", "1-29", "30-59", "60-89"]):
        g = x[x["estado"] == b]
        pd12 = np.average(g["inc12"], weights=g["saldo"])
        filas.append({"bucket": nombre, "observaciones": len(g), "pd_12m_historica": pd12,
                      "lgd": lgd, "tasa_provision": pd12 * lgd})
    filas.append({"bucket": "90+", "observaciones": np.nan, "pd_12m_historica": 1.0, "lgd": lgd, "tasa_provision": lgd})
    return pd.DataFrame(filas)


def main() -> None:
    c = Cartera()
    resumen_coh = pd.read_csv(SALIDA / "cosechas_resumen.csv")
    alertas = set(resumen_coh.loc[resumen_coh["alerta"], "cosecha"])

    et = tabla_etapas(c)
    escribir_csv(et.round(4), SALIDA / "reserva_por_etapa.csv")
    pds, curvas = pd_por_etapa(c)
    escribir_csv(pds.round(5), SALIDA / "pd_por_etapa.csv")
    escribir_csv(curvas.round(5), SALIDA / "pd_curvas_escenarios.csv")

    base = c.reserva()
    base["alerta"] = base["mes_origen"].isin(alertas)
    # Reserva por cosecha
    g = base.groupby("mes_origen")
    por_cos = pd.DataFrame({
        "saldo": g["saldo"].sum(), "reserva": g["reserva"].sum(),
        "saldo_etapa2": base[base["etapa"] == 2].groupby("mes_origen")["saldo"].sum(),
        "saldo_etapa3": base[base["etapa"] == 3].groupby("mes_origen")["saldo"].sum(),
        "reserva_etapa12": base[base["etapa"] < 3].groupby("mes_origen")["reserva"].sum(),
        "saldo_etapa12": base[base["etapa"] < 3].groupby("mes_origen")["saldo"].sum(),
    }).fillna(0).reset_index().rename(columns={"mes_origen": "cosecha"})
    por_cos.insert(1, "mes", por_cos["cosecha"].map(etiqueta_mes))
    por_cos["cobertura"] = por_cos["reserva"] / por_cos["saldo"]
    por_cos["cobertura_etapa12"] = por_cos["reserva_etapa12"] / por_cos["saldo_etapa12"].replace(0, np.nan)
    por_cos["pct_saldo_etapa3"] = por_cos["saldo_etapa3"] / por_cos["saldo"]
    por_cos["alerta"] = por_cos["cosecha"].isin(alertas)
    escribir_csv(por_cos.round(5), SALIDA / "reserva_por_cosecha.csv")

    # Transición de etapas
    tasa, cuenta, por_grupo = matriz_etapas(c, alertas)
    escribir_csv(tasa.round(5), SALIDA / "transicion_etapas_12m.csv", indice=True)
    escribir_csv(cuenta, SALIDA / "transicion_etapas_12m_conteo.csv", indice=True)
    escribir_csv(por_grupo.round(5), SALIDA / "transicion_etapas_12m_por_grupo.csv")

    # Comparación con el cálculo simplificado por buckets
    mp = matriz_provisiones(c)
    escribir_csv(mp.round(5), SALIDA / "matriz_provisiones_tasas.csv")
    tasas = dict(zip(mp["bucket"], mp["tasa_provision"]))
    bucket_nombre = {0: "al corriente", 1: "1-29", 2: "30-59", 3: "60-89", 4: "90+"}
    base["bucket"] = base["estado"].map(bucket_nombre)
    lgd_g = tasas["90+"]
    # Para 90+ se aplica la LGD promedio al saldo bruto y se resta lo ya recuperado
    s3 = c.s3
    idx3 = (base["etapa"] == 3).to_numpy()
    assert idx3.sum() == len(s3)
    base["reserva_matriz"] = base["bucket"].map(tasas) * base["saldo"]
    base.loc[idx3, "reserva_matriz"] = np.maximum(lgd_g * s3["saldo"].to_numpy() - s3["recup_acum"].to_numpy(), 0.0)
    comp = base.groupby("bucket").agg(creditos=("id", "size"), saldo=("saldo", "sum"),
                                      reserva_ifrs9=("reserva", "sum"), reserva_matriz=("reserva_matriz", "sum"))
    comp = comp.reindex(list(bucket_nombre.values()))
    comp["cobertura_ifrs9"] = comp["reserva_ifrs9"] / comp["saldo"]
    comp["cobertura_matriz"] = comp["reserva_matriz"] / comp["saldo"]
    comp["diferencia"] = comp["reserva_ifrs9"] - comp["reserva_matriz"]
    tot = comp[["creditos", "saldo", "reserva_ifrs9", "reserva_matriz", "diferencia"]].sum()
    tot["cobertura_ifrs9"] = tot["reserva_ifrs9"] / tot["saldo"]
    tot["cobertura_matriz"] = tot["reserva_matriz"] / tot["saldo"]
    comp.loc["total"] = tot
    escribir_csv(comp.round(4), SALIDA / "comparacion_matriz_buckets.csv", indice=True)

    cmp_cos = base.groupby("mes_origen").agg(saldo=("saldo", "sum"), reserva_ifrs9=("reserva", "sum"),
                                              reserva_matriz=("reserva_matriz", "sum")).reset_index()
    cmp_cos = cmp_cos.rename(columns={"mes_origen": "cosecha"})
    cmp_cos.insert(1, "mes", cmp_cos["cosecha"].map(etiqueta_mes))
    cmp_cos["diferencia"] = cmp_cos["reserva_ifrs9"] - cmp_cos["reserva_matriz"]
    cmp_cos["alerta"] = cmp_cos["cosecha"].isin(alertas)
    # Comparación solo en cartera vigente (etapas 1 y 2), donde la matriz no ve la cosecha ni la macro
    vig = base[base["etapa"] < 3].groupby("mes_origen").agg(saldo=("saldo", "sum"), r_ifrs=("reserva", "sum"),
                                                              r_mat=("reserva_matriz", "sum"))
    cmp_cos["cobertura_ifrs9_vigente"] = cmp_cos["cosecha"].map(vig["r_ifrs"] / vig["saldo"])
    cmp_cos["cobertura_matriz_vigente"] = cmp_cos["cosecha"].map(vig["r_mat"] / vig["saldo"])
    escribir_csv(cmp_cos.round(5), SALIDA / "comparacion_matriz_cosechas.csv")

    # Resumen en JSON para el README
    vig_tot = base[base["etapa"] < 3]
    resumen = {
        "corte": etiqueta_mes(MES_CORTE),
        "creditos_vigentes": int(len(c.lib)), "creditos_etapa3": int(len(c.s3)),
        "creditos_castigados_mas_24m": int(len(c.castigada)),
        "saldo_castigado_neto": float(c.castigada["neto"].sum()),
        "reserva_total": float(base["reserva"].sum()),
        "reserva_matriz_total": float(base["reserva_matriz"].sum()),
        "reserva_etapa12": float(vig_tot["reserva"].sum()), "reserva_matriz_etapa12": float(vig_tot["reserva_matriz"].sum()),
        "desempleo_corte": float(c.u[MES_CORTE]), "desempleo_medio_12m": {k: float(v) for k, v in c.u_medio.items()},
        "lgd_media": {k: float(v.mean()) for k, v in c.lgd.items()},
    }
    (SALIDA / "resumen_reserva.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    # Gráficas
    fig, ejes = plt.subplots(2, 2, figsize=(14, 9))
    ax = ejes[0, 0]
    e = et[et["etapa"] != "total"]
    x = np.arange(len(e))
    for k, esc in enumerate(ESCENARIOS):
        ax.bar(x + (k - 1) * 0.25, e[f"reserva_{esc}"] / 1e6, 0.25, label=esc)
    ax.plot(x, e["reserva_ponderada"] / 1e6, "k_", markersize=26, label="ponderada")
    ax.set_xticks(x, e["etapa"])
    ax.set_ylabel("Reserva (millones de pesos sintéticos)")
    ax.set_title("Reserva por etapa y escenario")
    ax.legend(fontsize=8)
    ax = ejes[0, 1]
    for esc, color in zip(ESCENARIOS, ["#1F3864", "#C00000", "#2E75B6"]):
        for et_, ls in [("etapa 1", "-"), ("etapa 2", "--")]:
            cc = curvas[(curvas["etapa"] == et_) & (curvas["escenario"] == esc)]
            ax.plot(cc["mes_adelante"], cc["pd_acumulada"] * 100, ls=ls, color=color, label=f"{et_} {esc}")
    ax.set_title("PD acumulada hacia adelante por etapa y escenario")
    ax.set_xlabel("Meses después del corte")
    ax.set_ylabel("%")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    ax = ejes[1, 0]
    pc = por_cos.dropna(subset=["cobertura_etapa12"])
    colores = ["#C00000" if a else "#7F7F7F" for a in pc["alerta"]]
    ax.bar(pc["cosecha"], pc["cobertura_etapa12"] * 100, color=colores)
    ax.set_title("Cobertura de etapas 1 y 2 por cosecha (rojo = en alerta)")
    ax.set_xlabel("Cosecha (0 = ene 2022)")
    ax.set_ylabel("Reserva / saldo (%)")
    ax = ejes[1, 1]
    im = ax.imshow(tasa.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), tasa.columns)
    ax.set_yticks(range(3), tasa.index)
    for i in range(3):
        for j in range(4):
            v = tasa.iloc[i, j]
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", color="white" if v > 0.5 else "black")
    ax.set_title("Transición de etapas a 12 meses (cinco cierres, dic 2022 a dic 2024)")
    fig.tight_layout()
    guardar_figura(fig, SALIDA / "05_reserva_ifrs9.png")

    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = ejes[0]
    cc = comp.drop(index="total")
    x = np.arange(len(cc))
    ax.bar(x - 0.2, cc["cobertura_ifrs9"] * 100, 0.4, label="IFRS 9 por etapas", color="#1F3864")
    ax.bar(x + 0.2, cc["cobertura_matriz"] * 100, 0.4, label="matriz de provisiones", color="#7F7F7F")
    ax.set_xticks(x, cc.index)
    ax.set_ylabel("Reserva / saldo (%)")
    ax.set_title("Cobertura por bucket de mora")
    ax.legend()
    ax = ejes[1]
    ax.bar(cmp_cos["cosecha"] - 0.2, cmp_cos["cobertura_ifrs9_vigente"] * 100, 0.4, label="IFRS 9", color="#1F3864")
    ax.bar(cmp_cos["cosecha"] + 0.2, cmp_cos["cobertura_matriz_vigente"] * 100, 0.4, label="matriz", color="#7F7F7F")
    ax.set_title("Cobertura de la cartera vigente por cosecha")
    ax.set_xlabel("Cosecha (0 = ene 2022)")
    ax.set_ylabel("%")
    ax.legend()
    fig.tight_layout()
    guardar_figura(fig, SALIDA / "06_ifrs9_contra_matriz.png")

    pd.set_option("display.width", 200)
    print(et.round(3).to_string(index=False))
    print(pds.round(4).to_string(index=False))
    print(tasa.round(4).to_string())
    print(cuenta.to_string())
    print(por_grupo.round(3).to_string(index=False))
    print(mp.round(4).to_string(index=False))
    print(comp.round(3).to_string())
    print(por_cos[["cosecha", "saldo", "cobertura", "cobertura_etapa12", "pct_saldo_etapa3", "alerta"]].round(3).to_string(index=False))
    print(json.dumps(resumen, indent=1))


if __name__ == "__main__":
    main()
