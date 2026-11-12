"""Paso 4. LGD con recuperaciones y EAD por amortización.

LGD. Tras el incumplimiento (primer mes en 90+) se observan las recuperaciones
mes a mes. Como los incumplimientos recientes llevan pocos meses de cobro, la
curva de recuperación se arma con el método de escalera: la recuperación del
mes k después del incumplimiento es el cociente entre lo cobrado y la
exposición, usando solo los incumplimientos que ya cumplieron k meses. La LGD
es uno menos el valor presente de esa curva a la tasa de los créditos. Se
estima por canal y se contrasta con los incumplimientos que ya cerraron su
ventana de 24 meses.

EAD. En un crédito de cuota fija la exposición baja con la amortización
contractual. Al incumplir, el saldo queda congelado desde el primer pago
fallido; se mide cuántos meses de amortización antes del incumplimiento
describen mejor ese saldo."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import (CANALES, MES_CORTE, MESES_RECUPERACION, SALIDA, cargar, escribir_csv,
                   guardar_figura, saldo_programado)


def tabla_incumplimientos(pres: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    inc = panel[panel["estado"] == 4]
    primero = inc.sort_values("mes").groupby("id").head(1)[["id", "mes", "edad", "saldo"]]
    primero = primero.rename(columns={"mes": "mes_inc", "edad": "edad_inc", "saldo": "ead"})
    d = primero.merge(pres, on="id")
    d["r_m"] = d["tasa_anual"] / 12
    return d


def curva_escalera(d: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """Recuperación incremental por mes desde el incumplimiento, con datos censurados."""
    r = rec.merge(d[["id", "mes_inc", "ead"]], on="id")
    r["k"] = r["mes"] - r["mes_inc"]
    r = r[(r["k"] >= 1) & (r["k"] <= MESES_RECUPERACION)]
    filas = []
    for k in range(1, MESES_RECUPERACION + 1):
        expuestos = d[d["mes_inc"] + k <= MES_CORTE]
        cobrado = r.loc[r["k"] == k, "recup"].sum()
        base = expuestos["ead"].sum()
        filas.append({"mes_desde_incumplimiento": k, "incumplimientos_observados": len(expuestos),
                      "recuperacion_incremental": cobrado / base if base > 0 else np.nan})
    c = pd.DataFrame(filas)
    c["recuperacion_acumulada"] = c["recuperacion_incremental"].cumsum()
    return c


def lgd_desde_curva(c: pd.DataFrame, r_m: float) -> tuple[float, float]:
    k = c["mes_desde_incumplimiento"].to_numpy()
    inc = c["recuperacion_incremental"].to_numpy()
    return 1 - inc.sum(), 1 - float((inc * (1 + r_m) ** -k).sum())


def main() -> None:
    pres, panel, macro = cargar()
    d = tabla_incumplimientos(pres, panel)
    rec = panel[(panel["estado"] == 4) & (panel["recup"] > 0)][["id", "mes", "recup"]]

    segmentos = {"global": d}
    segmentos.update({c: d[d["canal"] == c] for c in CANALES})
    curvas, filas, parametros = [], [], {"lgd_descontada": {}, "tasa_mensual": {}}
    completos = d[d["mes_inc"] + MESES_RECUPERACION <= MES_CORTE]
    for nombre, g in segmentos.items():
        c = curva_escalera(g, rec[rec["id"].isin(g["id"])])
        c.insert(0, "segmento", nombre)
        curvas.append(c)
        r_m = float(np.average(g["r_m"], weights=g["ead"]))
        lgd_nd, lgd_d = lgd_desde_curva(c, r_m)
        # LGD realizada en incumplimientos con ventana completa
        gc = completos[completos["id"].isin(g["id"])]
        rc = rec.merge(gc[["id", "mes_inc", "ead", "r_m"]], on="id")
        rc["k"] = rc["mes"] - rc["mes_inc"]
        rc = rc[(rc["k"] >= 1) & (rc["k"] <= MESES_RECUPERACION)]
        vp = (rc["recup"] * (1 + r_m) ** -rc["k"]).sum()
        realizada = 1 - vp / gc["ead"].sum() if len(gc) else np.nan
        filas.append({"segmento": nombre, "incumplimientos": len(g), "ead_total": g["ead"].sum(),
                      "tasa_descuento_mensual": r_m, "lgd_sin_descontar": lgd_nd,
                      "lgd_descontada": lgd_d, "incumplimientos_ventana_completa": len(gc),
                      "lgd_realizada_ventana_completa": realizada})
        parametros["lgd_descontada"][nombre] = lgd_d
        parametros["tasa_mensual"][nombre] = r_m
        parametros.setdefault("curva_incremental", {})[nombre] = c["recuperacion_incremental"].tolist()
    curvas = pd.concat(curvas)
    lgd = pd.DataFrame(filas)
    escribir_csv(curvas.round(6), SALIDA / "lgd_curva_recuperacion.csv")
    escribir_csv(lgd.round(5), SALIDA / "lgd_por_segmento.csv")

    # Sensibilidad de la recuperación al desempleo (ventana completa)
    rt = rec.merge(completos[["id", "mes_inc", "ead", "score", "canal"]], on="id")
    rt = rt[(rt["mes"] - rt["mes_inc"] >= 1) & (rt["mes"] - rt["mes_inc"] <= MESES_RECUPERACION)]
    tot = rt.groupby("id")["recup"].sum().rename("rec_total")
    c = completos.set_index("id").join(tot).fillna({"rec_total": 0})
    c = c[c["rec_total"] > 0]
    c["R"] = c["rec_total"] / c["ead"]
    c["u"] = macro["desempleo"].to_numpy()[c["mes_inc"].to_numpy()]
    X = np.column_stack([np.ones(len(c)), c["u"], (c["score"] - 650) / 60, (c["canal"] == "digital").astype(float)])
    y = np.log(c["R"].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=X.shape[1]))
    u_ref = float(np.average(c["u"], weights=c["ead"]))
    sens = pd.DataFrame({"termino": ["intercepto", "desempleo", "score_z", "canal_digital"],
                         "coeficiente": beta, "error_estandar": se})
    escribir_csv(sens.round(5), SALIDA / "lgd_sensibilidad_macro.csv")
    parametros["beta_desempleo_recuperacion"] = float(beta[1])
    parametros["u_referencia"] = u_ref

    # EAD: cuántos meses de amortización antes del incumplimiento describen el saldo congelado
    filas = []
    for lag in range(0, 8):
        edad_ref = np.maximum(d["edad_inc"].to_numpy() - lag, 0)
        proxy = saldo_programado(d["monto"], d["r_m"], d["plazo"], edad_ref)
        razon = proxy / d["ead"]
        filas.append({"rezago_meses": lag, "razon_media_estimado_sobre_observado": float(razon.mean()),
                      "error_absoluto_medio": float(np.abs(razon - 1).mean()),
                      "ead_estimado_total_sobre_observado": float(proxy.sum() / d["ead"].sum())})
    ead = pd.DataFrame(filas)
    mejor = int(ead.loc[(ead["ead_estimado_total_sobre_observado"] - 1).abs().idxmin(), "rezago_meses"])
    escribir_csv(ead.round(5), SALIDA / "ead_validacion.csv")
    parametros["ead_rezago_meses"] = mejor

    # Corrida de amortización contractual de la cartera vigente al corte
    ult = panel[panel["mes"] == MES_CORTE].merge(pres, on="id")
    vig = ult[ult["estado"] <= 3]
    h = np.arange(0, 37)
    corrida = [{"mes_adelante": int(k), "saldo_contractual": float(
        saldo_programado(vig["monto"], vig["tasa_anual"] / 12, vig["plazo"], vig["edad"] + k).sum())} for k in h]
    corrida = pd.DataFrame(corrida)
    escribir_csv(corrida.round(2), SALIDA / "ead_corrida_contractual.csv")

    (SALIDA / "parametros_lgd_ead.json").write_text(json.dumps(parametros, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")

    # Gráfica
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = ejes[0]
    for nombre in ["global"] + CANALES:
        cc = curvas[curvas["segmento"] == nombre]
        ax.plot(cc["mes_desde_incumplimiento"], cc["recuperacion_acumulada"] * 100,
                lw=2.2 if nombre == "global" else 1.3, label=nombre)
    ax.set_title("Recuperación acumulada tras el incumplimiento")
    ax.set_xlabel("Meses desde el incumplimiento")
    ax.set_ylabel("% de la exposición")
    ax.legend()
    ax.grid(alpha=0.25)
    ax = ejes[1]
    x = np.arange(len(lgd))
    ax.bar(x - 0.2, lgd["lgd_descontada"] * 100, 0.4, label="curva de escalera (descontada)", color="#1F3864")
    ax.bar(x + 0.2, lgd["lgd_realizada_ventana_completa"] * 100, 0.4, label="realizada, ventana completa", color="#7F7F7F")
    ax.set_xticks(x, lgd["segmento"])
    ax.set_ylabel("LGD (%)")
    ax.set_title("LGD por canal")
    ax.legend(fontsize=8)
    ax = ejes[2]
    ax.plot(corrida["mes_adelante"], corrida["saldo_contractual"] / corrida["saldo_contractual"].iloc[0] * 100, color="#1F3864")
    ax.set_title("EAD: amortización contractual de la cartera vigente")
    ax.set_xlabel("Meses después del corte")
    ax.set_ylabel("% del saldo vigente al corte")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    guardar_figura(fig, SALIDA / "04_lgd_ead.png")

    print(lgd.round(4).to_string(index=False))
    print(sens.round(4).to_string(index=False), "u_ref", round(u_ref, 3))
    print(ead.round(4).to_string(index=False), "-> rezago", mejor)


if __name__ == "__main__":
    main()
