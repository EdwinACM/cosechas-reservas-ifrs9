"""Paso 3. Curvas de supervivencia por cosecha y modelo de PD de tiempo discreto.

1. Kaplan-Meier por cosecha del tiempo hasta el primer 90+ (los prepagos y
   vencimientos se tratan como censura).
2. Ajuste del modelo de cuatro estados de `modelo_pd.py` sobre todo el panel.
3. Validación dentro de muestra: incumplimiento observado contra esperado por
   cosecha a 12 y 24 meses, con y sin el efecto de cosecha. La razón
   observado/esperado sin efecto de cosecha dice cuánto riesgo extra traía
   cada cosecha que sus atributos no explican."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from comun import (MES_CORTE, MESES_ORIGEN, SALIDA, cargar, escribir_csv,
                   etiqueta_mes, guardar_figura)
from modelo_pd import NOMBRES_ESTATICAS, Modelo
from paso1_cosechas import primeros_eventos


def kaplan_meier(pres: pd.DataFrame, panel: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Supervivencia al primer 90+ por cosecha. Filas: cosecha. Columnas: MOB."""
    ultima = panel.groupby("id")["edad"].max()
    d = pres.set_index("id")[["mes_origen"]].join(ev).join(ultima.rename("t_fin"))
    d["evento"] = d["mob_90"].notna()
    d["t"] = np.where(d["evento"], d["mob_90"], d["t_fin"])
    out = {}
    for v, g in d.groupby("mes_origen"):
        max_mob = MES_CORTE - v
        s, curva = 1.0, {0: 1.0}
        for m in range(1, max_mob + 1):
            riesgo = (g["t"] >= m).sum()
            eventos = ((g["t"] == m) & g["evento"]).sum()
            s *= 1 - eventos / riesgo if riesgo else 1
            curva[m] = s
        out[v] = pd.Series(curva)
    tabla = pd.DataFrame(out).T.reindex(columns=range(MES_CORTE + 1))
    tabla.index.name = "cosecha"
    return tabla


def validar(modelo: Modelo, pres: pd.DataFrame, macro: pd.DataFrame, c90: pd.DataFrame, ev: pd.DataFrame):
    u = macro["desempleo"].to_numpy()
    filas, curvas, pd12 = [], {}, {}
    for v in range(MESES_ORIGEN):
        g = pres[pres["mes_origen"] == v]
        H = min(24, MES_CORTE - v)
        ceros = np.zeros(len(g), int)
        res = {}
        for etiqueta, con in [("con", True), ("sin", False)]:
            dD, _ = modelo.proyectar(g, ceros, ceros, v, u[v: v + H], H, con_cosecha=con)
            res[etiqueta] = dD
        fila = {"cosecha": v, "mes": etiqueta_mes(v), "originados": len(g)}
        for m in (12, 24):
            if m <= H:
                obs = c90.loc[v, m]
                e_con, e_sin = res["con"][:, :m].sum(axis=1).mean(), res["sin"][:, :m].sum(axis=1).mean()
                fila.update({f"observado_mob{m}": obs, f"esperado_con_mob{m}": e_con,
                             f"esperado_sin_mob{m}": e_sin,
                             f"oe_con_mob{m}": obs / e_con, f"oe_sin_mob{m}": obs / e_sin})
        filas.append(fila)
        curvas[v] = res["con"].mean(axis=0).cumsum()
        if H >= 12:
            pd12[v] = (g["id"].to_numpy(), res["con"][:, :12].sum(axis=1), res["sin"][:, :12].sum(axis=1))
    # Capacidad de ordenar riesgo (dentro de muestra): AUC del PD a 12 meses
    ids = np.concatenate([x[0] for x in pd12.values()])
    p_con = np.concatenate([x[1] for x in pd12.values()])
    p_sin = np.concatenate([x[2] for x in pd12.values()])
    y = (ev["mob_90"].reindex(ids).fillna(999).to_numpy() <= 12).astype(int)
    auc = {"auc_pd12_con_cosecha": roc_auc_score(y, p_con), "auc_pd12_sin_cosecha": roc_auc_score(y, p_sin),
           "pd12_media_modelo": float(p_con.mean()), "incumplimiento_observado_12m": float(y.mean())}
    return pd.DataFrame(filas), curvas, auc


def main() -> None:
    pres, panel, macro = cargar()
    ev = primeros_eventos(panel)
    km = kaplan_meier(pres, panel, ev)
    escribir_csv(km.round(5), SALIDA / "supervivencia_cosechas.csv", indice=True)

    modelo = Modelo.ajustar(pres, panel, macro)
    tabla = modelo.a_tabla()
    escribir_csv(tabla.round(6), SALIDA / "pd_modelo_coeficientes.csv")

    # Resumen legible: razones de momios de las variables principales
    filas = []
    for clave in ["entrada", "sube1", "sube2", "sube3"]:
        c = modelo.coef[clave]
        for nombre, b in zip(NOMBRES_ESTATICAS, c["estatico"]):
            filas.append({"modelo": clave, "variable": nombre, "coeficiente": b, "razon_de_momios": np.exp(b)})
        filas.append({"modelo": clave, "variable": "desempleo (por punto porcentual)",
                      "coeficiente": c["u"][0], "razon_de_momios": np.exp(c["u"][0])})
    escribir_csv(pd.DataFrame(filas).round(4), SALIDA / "pd_modelo_resumen.csv")
    efecto = pd.DataFrame({"cosecha": range(MESES_ORIGEN), "mes": [etiqueta_mes(v) for v in range(MESES_ORIGEN)],
                           "efecto_logit": modelo.efecto_cosecha,
                           "razon_de_momios": np.exp(modelo.efecto_cosecha)})
    escribir_csv(efecto.round(4), SALIDA / "pd_efecto_cosecha.csv")

    c90 = pd.read_csv(SALIDA / "cosechas_90_acumulada.csv", index_col=0)
    c90.columns = c90.columns.astype(int)
    val, curvas, auc = validar(modelo, pres, macro, c90, ev)
    escribir_csv(val.round(5), SALIDA / "pd_validacion_cosechas.csv")
    escribir_csv(pd.DataFrame([auc]).round(5), SALIDA / "pd_validacion_global.csv")

    # Gráfica
    resumen = pd.read_csv(SALIDA / "cosechas_resumen.csv").set_index("cosecha")
    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.8))
    cmap = plt.get_cmap("Blues")
    ax = ejes[0]
    for v in range(MESES_ORIGEN):
        fila = km.loc[v].dropna()
        color = "#C00000" if resumen.loc[v, "alerta"] else cmap(0.35 + 0.6 * v / MESES_ORIGEN)
        ax.plot(fila.index, fila.values, color=color, lw=1.3 if resumen.loc[v, "alerta"] else 0.9)
    ax.set_title("Supervivencia al primer 90+ por cosecha (Kaplan-Meier)")
    ax.set_xlabel("Meses en libros")
    ax.set_ylabel("Probabilidad de no haber incumplido")
    ax.grid(alpha=0.25)
    ax = ejes[1]
    for v, color in [(5, "#1F3864"), (17, "#C00000"), (25, "#2E75B6"), (30, "#ED7D31")]:
        obs = c90.loc[v].dropna()
        obs = obs[obs.index <= 24]
        ax.plot(obs.index, obs.values * 100, color=color, lw=1.8, label=f"{etiqueta_mes(v)} observado")
        mod = curvas[v]
        ax.plot(range(1, len(mod) + 1), mod * 100, color=color, lw=1.2, ls="--", label=f"{etiqueta_mes(v)} modelo")
    ax.set_title("Incumplimiento acumulado: observado y modelo")
    ax.set_xlabel("Meses en libros")
    ax.set_ylabel("% de los créditos originados")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    ax = ejes[2]
    x = val["cosecha"]
    ax.bar(x - 0.2, val["oe_sin_mob12"], 0.4, color="#C00000", label="sin efecto de cosecha")
    ax.bar(x + 0.2, val["oe_con_mob12"], 0.4, color="#7F7F7F", label="con efecto de cosecha")
    ax.axhline(1, color="black", lw=0.8)
    ax.set_title("Observado / esperado a MOB 12")
    ax.set_xlabel("Cosecha (0 = ene 2022)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    guardar_figura(fig, SALIDA / "03_pd_supervivencia.png")

    print(pd.DataFrame(filas).round(3).to_string(index=False))
    print(efecto.round(3).to_string(index=False))
    print(val[["cosecha", "observado_mob12", "esperado_con_mob12", "esperado_sin_mob12", "oe_sin_mob12", "oe_con_mob12"]].round(3).to_string(index=False))
    print({k: round(float(v), 4) for k, v in auc.items()})
    print("fracciones de baja:", modelo.baja)


if __name__ == "__main__":
    main()
