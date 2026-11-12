"""Motor de la reserva de pérdida esperada (ECL) IFRS 9.

Reúne lo que usan los pasos 5 y 6:
- PD a 12 meses y a vida de cada crédito vigente, por escenario macro
- PD "al origen" para el mismo plazo remanente (para el aumento significativo
  de riesgo, SICR)
- EAD por amortización contractual y LGD por canal ajustada por escenario
- Reserva por crédito para cada escenario, en versión de 12 meses y a vida.

Todo es una simplificación didáctica y no reproduce la metodología de ninguna
institución."""
import json

import numpy as np
import pandas as pd

from comun import (HORIZONTE_MAX, MES_CORTE, MESES_RECUPERACION, PESOS_ESCENARIOS, SALIDA,
                   cargar, ruta_macro, saldo_programado)
from modelo_pd import Modelo

ESCENARIOS = list(PESOS_ESCENARIOS)
H = HORIZONTE_MAX

# Criterio base de aumento significativo de riesgo
SICR_BASE = {"bucket_min": 2, "razon_pd": 2.0, "delta_pd": 0.005}


def trayectoria(u_hist: np.ndarray, mes0: int, escenario: str, horizonte: int = H) -> np.ndarray:
    """Desempleo del mes de inicio de cada paso: el valor observado en mes0 y luego el escenario."""
    u0 = float(u_hist[mes0])
    return np.concatenate([[u0], ruta_macro(u0, escenario, horizonte)[: horizonte - 1]])


def pd_origen(libro: pd.DataFrame, modelo: Modelo, macro: pd.DataFrame, mes0: int) -> np.ndarray:
    """PD de vida remanente que se habría estimado al origen para un crédito de esos atributos
    que llega al corriente a su edad actual: escenario base desde el origen, sin efecto de cosecha
    (el banco no lo conocía al otorgar el crédito)."""
    u = macro["desempleo"].to_numpy()
    out = np.zeros(len(libro))
    for v, f in libro.groupby("mes_origen").indices.items():
        g = libro.iloc[f]
        a = mes0 - v
        U = trayectoria(u, v, "base", a + H)[a: a + H]
        z = np.zeros(len(g), int)
        dD, _ = modelo.proyectar(g, z, z + a, mes0, U, H, con_cosecha=False)
        out[f] = dD.sum(axis=1)
    return np.clip(out, 1e-5, 1)


def asignar_etapa(estado, pd_vida, pd_orig, bucket_min=2, razon_pd=2.0, delta_pd=0.005) -> np.ndarray:
    """Etapa 2 si hay mora de bucket_min o más, o si la PD de vida remanente
    se multiplicó por razon_pd y subió al menos delta_pd. De lo contrario, etapa 1."""
    razon = pd_vida / np.maximum(pd_orig, 1e-5)
    sicr = (np.asarray(estado) >= bucket_min) | ((razon >= razon_pd) & ((pd_vida - pd_orig) >= delta_pd))
    return np.where(sicr, 2, 1)


def libro_en(panel: pd.DataFrame, pres: pd.DataFrame, mes0: int) -> pd.DataFrame:
    """Créditos vigentes (no incumplidos ni cerrados) al cierre de mes0."""
    x = panel[(panel["mes"] == mes0) & (panel["estado"] <= 3)]
    return x.merge(pres, on="id").reset_index(drop=True)


def pd_libro(libro, modelo, macro, mes0, escenario="base"):
    """Flujos de incumplimiento (N, H) de cada crédito vigente en el escenario dado."""
    u = macro["desempleo"].to_numpy()
    return modelo.proyectar(libro, libro["estado"].to_numpy(), libro["edad"].to_numpy(), mes0,
                            trayectoria(u, mes0, escenario), H)[0]


def etapas_en(panel, pres, modelo, macro, mes0, sicr=SICR_BASE) -> pd.DataFrame:
    libro = libro_en(panel, pres, mes0)
    dD = pd_libro(libro, modelo, macro, mes0)
    pd_vida = dD.sum(axis=1)
    pd_o = pd_origen(libro, modelo, macro, mes0)
    libro["pd_vida"], libro["pd_origen"] = pd_vida, pd_o
    libro["pd_12m"] = dD[:, :12].sum(axis=1)
    libro["etapa"] = asignar_etapa(libro["estado"], pd_vida, pd_o, **sicr)
    return libro


class Cartera:
    """Foto de la cartera a la fecha de corte con todo lo necesario para reservar."""

    def __init__(self):
        self.pres, self.panel, self.macro = cargar()
        self.modelo = Modelo.de_tabla(pd.read_csv(SALIDA / "pd_modelo_coeficientes.csv"))
        self.par = json.loads((SALIDA / "parametros_lgd_ead.json").read_text(encoding="utf-8"))
        self.u = self.macro["desempleo"].to_numpy()
        self._vigente()
        self._incumplida()

    # --- cartera vigente (etapas 1 y 2) ----------------------------------
    def _vigente(self):
        lib = libro_en(self.panel, self.pres, MES_CORTE)
        self.lib = lib
        n = len(lib)
        r_m = (lib["tasa_anual"] / 12).to_numpy()
        a = lib["edad"].to_numpy()
        lag = self.par["ead_rezago_meses"]
        t = np.arange(1, H + 1)[None, :]
        prog = saldo_programado(lib["monto"].to_numpy()[:, None], r_m[:, None], lib["plazo"].to_numpy()[:, None],
                                a[:, None] + np.maximum(t - lag, 0))
        self.ead_t = np.minimum(lib["saldo"].to_numpy()[:, None], prog)
        self.v = (1 + r_m[:, None]) ** -t
        lgd_base = lib["canal"].map(self.par["lgd_descontada"]).to_numpy()
        self.pd_orig = pd_origen(lib, self.modelo, self.macro, MES_CORTE)
        self.lgd_base = lgd_base
        self.res = {esc: self.evaluar_ruta(trayectoria(self.u, MES_CORTE, esc)) for esc in ESCENARIOS}
        self.dD = {e: r["dD"] for e, r in self.res.items()}
        self.pd12 = {e: r["pd12"] for e, r in self.res.items()}
        self.pdv = {e: r["pdv"] for e, r in self.res.items()}
        self.ecl12 = {e: r["ecl12"] for e, r in self.res.items()}
        self.ecl_vida = {e: r["ecl_vida"] for e, r in self.res.items()}
        self.lgd = {e: r["lgd"] for e, r in self.res.items()}
        self.u_medio = {e: r["u12"] for e, r in self.res.items()}

    def evaluar_ruta(self, U: np.ndarray) -> dict:
        """PD y reserva por crédito (vigente) para una trayectoria de desempleo U (H meses desde el corte)."""
        beta, u_ref = self.par["beta_desempleo_recuperacion"], self.par["u_referencia"]
        dD = self.modelo.proyectar(self.lib, self.lib["estado"].to_numpy(), self.lib["edad"].to_numpy(),
                                   MES_CORTE, U, H)[0]
        u12 = float(U[:12].mean())
        lgd = np.clip(1 - (1 - self.lgd_base) * np.exp(beta * (u12 - u_ref)), 0.05, 1.0)
        flujo = dD * self.ead_t * self.v * lgd[:, None]
        return {"dD": dD, "pd12": dD[:, :12].sum(axis=1), "pdv": dD.sum(axis=1), "lgd": lgd, "u12": u12,
                "ecl12": flujo[:, :12].sum(axis=1), "ecl_vida": flujo.sum(axis=1),
                "ecl3_factor": float(np.exp(beta * (u12 - u_ref)))}

    # --- cartera en incumplimiento (etapa 3) ------------------------------
    def _incumplida(self):
        ult = self.panel[(self.panel["mes"] == MES_CORTE) & (self.panel["estado"] == 4)].merge(self.pres, on="id")
        primero = (self.panel[self.panel["estado"] == 4].groupby("id")["mes"].min().rename("mes_inc"))
        ult = ult.join(primero, on="id")
        ult["meses_inc"] = MES_CORTE - ult["mes_inc"]
        ult["neto"] = ult["saldo"] - ult["recup_acum"]
        self.castigada = ult[ult["meses_inc"] > MESES_RECUPERACION].reset_index(drop=True)
        s3 = ult[ult["meses_inc"] <= MESES_RECUPERACION].reset_index(drop=True)
        self.s3 = s3
        # Valor presente de lo que falta por recuperar según la curva y los meses ya transcurridos
        vp = np.zeros(len(s3))
        for canal in s3["canal"].unique():
            curva = np.array(self.par["curva_incremental"][canal])
            tasa = self.par["tasa_mensual"][canal]
            for i in np.flatnonzero((s3["canal"] == canal).to_numpy()):
                k = int(s3.at[i, "meses_inc"])
                j = np.arange(k + 1, MESES_RECUPERACION + 1)
                vp[i] = (curva[j - 1] * (1 + tasa) ** -(j - k)).sum() * s3.at[i, "saldo"]
        self.vp_recup = vp
        beta, u_ref = self.par["beta_desempleo_recuperacion"], self.par["u_referencia"]
        self.ecl3 = {esc: self.reserva_etapa3(np.exp(beta * (self.u_medio[esc] - u_ref))) for esc in ESCENARIOS}

    def reserva_etapa3(self, factor: float) -> np.ndarray:
        """Saldo neto de lo cobrado menos el valor presente de lo que falta recuperar (ajustado por escenario)."""
        return np.maximum(self.s3["neto"].to_numpy() - self.vp_recup * factor, 0.0)

    # --- reserva ------------------------------------------------------------
    def pesos(self, pesos=None):
        return pesos or PESOS_ESCENARIOS

    def etapas(self, sicr=None) -> np.ndarray:
        sicr = sicr or SICR_BASE
        return asignar_etapa(self.lib["estado"].to_numpy(), self.pdv["base"], self.pd_orig, **sicr)

    def reserva(self, sicr=None, pesos=None, escenario=None) -> pd.DataFrame:
        """Reserva por crédito. Con `escenario` usa solo ese; si no, la ponderación de los tres."""
        w = {escenario: 1.0} if escenario else self.pesos(pesos)
        et = self.etapas(sicr)
        e12 = sum(w.get(e, 0) * self.ecl12[e] for e in ESCENARIOS)
        evida = sum(w.get(e, 0) * self.ecl_vida[e] for e in ESCENARIOS)
        lib = self.lib[["id", "mes_origen", "estado", "saldo", "canal", "plazo", "edad"]].copy()
        lib["etapa"] = et
        lib["reserva"] = np.where(et == 1, e12, evida)
        lib["reserva_12m"], lib["reserva_vida"] = e12, evida
        s3 = self.s3[["id", "mes_origen", "estado", "neto", "canal", "plazo", "edad"]].rename(columns={"neto": "saldo"}).copy()
        s3["etapa"] = 3
        s3["reserva"] = sum(w.get(e, 0) * self.ecl3[e] for e in ESCENARIOS)
        s3["reserva_12m"] = s3["reserva_vida"] = s3["reserva"]
        return pd.concat([lib, s3], ignore_index=True)
