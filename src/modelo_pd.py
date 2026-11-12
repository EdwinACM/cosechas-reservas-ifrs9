"""Modelo de PD de tiempo discreto con cuatro estados de mora.

Es una cadena de Markov mensual cuyas probabilidades de transición dependen de
cada crédito (regresiones logísticas) y del calendario:

  cierre   : un crédito al corriente se liquida antes de tiempo
  entrada  : un crédito al corriente cae en 1-29 días
  sube1..3 : un crédito en 1-29, 30-59 o 60-89 días empeora un bucket
             (si no empeora, cura o baja un bucket; esa mezcla se toma de los datos)

El incumplimiento es llegar a 90+. Con la cadena se obtiene, para cualquier
crédito y cualquier trayectoria de desempleo, la probabilidad de incumplir mes
a mes (PD marginal), la acumulada (PD a 12 meses y a vida) y la de cierre. Es
el equivalente discreto de un modelo de riesgos proporcionales, con la
antigüedad como base y el desempleo y la estacionalidad como covariables.
Las cosechas entran como efectos fijos."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from comun import MESES_ORIGEN, sigmoide

U_REF = 4.0


def estaticas(pres: pd.DataFrame) -> np.ndarray:
    """Variables fijas del crédito: score, dti, canal y plazo (estandarizadas)."""
    return np.column_stack([
        (pres["score"].to_numpy() - 650) / 60,
        (pres["dti"].to_numpy() - 0.25) / 0.10,
        (pres["canal"].to_numpy() == "sucursal").astype(float),
        (pres["canal"].to_numpy() == "digital").astype(float),
        (pres["plazo"].to_numpy() - 24) / 12,
    ])


NOMBRES_ESTATICAS = ["score_z", "dti_z", "canal_sucursal", "canal_digital", "plazo_z"]


def base_edad(a) -> np.ndarray:
    a = np.asarray(a, float)
    return np.column_stack([np.log1p(a), a / 12, (a / 12) ** 2])


def dummies_mes(c) -> np.ndarray:
    c = np.asarray(c)
    return (c[:, None] == np.arange(1, 12)[None, :]).astype(float)


def dummies_cosecha(v) -> np.ndarray:
    v = np.asarray(v)
    return (v[:, None] == np.arange(1, MESES_ORIGEN)[None, :]).astype(float)


class Modelo:
    def __init__(self, coef: dict, baja: dict):
        self.coef = coef          # coef[clave] = {intercepto, estatico, edad, mes, u, cosecha...}
        self.baja = baja          # fracción de "no empeora" que baja un bucket {2: d2, 3: d3}

    # --- ajuste -----------------------------------------------------------
    @classmethod
    def ajustar(cls, pres: pd.DataFrame, panel: pd.DataFrame, macro: pd.DataFrame, C: float = 10.0):
        S = estaticas(pres)
        orig = pres["mes_origen"].to_numpy()
        plazo = pres["plazo"].to_numpy()
        u_mes = macro["desempleo"].to_numpy()
        a = panel[["id", "mes", "edad", "estado"]]
        b = panel[["id", "mes", "estado"]].rename(columns={"estado": "sig"}).assign(mes=lambda d: d["mes"] - 1)
        p = a.merge(b, on=["id", "mes"])
        p = p[p["estado"].between(0, 3)]
        idx = p["id"].to_numpy() - 1
        p = p[(p["edad"].to_numpy() + 1) != plazo[idx]]     # el último mes lo decide el vencimiento
        idx = p["id"].to_numpy() - 1
        edad, mes, est, sig = (p[c].to_numpy() for c in ("edad", "mes", "estado", "sig"))
        X_est, X_edad = S[idx], base_edad(edad)
        X_mes, X_u = dummies_mes(mes % 12), (u_mes[mes] - U_REF)[:, None]
        v = orig[idx]
        coef = {}

        def ajusta(clave, filas, y, bloques, nombres):
            X = np.hstack([bl[filas] for bl in bloques])
            m = LogisticRegression(C=C, max_iter=500, tol=1e-6).fit(X, y[filas])
            w, pos = m.coef_[0], 0
            cd = {"intercepto": float(m.intercept_[0])}
            for nombre, bl in zip(nombres, bloques):
                k = bl.shape[1]
                cd[nombre] = w[pos: pos + k].copy()
                pos += k
            coef[clave] = cd

        # cierre anticipado desde "al corriente"
        f = est == 0
        ajusta("cierre", f, (sig == 5).astype(int), [X_est, X_edad], ["estatico", "edad"])
        # entrada en mora desde "al corriente" entre los que no se cierran
        f = (est == 0) & (sig != 5)
        ajusta("entrada", f, (sig == 1).astype(int),
               [X_est, dummies_cosecha(v), X_edad, X_mes, X_u], ["estatico", "cosecha", "edad", "mes", "u"])
        e = coef["entrada"]
        n_v = np.bincount(orig, minlength=MESES_ORIGEN)
        efecto = np.concatenate([[0.0], e["cosecha"]])
        media = float(np.average(efecto, weights=n_v))
        e["intercepto"] += media
        e["cosecha"] = efecto - media          # efecto de cada cosecha contra la cosecha promedio
        e["mes"] = np.concatenate([[0.0], e["mes"]])
        # empeoramiento desde 1-29, 30-59 y 60-89
        ef_obs = e["cosecha"][v][:, None]
        for k in (1, 2, 3):
            f = est == k
            ajusta(f"sube{k}", f, (sig == k + 1).astype(int),
                   [X_est, ef_obs, X_edad, X_mes, X_u], ["estatico", "cosecha_pend", "edad", "mes", "u"])
            coef[f"sube{k}"]["mes"] = np.concatenate([[0.0], coef[f"sube{k}"]["mes"]])
        baja = {}
        for k in (2, 3):
            f = (est == k) & (sig != k + 1) & (sig != 5)
            baja[k] = float((sig[f] == k - 1).mean())
        return cls(coef, baja)

    # --- persistencia -----------------------------------------------------
    def a_tabla(self) -> pd.DataFrame:
        filas = []
        for clave, cd in self.coef.items():
            for termino, valor in cd.items():
                for i, x in enumerate(np.atleast_1d(valor)):
                    filas.append({"modelo": clave, "termino": termino, "indice": i, "coef": float(x)})
        for k, d in self.baja.items():
            filas.append({"modelo": f"baja{k}", "termino": "fraccion", "indice": 0, "coef": d})
        return pd.DataFrame(filas)

    @classmethod
    def de_tabla(cls, t: pd.DataFrame):
        coef, baja = {}, {}
        for (clave, termino), g in t.groupby(["modelo", "termino"]):
            valores = g.sort_values("indice")["coef"].to_numpy()
            if clave.startswith("baja"):
                baja[int(clave[-1])] = float(valores[0])
            else:
                coef.setdefault(clave, {})[termino] = float(valores[0]) if termino == "intercepto" else valores
        return cls(coef, baja)

    @property
    def efecto_cosecha(self) -> np.ndarray:
        return self.coef["entrada"]["cosecha"]

    # --- proyección -------------------------------------------------------
    def _estatico(self, clave, S, v, con_cosecha):
        c = self.coef[clave]
        r = c["intercepto"] + S @ c["estatico"]
        if con_cosecha:
            if clave == "entrada":
                r = r + c["cosecha"][v]
            elif clave.startswith("sube"):
                r = r + c["cosecha_pend"][0] * self.efecto_cosecha[v]
        return r

    def _dinamico(self, clave, a, cmes, u):
        c = self.coef[clave]
        r = base_edad(a) @ c["edad"]
        if "mes" in c:
            r = r + c["mes"][cmes]
        if "u" in c:
            r = r + c["u"][0] * (u - U_REF)
        return r

    def proyectar(self, pres: pd.DataFrame, estado0, edad0, mes0, U, H: int, con_cosecha: bool = True):
        """Flujos mensuales de incumplimiento y de cierre durante H meses.

        pres: atributos de los créditos (mismo orden que estado0 y edad0).
        mes0: mes calendario (0 = ene 2022) del estado inicial.
        U: desempleo del mes de inicio de cada paso (longitud H).
        Devuelve (dD, dC): matrices (créditos, H) con la probabilidad de entrar
        a incumplimiento y de cerrarse en cada mes."""
        S, v = estaticas(pres), pres["mes_origen"].to_numpy()
        plazo = pres["plazo"].to_numpy()
        n = len(pres)
        L = {k: self._estatico(k, S, v, con_cosecha) for k in self.coef}
        estado0, edad0 = np.asarray(estado0), np.asarray(edad0)
        P = [(estado0 == k).astype(float) for k in range(4)]
        dD, dC = np.zeros((n, H)), np.zeros((n, H))
        d2, d3 = self.baja[2], self.baja[3]
        for t in range(H):
            a = edad0 + t
            cm = (mes0 + t) % 12
            u = U[t]
            pc = sigmoide(L["cierre"] + self._dinamico("cierre", a, cm, u))
            p0 = sigmoide(L["entrada"] + self._dinamico("entrada", a, cm, u))
            p1, p2, p3 = (sigmoide(L[f"sube{k}"] + self._dinamico(f"sube{k}", a, cm, u)) for k in (1, 2, 3))
            P0, P1, P2, P3 = P
            n0 = P0 * (1 - pc) * (1 - p0) + P1 * (1 - p1) + P2 * (1 - p2) * (1 - d2) + P3 * (1 - p3) * (1 - d3)
            n1 = P0 * (1 - pc) * p0 + P2 * (1 - p2) * d2
            n2 = P1 * p1 + P3 * (1 - p3) * d3
            n3 = P2 * p2
            dD[:, t] = P3 * p3
            dC[:, t] = P0 * pc
            # vencimiento: 0 y 1-29 se cierran; 30 días o más pasan a incumplimiento
            vence = ((a + 1) == plazo)
            dC[:, t] += np.where(vence, n0 + n1, 0.0)
            dD[:, t] += np.where(vence, n2 + n3, 0.0)
            P = [np.where(vence, 0.0, x) for x in (n0, n1, n2, n3)]
        return dD, dC
