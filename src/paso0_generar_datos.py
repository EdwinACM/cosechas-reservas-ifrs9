"""Paso 0. Genera el portafolio sintético de préstamos de consumo.

TODOS LOS DATOS SON SINTÉTICOS. No hay información de ningún banco ni de sus
clientes; el generador inventa los créditos con una semilla fija.

Qué simula
- 50,000 créditos de consumo de pago fijo mensual, originados de enero de 2022 a
  diciembre de 2024 (36 cosechas) y observados hasta diciembre de 2025.
- Atributos al origen: score, carga de pago sobre ingreso (dti), canal, plazo,
  monto y tasa (la tasa sube cuando el score baja).
- Cada mes el crédito se mueve entre estados de mora: al corriente, 1-29,
  30-59, 60-89 y 90+ días. El estado 90+ es el incumplimiento y no se revierte.
  Un crédito al corriente también puede liquidarse antes (prepago). Al llegar
  al plazo, el que está al corriente o con 1-29 días se cierra y el que tiene
  30 días o más pasa a incumplimiento.
- La probabilidad de empeorar depende de un riesgo individual (score, dti,
  canal, plazo y un factor no observable), de la antigüedad (los créditos
  muy nuevos casi no fallan, luego el riesgo se asienta), de la estacionalidad
  (enero y febrero pesan más, diciembre menos por el aguinaldo) y del
  desempleo del mes. El desempleo sube desde finales de 2023 hasta un pico en abril de
  2024 y vuelve a subir en los últimos meses de 2025.
- Deterioro de calidad: las cosechas de marzo a septiembre de 2023 (pico en
  junio) y, más leve, las de mayo a septiembre de 2024 (pico en julio) se originan con peor score,
  más dti, más canal digital y un riesgo extra que no se ve en los atributos.
- Tras el incumplimiento se recupera una parte del saldo durante hasta 24
  meses; la recuperación total baja cuando el desempleo es alto.

Salidas en datos/: prestamos.csv, macro.csv y panel.csv.gz (una fila por
crédito y mes; el panel se regenera y no se versiona)."""
import io
import json

import numpy as np
import pandas as pd

from comun import (CANALES, COSECHAS_DETERIORADAS, ANCHO_DETERIORO, DATOS,
                   MESES_OBS, MESES_ORIGEN, MESES_RECUPERACION, N_PRESTAMOS,
                   PLAZOS, SEMILLA, saldo_programado, sigmoide)

# Parámetros del generador (todos inventados)
PARAMETROS = {
    "A0_entrada_mora": -4.75,
    "A_ascenso": {"1": -0.55, "2": 0.25, "3": 0.95},
    "gamma_ascenso": 0.6,
    "prob_baja_un_bucket": {"2": 0.15, "3": 0.10},
    "beta_score": -0.55, "beta_dti": 0.30, "beta_plazo": 0.10,
    "efecto_canal": {"nomina": -0.35, "sucursal": 0.0, "digital": 0.30},
    "sigma_fragilidad": 0.45, "sigma_cosecha": 0.06,
    "beta_desempleo": 0.35,
    "estacionalidad": [0.20, 0.10, 0.0, 0.0, 0.0, -0.02, 0.05, 0.05, 0.05, 0.0, -0.08, -0.20],
    "recuperacion_media": 0.40, "recuperacion_pend_desempleo": -0.06,
}


def macro(rng: np.random.Generator) -> pd.DataFrame:
    t = np.arange(MESES_OBS)
    ruido = np.zeros(MESES_OBS)
    for i in range(1, MESES_OBS):
        ruido[i] = 0.7 * ruido[i - 1] + rng.normal(0, 0.08)
    u = 3.6 + 1.7 * np.exp(-0.5 * ((t - 27) / 5) ** 2) + 0.9 * sigmoide((t - 42) / 1.5) + ruido
    return pd.DataFrame({"mes": t, "desempleo": u.round(3)})


def efecto_cosecha(rng: np.random.Generator) -> np.ndarray:
    """Riesgo extra de cada cosecha (logit). Ruido pequeño más los episodios."""
    v = rng.normal(0, PARAMETROS["sigma_cosecha"], MESES_ORIGEN)
    for pico, tam in COSECHAS_DETERIORADAS.items():
        ancho = ANCHO_DETERIORO[pico]
        for m in range(MESES_ORIGEN):
            v[m] += tam * max(0.0, 1 - abs(m - pico) / ancho)
    return v


def generar_prestamos(n: int, rng: np.random.Generator, v_cosecha: np.ndarray) -> pd.DataFrame:
    m = np.arange(MESES_ORIGEN)
    estacion_orig = 1 + 0.15 * np.isin(m % 12, [10, 11]) - 0.10 * np.isin(m % 12, [0, 1])
    pesos = (1 + 0.012 * m) * estacion_orig
    conteo = np.floor(pesos / pesos.sum() * n).astype(int)
    conteo[: n - conteo.sum()] += 1
    orig = np.repeat(m, conteo)
    # Intensidad del deterioro observable (0 a 1) en la cosecha de cada crédito
    exceso = np.clip(v_cosecha[orig] - 0.0, 0, None)
    intens = np.clip(exceso / 0.40, 0, 1)

    score = np.clip(rng.normal(655 - 30 * intens, 62), 450, 850).round()
    dti = np.clip(rng.beta(2.6, 7.8, n) + 0.035 * intens, 0.03, 0.70)
    p_dig = 0.25 + 0.14 * intens
    u = rng.random(n)
    canal = np.where(u < p_dig, "digital", np.where(u < p_dig + (1 - p_dig) * 0.53, "nomina", "sucursal"))
    plazo = rng.choice(PLAZOS, n, p=[0.2, 0.4, 0.4])
    monto = np.clip(rng.lognormal(np.log(32_000), 0.6, n), 5_000, 250_000).round(-2)
    tasa_a = np.clip(0.26 + (690 - score) * 0.0009 + rng.normal(0, 0.015, n), 0.18, 0.62)

    sd = PARAMETROS
    z = (sd["beta_score"] * (score - 650) / 60 + sd["beta_dti"] * (dti - 0.25) / 0.10
         + np.vectorize(sd["efecto_canal"].get)(canal)
         + sd["beta_plazo"] * (plazo - 24) / 12
         + rng.normal(0, sd["sigma_fragilidad"], n) + v_cosecha[orig])
    df = pd.DataFrame({"id": np.arange(1, n + 1), "mes_origen": orig, "score": score.astype(int),
                       "dti": dti.round(4), "canal": canal, "plazo": plazo, "monto": monto,
                       "tasa_anual": tasa_a.round(4)})
    df["_z"] = z
    return df


def g_edad(a):
    """Efecto de la antigüedad: casi nulo al inicio, luego se asienta."""
    return -1.8 * np.exp(-a / 3.0) - 0.012 * np.maximum(a - 12, 0)


def simular(pres: pd.DataFrame, u_macro: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    p = PARAMETROS
    n = len(pres)
    orig = pres["mes_origen"].to_numpy()
    plazo = pres["plazo"].to_numpy()
    monto = pres["monto"].to_numpy(float)
    r_m = pres["tasa_anual"].to_numpy() / 12
    z = pres["_z"].to_numpy()
    score_z = (pres["score"].to_numpy() - 650) / 60
    digital = (pres["canal"].to_numpy() == "digital")
    estado = np.zeros(n, int)
    ult_ok = monto.copy()
    ead = np.zeros(n)
    mes_inc = np.full(n, -1)
    r_tot = np.zeros(n)
    rec_acum = np.zeros(n)
    estac = np.array(p["estacionalidad"])
    a_asc = p["A_ascenso"]
    k = np.arange(1, MESES_RECUPERACION + 1)
    w = k * np.exp(-k / 4.0)
    w = w / w.sum()

    filas = []
    cerrado_mes = np.full(n, -1)

    for t in range(MESES_OBS):
        rec_mes = np.zeros(n)
        if t > 0:
            u_ant, c_ant = u_macro[t - 1], (t - 1) % 12
            s = p["beta_desempleo"] * (u_ant - 4.0) + estac[c_ant]
            act = (orig < t) & (estado <= 3)
            a_prev = (t - 1 - orig)
            nuevo = estado.copy()
            # estado 0: prepago o entrada en mora
            m0 = act & (estado == 0)
            p_cierre = sigmoide(-5.0 + 0.3 * score_z + 0.03 * np.minimum(a_prev, 12))
            p_ent = sigmoide(p["A0_entrada_mora"] + z + s + g_edad(a_prev))
            cierra = m0 & (rng.random(n) < p_cierre)
            entra = m0 & ~cierra & (rng.random(n) < p_ent)
            nuevo[cierra] = 5
            nuevo[entra] = 1
            # estados 1 a 3: ascenso, baja un bucket o cura
            for kst in (1, 2, 3):
                mk = act & (estado == kst)
                sube = mk & (rng.random(n) < sigmoide(a_asc[str(kst)] + p["gamma_ascenso"] * (z + s)))
                resto = mk & ~sube
                baja = resto & (rng.random(n) < p["prob_baja_un_bucket"].get(str(kst), 0.0))
                nuevo[sube] = kst + 1
                nuevo[resto] = 0
                nuevo[baja] = kst - 1
            # llegada al plazo
            termina = act & ((t - orig) == plazo) & (nuevo <= 3)
            cierra_pl, inc_pl = termina & (nuevo <= 1), termina & (nuevo >= 2)
            nuevo[cierra_pl] = 5
            nuevo[inc_pl] = 4
            # incumplimientos nuevos: congelar saldo y sortear la recuperación total
            nuevos_inc = (nuevo == 4) & (estado != 4)
            if nuevos_inc.any():
                ead[nuevos_inc] = ult_ok[nuevos_inc]
                mes_inc[nuevos_inc] = t
                base = (np.log(p["recuperacion_media"] / (1 - p["recuperacion_media"]))
                        + 0.2 * score_z[nuevos_inc] - 0.15 * digital[nuevos_inc]
                        + rng.normal(0, 0.6, nuevos_inc.sum()))
                ajuste = np.maximum(1 + p["recuperacion_pend_desempleo"] * (u_macro[t] - 3.6), 0.3)
                r_tot[nuevos_inc] = sigmoide(base) * ajuste
            cerrado_mes[(nuevo == 5) & (estado != 5)] = t
            estado = nuevo
            ok0 = estado == 0
            ult_ok[ok0] = saldo_programado(monto[ok0], r_m[ok0], plazo[ok0], (t - orig[ok0]))
            # recuperaciones de créditos ya incumplidos
            rel = t - mes_inc
            cobra = (estado == 4) & (mes_inc >= 0) & (rel >= 1) & (rel <= MESES_RECUPERACION)
            rec_mes[cobra] = ead[cobra] * r_tot[cobra] * w[rel[cobra] - 1]
            rec_acum += rec_mes
        # filas del panel: créditos originados y no cerrados antes de este mes
        idx = np.flatnonzero((orig <= t) & ((cerrado_mes == -1) | (cerrado_mes == t)))
        e = estado[idx]
        saldo = np.where(e == 0, saldo_programado(monto[idx], r_m[idx], plazo[idx], t - orig[idx]),
                         np.where(e == 5, 0.0, np.where(e == 4, ead[idx], ult_ok[idx])))
        filas.append(pd.DataFrame({
            "id": idx + 1, "mes": t, "edad": t - orig[idx], "estado": e,
            "saldo": saldo.round(2), "recup": rec_mes[idx].round(2),
            "recup_acum": rec_acum[idx].round(2)}))
    return pd.concat(filas, ignore_index=True)


def generar(n: int = N_PRESTAMOS, semilla: int = SEMILLA):
    rng = np.random.default_rng(semilla)
    mac = macro(rng)
    v = efecto_cosecha(rng)
    pres = generar_prestamos(n, rng, v)
    panel = simular(pres, mac["desempleo"].to_numpy(), rng)
    return pres.drop(columns="_z"), panel, mac


def main() -> None:
    DATOS.mkdir(exist_ok=True)
    pres, panel, mac = generar()
    pres.to_csv(DATOS / "prestamos.csv", index=False)
    mac.to_csv(DATOS / "macro.csv", index=False)
    # Se arma en memoria y se escribe de una vez (más estable en carpetas sincronizadas)
    memoria = io.BytesIO()
    panel.to_csv(memoria, index=False, compression={"method": "gzip", "mtime": 0})
    (DATOS / "panel.csv.gz").write_bytes(memoria.getvalue())
    (DATOS / "parametros_generador.json").write_text(
        json.dumps({"semilla": SEMILLA, "n_prestamos": N_PRESTAMOS, **PARAMETROS}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print(f"{len(pres):,} créditos, {len(panel):,} filas de panel")
    print(panel[panel["edad"] == 0].shape[0], "créditos originados")
    ult = panel.sort_values("mes").groupby("id").tail(1)
    print("Estado final:", ult["estado"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
