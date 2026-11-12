"""Constantes, escenarios y utilidades compartidas por todos los pasos.

Los datos son sintéticos. Los parámetros del generador viven aquí para que
el paso 0 y las pruebas usen exactamente los mismos."""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"
SALIDA = RAIZ / "resultados"
SEMILLA = 42

# Calendario: 36 cosechas mensuales (ene 2022 a dic 2024) y observación
# hasta dic 2025 (mes 47). El mes 0 es enero de 2022.
N_PRESTAMOS = 50_000
MESES_ORIGEN = 36
MESES_OBS = 48
FECHA_INICIO = "2022-01-01"
MES_CORTE = MESES_OBS - 1          # fecha de reporte para la reserva

# Estados de mora al cierre de cada mes
ESTADOS = {0: "al corriente", 1: "1 a 29 días", 2: "30 a 59 días", 3: "60 a 89 días",
           4: "90 o más (incumplimiento)", 5: "cerrado"}
BUCKETS = ["al corriente", "1-29", "30-59", "60-89", "90+"]
CERRADO = 5

# Recuperaciones: se cobra hasta 24 meses después del incumplimiento y luego
# el crédito se castiga.
MESES_RECUPERACION = 24

# Cosechas con deterioro de calidad inyectado (se usan solo para generar los
# datos y para evaluar si el análisis las detecta; el análisis no las lee).
COSECHAS_DETERIORADAS = {17: 0.40, 30: 0.22}   # pico de cada episodio y su tamaño
ANCHO_DETERIORO = {17: 4, 30: 3}

CANALES = ["nomina", "sucursal", "digital"]
PLAZOS = [12, 24, 36]

# Escenarios macro (desempleo, %) y ponderaciones para la reserva
PESOS_ESCENARIOS = {"base": 0.50, "adverso": 0.30, "optimista": 0.20}
HORIZONTE_MAX = 36


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def saldo_programado(monto, tasa_m, plazo, edad):
    """Saldo de un crédito de cuota fija después de `edad` pagos."""
    monto, tasa_m, plazo, edad = (np.asarray(v, dtype=float) for v in (monto, tasa_m, plazo, edad))
    edad = np.minimum(edad, plazo)
    f_n, f_a = (1 + tasa_m) ** plazo, (1 + tasa_m) ** edad
    return monto * (f_n - f_a) / (f_n - 1)


def etiqueta_mes(mes: int) -> str:
    return str((pd.Timestamp(FECHA_INICIO) + pd.DateOffset(months=int(mes))).strftime("%Y-%m"))


def cargar():
    """Devuelve (préstamos, panel, macro). Hay que correr antes el paso 0."""
    prestamos = pd.read_csv(DATOS / "prestamos.csv")
    panel = pd.read_csv(DATOS / "panel.csv.gz")
    macro = pd.read_csv(DATOS / "macro.csv")
    return prestamos, panel, macro


def ruta_macro(u0: float, escenario: str, horizonte: int = HORIZONTE_MAX) -> np.ndarray:
    """Trayectoria de desempleo (%) de los meses 1..horizonte después de la fecha
    de reporte, a partir del último valor observado u0.

    base: regresa lentamente a 4.2%. adverso: sube 2.3 puntos en un año y regresa
    despacio. optimista: baja hasta 3.4%."""
    t = np.arange(1, horizonte + 1)
    if escenario == "base":
        return 4.2 + (u0 - 4.2) * 0.96 ** t
    if escenario == "adverso":
        pico = u0 + 2.3
        subida = u0 + (pico - u0) * np.minimum(t / 12, 1)
        return np.where(t <= 12, subida, 4.2 + (pico - 4.2) * 0.95 ** (t - 12))
    if escenario == "optimista":
        return 3.4 + (u0 - 3.4) * 0.93 ** t
    raise ValueError(escenario)


def escribir_csv(df: pd.DataFrame, ruta: Path, indice: bool = False) -> None:
    """Escribe el CSV de una vez desde memoria (más estable en carpetas sincronizadas)."""
    ruta.write_bytes(df.to_csv(index=indice).encode("utf-8"))


def guardar_figura(fig, ruta: Path) -> None:
    import io
    memoria = io.BytesIO()
    fig.savefig(memoria, format="png", dpi=140)
    ruta.write_bytes(memoria.getvalue())
