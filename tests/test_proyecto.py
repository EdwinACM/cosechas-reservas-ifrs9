"""Pruebas de integridad. Se corren con: python -m pytest

Las primeras usan una cartera sintética pequeña generada al vuelo. Las del final
leen los resultados de `python src/ejecutar_todo.py` y se omiten si no existen."""
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np
import pandas as pd

from comun import MESES_ORIGEN, MESES_OBS, SALIDA, ruta_macro, saldo_programado
from modelo_pd import Modelo
from paso0_generar_datos import generar
from paso1_cosechas import curvas_acumuladas, primeros_eventos
from paso2_transiciones import matriz, pares
from reserva import asignar_etapa


@pytest.fixture(scope="module")
def pequena():
    return generar(n=6000, semilla=42)


def test_generador_es_reproducible():
    a = generar(n=1500, semilla=7)
    b = generar(n=1500, semilla=7)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_prestamos_bien_formados(pequena):
    pres, panel, macro = pequena
    assert len(pres) == 6000 and pres["id"].is_unique
    assert pres["mes_origen"].between(0, MESES_ORIGEN - 1).all()
    assert pres["mes_origen"].nunique() == MESES_ORIGEN
    assert pres["plazo"].isin([12, 24, 36]).all()
    assert pres["score"].between(450, 850).all()
    assert len(macro) == MESES_OBS and macro["desempleo"].between(2, 9).all()
    assert pres.isna().sum().sum() == 0


def test_panel_estados_y_saldos(pequena):
    pres, panel, _ = pequena
    assert set(panel["estado"].unique()) <= {0, 1, 2, 3, 4, 5}
    assert (panel["saldo"] >= 0).all() and (panel["recup"] >= 0).all()
    # todos los créditos aparecen en su mes de originación con estado 0 y saldo igual al monto
    nuevo = panel[panel["edad"] == 0].merge(pres, on="id")
    assert len(nuevo) == len(pres)
    assert (nuevo["estado"] == 0).all()
    assert np.allclose(nuevo["saldo"], nuevo["monto"], atol=0.01)


def test_incumplimiento_y_cierre_son_absorbentes(pequena):
    _, panel, _ = pequena
    p = panel.sort_values(["id", "mes"])
    p["sig"] = p.groupby("id")["estado"].shift(-1)
    assert (p.loc[p["estado"] == 4, "sig"].dropna() == 4).all()
    assert (p["estado"] == 5).groupby(p["id"]).sum().max() == 1      # un solo mes en cerrado


def test_recuperaciones_no_superan_la_exposicion(pequena):
    _, panel, _ = pequena
    inc = panel[panel["estado"] == 4].groupby("id").agg(ead=("saldo", "max"), rec=("recup_acum", "max"))
    assert (inc["rec"] <= inc["ead"] + 0.01).all()


def test_saldo_programado():
    edades = np.arange(0, 37)
    s = saldo_programado(10_000, 0.03, 36, edades)
    assert abs(s[0] - 10_000) < 1e-6 and abs(s[-1]) < 1e-6
    assert (np.diff(s) < 0).all()


def test_curvas_de_cosechas_son_acumuladas(pequena):
    pres, panel, _ = pequena
    c90 = curvas_acumuladas(pres, primeros_eventos(panel), "mob_90")
    for v in range(MESES_ORIGEN):
        fila = c90.loc[v].dropna()
        assert len(fila) == MESES_OBS - v
        assert (np.diff(fila.to_numpy()) >= -1e-12).all()
        assert fila.between(0, 1).all()
    # 30+ siempre llega antes que 90+
    c30 = curvas_acumuladas(pres, primeros_eventos(panel), "mob_30")
    assert (c30.fillna(0).to_numpy() >= c90.fillna(0).to_numpy() - 1e-12).all()


def test_filas_de_matrices_suman_uno(pequena):
    _, panel, _ = pequena
    tasa, cuenta = matriz(pares(panel, 1))
    assert np.allclose(tasa.sum(axis=1).dropna(), 1.0)
    # el incumplimiento no se revierte
    assert tasa.loc["90+", "90+"] == 1.0
    tasa12, _ = matriz(pares(panel, 12))
    assert np.allclose(tasa12.sum(axis=1).dropna(), 1.0)


def test_modelo_pd_probabilidades_coherentes(pequena):
    pres, panel, macro = pequena
    m = Modelo.ajustar(pres, panel, macro)
    otro = Modelo.de_tabla(m.a_tabla())               # ida y vuelta por tabla
    g = pres[pres["mes_origen"] == 10].reset_index(drop=True)
    u = macro["desempleo"].to_numpy()
    n = len(g)
    estados = np.repeat([0, 1, 2, 3], n)
    gg = pd.concat([g] * 4, ignore_index=True)
    U = np.concatenate([[u[30]], ruta_macro(u[30], "base", 35)])
    dD, dC = m.proyectar(gg, estados, np.full(4 * n, 6), 30, U, 36)
    dD2, _ = otro.proyectar(gg, estados, np.full(4 * n, 6), 30, U, 36)
    assert np.allclose(dD, dD2)
    assert (dD >= 0).all() and (dC >= 0).all()
    assert ((dD + dC).sum(axis=1) <= 1 + 1e-9).all()
    pd_vida = dD.sum(axis=1).reshape(4, n).mean(axis=1)
    assert (np.diff(pd_vida) > 0).all()               # más mora, más PD


def test_pd_sube_con_el_escenario_adverso(pequena):
    pres, panel, macro = pequena
    m = Modelo.ajustar(pres, panel, macro)
    g = pres[pres["mes_origen"] == 20].reset_index(drop=True)
    z = np.zeros(len(g), int)
    res = {}
    for esc in ["optimista", "base", "adverso"]:
        U = np.concatenate([[3.8], ruta_macro(3.8, esc, 35)])
        res[esc] = m.proyectar(g, z, z + 5, 25, U, 36)[0].sum()
    assert res["optimista"] < res["base"] < res["adverso"]


def test_escenarios_macro_ordenados():
    o, b, a = (ruta_macro(4.5, e) for e in ["optimista", "base", "adverso"])
    assert (o <= b + 1e-9).all() and (b <= a + 1e-9).all()


def test_asignar_etapa():
    estado = np.array([0, 0, 2, 3])
    pd_v = np.array([0.02, 0.10, 0.03, 0.60])
    pd_o = np.array([0.02, 0.03, 0.03, 0.04])
    assert list(asignar_etapa(estado, pd_v, pd_o)) == [1, 2, 2, 2]
    # sin criterio relativo solo cuenta la mora
    assert list(asignar_etapa(estado, pd_v, pd_o, razon_pd=np.inf)) == [1, 1, 2, 2]
    # un aumento relativo grande pero diminuto en puntos no basta
    assert list(asignar_etapa(np.array([0]), np.array([0.004]), np.array([0.001]))) == [1]


# --- Pruebas sobre los resultados del pipeline completo ---
hay_resultados = (SALIDA / "reserva_por_etapa.csv").exists() and (SALIDA / "cosechas_resumen.csv").exists()
requiere = pytest.mark.skipif(not hay_resultados, reason="falta correr src/ejecutar_todo.py")


@requiere
def test_reserva_por_etapa_cuadra():
    t = pd.read_csv(SALIDA / "reserva_por_etapa.csv").set_index("etapa")
    partes = t.loc[["etapa 1", "etapa 2", "etapa 3"]]
    assert np.isclose(partes["reserva_ponderada"].sum(), t.loc["total", "reserva_ponderada"])
    assert np.isclose(partes["saldo"].sum(), t.loc["total", "saldo"])
    # ponderada entre el mejor y el peor escenario
    assert t.loc["total", "reserva_optimista"] <= t.loc["total", "reserva_ponderada"] <= t.loc["total", "reserva_adverso"]
    # la cobertura crece con la etapa
    cob = partes["cobertura_ponderada"].to_numpy()
    assert (np.diff(cob) > 0).all()


@requiere
def test_cosechas_deterioradas_se_detectan():
    r = pd.read_csv(SALIDA / "cosechas_resumen.csv").set_index("cosecha")
    # los meses con deterioro inyectado deben aparecer en alerta
    assert r.loc[[16, 17, 18], "alerta"].all()
    # el comienzo de la cartera, no
    assert not r.loc[0:9, "alerta"].any()


@requiere
def test_reserva_12m_no_supera_la_de_vida():
    t = pd.read_csv(SALIDA / "sensibilidad_cotas_clasificacion.csv").set_index("supuesto")
    a, b = t.loc["toda la cartera vigente a 12 meses", "reserva_etapas12"], t.loc["toda la cartera vigente a vida", "reserva_etapas12"]
    assert a < b
    base = t.loc["clasificación base", "reserva_etapas12"]
    assert a <= base <= b


@requiere
def test_matriz_de_etapas_filas_suman_uno():
    t = pd.read_csv(SALIDA / "transicion_etapas_12m.csv", index_col=0)
    assert np.allclose(t.sum(axis=1), 1.0)
