# Cosechas y reserva de pérdida esperada IFRS 9 con datos sintéticos

Proyecto personal de riesgo de crédito sobre un portafolio de préstamos de consumo. Va desde las cosechas (vintages) de mora hasta la reserva por etapas: curvas de mora 30+ y 90+ por cosecha, matrices de transición, PD por etapa y a vida con escenarios macro, LGD con recuperaciones, EAD por amortización, reserva de pérdida esperada IFRS 9, sensibilidades y comparación con el cálculo simplificado por buckets. Corre con un comando y da los mismos resultados cada vez (semilla 42).

## Aviso sobre los datos y el alcance

Los datos son sintéticos. Los genera `src/paso0_generar_datos.py` con parámetros inventados; no hay información de ningún banco ni de sus clientes, y no se descargó nada. Los montos están en pesos ficticios.

Los resultados dependen de cómo se programó el generador, así que sirven para ver si el método recupera lo que el generador puso, en una cartera donde se conoce la respuesta. No dicen nada sobre ninguna cartera real. Tampoco sustituyen el cumplimiento regulatorio de ninguna institución: la Circular Única de Bancos (CUB) de la CNBV tiene su propia metodología de calificación y reservas, y este ejercicio no la reproduce.

## Cómo se genera el portafolio

50,000 créditos de pago fijo mensual, originados de enero de 2022 a diciembre de 2024 (36 cosechas) y observados hasta diciembre de 2025, que es la fecha de corte. Cada crédito tiene score, carga de pago sobre ingreso, canal (nómina, sucursal, digital), plazo de 12, 24 o 36 meses, monto y una tasa que sube cuando el score baja.

Cada mes el crédito se mueve entre cinco estados de mora (al corriente, 1-29, 30-59, 60-89 y 90+ días). Llegar a 90+ es el incumplimiento y ya no se revierte. Un crédito al corriente también puede prepagarse, y al vencimiento el que tiene 30 días o más pasa a incumplimiento. La probabilidad de empeorar sube con el riesgo individual (incluye un factor no observable), cambia con la antigüedad del crédito, con la estacionalidad (enero y febrero pesan más, diciembre menos por el aguinaldo) y con el desempleo del mes. El desempleo sube desde finales de 2023 hasta un pico en abril de 2024 y vuelve a subir al final de 2025. Tras el incumplimiento se recupera parte del saldo durante hasta 24 meses, y menos si el desempleo es alto.

Dos episodios de deterioro de calidad dan algo que detectar: las cosechas de marzo a septiembre de 2023 (pico en junio, el efecto crece y decae de forma triangular) y, más leve, las de mayo a septiembre de 2024 (pico en julio). Esas cosechas se originan con peor score, más carga de pago y más canal digital, y además traen un riesgo extra que no se ve en los atributos. Los pasos de análisis no leen dónde se inyectó el deterioro; las pruebas comprueban que lo encuentren. Los parámetros están en `datos/parametros_generador.json`.

## Qué encontré

Las cifras salen de la corrida de `ejecutar_todo.py`.

Cosechas. Las cosechas sin problema llegan a 90+ en 1.9% de los créditos a los 12 meses en promedio. La de junio de 2023 llega a 6.7%, casi tres veces la mediana (2.26%). La comparación contra la mediana, con una prueba binomial, enciende alerta en las cosechas de abril a agosto de 2023, en la de octubre de 2023 y en la de julio de 2024. La mora 30+ avisa antes: en mayo, junio y julio de 2023 ya había alerta a los 6 meses en libros, mientras que la de 90+ tarda hasta los 12.

Qué es cosecha y qué es macro. La cosecha de octubre de 2023 se enciende en la comparación simple, pero cumplió sus 12 meses en pleno estrés de desempleo. El modelo de PD le asigna un efecto de cosecha casi nulo (razón de momios de 1.02), así que su mora más alta se explica por el entorno. Las de mayo a julio de 2023 tienen efectos de 1.34 a 1.45 y la de julio de 2024 de 1.24. Sin el efecto de cosecha, el modelo espera entre 25% y 33% menos mora a 12 meses de la que se observó en esas cosechas; con él, la razón observado/esperado queda entre 0.93 y 1.02.

Transiciones. De un mes al siguiente, un crédito al corriente cae en 1-29 días el 1.1% de las veces. Quien está en 1-29 pasa a 30-59 el 49%, de 30-59 pasa a 60-89 el 69% y de 60-89 a 90+ el 84%. En las cosechas en alerta esos pasos suben a 1.6%, 55%, 74% y 87%. La cadena completa de al corriente a 90+ es 0.57% al mes en las cosechas en alerta contra 0.25% en el resto.

PD. El modelo de cuatro estados tiene PD a 12 meses promedio de 2.44% contra 2.47% observado, y un AUC dentro de muestra de 0.81. Al corte, la PD a 12 meses promedio de la etapa 1 es 2.9% en el escenario base, 4.2% en el adverso y 2.6% en el optimista; la de etapa 2 ronda 38%.

LGD y EAD. La LGD descontada global es 72.6% (66.3% sin descontar) y la realizada en los incumplimientos con ventana completa es 71.3%. Digital pierde más (74.4%) que nómina (71.1%). Cada punto de desempleo reduce la recuperación total en alrededor de 9% (coeficiente estimado de -0.094 en logaritmo, con error estándar de 0.042). El saldo congelado al incumplir se describe mejor con la amortización contractual cuatro meses antes del incumplimiento.

Reserva al corte (diciembre de 2025), ponderando base 50%, adverso 30% y optimista 20%:

| Etapa | Créditos | Saldo (millones) | Reserva (millones) | Cobertura |
|---|---|---|---|---|
| 1 | 16,154 | 266.8 | 4.60 | 1.7% |
| 2 | 590 | 9.3 | 2.57 | 27.6% |
| 3 | 2,668 | 47.8 | 43.57 | 91.2% |
| Total | 19,412 | 323.9 | 50.75 | 15.7% |

El saldo de la etapa 3 está neto de lo ya recuperado. Quedan fuera 397 créditos con más de 24 meses en incumplimiento, que se consideran castigados (7.1 millones netos). La etapa 3 concentra casi toda la reserva, porque en la simulación los incumplidos permanecen en libros hasta cumplir 24 meses.

Sensibilidad a escenarios. Con el escenario adverso por separado la reserva de etapas 1 y 2 sube 30.6% (8.68 millones contra 6.65 del base) y con el optimista baja 6.4%. La reserva ponderada queda 7.9% arriba de la base (7.17 millones), porque el escenario adverso pesa más de lo que el optimista compensa. Un desplazamiento paralelo de +1 punto en el desempleo base sube esa reserva a 8.98 millones y uno de +3 puntos a 16.95 millones. Sobre el total, incluida la etapa 3, el escenario adverso suma 4.9%.

Sensibilidad al SICR. Las definiciones probadas (mora desde 1 día o desde 30 días, razón de PD de 1.25 a 3 veces, aumento mínimo de 0.2 a 1 punto) mueven el saldo en etapa 2 de 4.2 a 38.4 millones, pero la reserva de etapas 1 y 2 cambia como máximo 1.4%. La razón es que los créditos tienen plazos cortos y la reserva a 12 meses se parece a la de vida. Como cota, pasar toda la cartera vigente a 12 meses daría 7.15 millones y pasarla toda a vida daría 8.15 millones, contra 7.17 con la clasificación base. El criterio sí cambia el comportamiento de las etapas a lo largo del tiempo: entre los créditos al corriente de las cosechas en alerta, el 32% (cierre de diciembre de 2023) y el 40% (junio de 2024) habrían estado en etapa 2 por deterioro relativo de PD, contra casi ninguno en el resto. En la matriz de etapas a 12 meses, el paso de etapa 1 a etapa 2 es de 11.7% en las cosechas en alerta y de 1.3% en el resto.

Contra la matriz de provisiones. El cálculo simplificado (incumplimiento a 12 meses histórico por bucket, por la LGD promedio) da 39.8 millones en total contra 50.7 de IFRS 9. En la cartera vigente ocurre lo contrario: la matriz da 9.57 millones contra 7.17, porque no amortiza la exposición ni descuenta y promedia toda la historia. La cobertura de la matriz de provisiones por cosecha va de 2.6% a 4.5%, y la de IFRS 9 de 0.6% a 4.3%, así que IFRS 9 varía más entre cosechas y es la única que reacciona al escenario. En el bucket 90+ la matriz queda 13.3 millones abajo, porque aplica la LGD del día del incumplimiento sin considerar los meses transcurridos.

## Límites

- Los datos son sintéticos y el modelo se ajusta con la misma estructura que usó el generador (cadena de estados con regresiones logísticas). En una cartera real el ajuste probablemente sería peor y habría más trabajo de datos.
- La validación del modelo es dentro de muestra. No hay una muestra de prueba separada.
- El efecto de cosecha usa información posterior al origen, así que en la práctica sirve para diagnóstico y reserva, no para fijar el SICR al otorgar el crédito. En el SICR se compara contra una PD al origen calculada sin ese efecto.
- La etapa 3 no tiene reversión (cura) ni periodo de prueba. Un crédito en 90+ no regresa a otra etapa.
- El estado "1-29 días" nunca se queda igual de un mes a otro: o cura o pasa a 30-59. Es un efecto de cómo se programó el generador con cortes mensuales.
- Las trayectorias de desempleo de los tres escenarios y sus ponderaciones son supuestos propios y se pueden cambiar en `src/comun.py`. Un modelo real vincularía varios indicadores macro y revisaría las ponderaciones con el área de economía.
- La EAD es la amortización contractual, sin líneas por disponer, con un rezago de cuatro meses estimado de los propios incumplimientos. La LGD descuenta a la tasa contractual de la cartera, que es alta.
- La matriz de provisiones del paso 5 es una versión propia, con supuestos simples para el bucket 90+. No es el esquema de ninguna norma.
- No hay ajustes de gestión, intereses devengados ni cobertura de garantías.

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/ejecutar_todo.py
python -m pytest
```

El pipeline tarda unos 20 segundos. Una acción de GitHub corre las pruebas en cada cambio. Los pasos, en orden:

- `paso0_generar_datos.py`: genera los préstamos, el panel mensual y el desempleo en `datos/`. El panel se regenera y no se versiona.
- `paso1_cosechas.py`: curvas de mora 30+ y 90+ por cosecha y alertas.
- `paso2_transiciones.py`: matrices de transición de buckets a 1 y 12 meses.
- `paso3_pd.py`: supervivencia por cosecha, modelo de PD de tiempo discreto y validación.
- `paso4_lgd_ead.py`: curva de recuperación, LGD por canal y EAD.
- `paso5_reserva.py`: etapas, reserva ponderada, matriz de etapas y comparación con la matriz de provisiones.
- `paso6_sensibilidad.py`: sensibilidad a escenarios, ponderaciones, desempleo y SICR.

`modelo_pd.py` y `reserva.py` son los módulos con el modelo y el motor de reserva. Los resultados quedan en `resultados/` como tablas CSV, un par de JSON y siete gráficas.

## Bitácora

`docs/bitacora.md` registra las decisiones del análisis, lo que se midió y los cambios que hubo durante el trabajo.

Licencia MIT.
