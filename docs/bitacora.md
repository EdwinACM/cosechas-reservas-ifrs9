# Bitácora de decisiones

Semilla 42 en todos los pasos. Todos los datos son sintéticos.

## Datos sintéticos

1. Se generan 50,000 créditos con cuota fija en 36 cosechas y se observan 48 meses. El panel mensual (un renglón por crédito y mes) pesa unos 8 MB comprimido y se regenera con el paso 0, por eso no se versiona.
2. El generador es una cadena de estados mensual con probabilidades logísticas. Dependen de score, carga de pago, canal, plazo, un factor individual no observable, antigüedad, mes del año y desempleo. Los parámetros están en `datos/parametros_generador.json`.
3. El deterioro de calidad se mete de dos formas: peor mezcla observable (score más bajo, más carga de pago, más canal digital) y un riesgo extra no observable por cosecha. Se hizo así para que el modelo con efectos de cosecha tenga algo que separar.
4. El desempleo sube desde finales de 2023 hasta un pico en abril de 2024 y tiene una subida final en 2025. La subida final se añadió después de una primera versión en la que el desempleo al corte era bajo y el criterio relativo de PD no clasificaba ningún crédito vigente al corriente en etapa 2.
5. Primera versión del generador con una regla de vencimiento mal escrita (marcaba como incumplimiento a los créditos que debían cerrarse). Se detectó porque 22,000 de 50,000 créditos terminaban en 90+. Se corrigió y se recalibraron la entrada a mora y el tamaño del deterioro hasta tener tasas de 90+ a 12 meses cercanas a 2% en cosechas normales.
6. Al escribir los archivos grandes en una carpeta sincronizada hubo errores de tiempo de espera. Los archivos se arman en memoria y se escriben de una vez.

## Cosechas

7. La mora se mide sobre lo originado en cada cosecha, acumulada (llegó alguna vez a 30+ o a 90+ en ese mes en libros) y no como foto del mes. También se calcula por monto y la mora 30+ en libros (foto).
8. Alerta de cosecha: razón de al menos 1.5 contra la mediana de las cosechas observadas al mismo MOB y prueba binomial unilateral con p menor a 0.001. Con umbral de 1.3 y p de 0.01 aparecían 12 cosechas en alerta, varias por el estrés macro. Se subió el umbral.
9. Queda una alerta (octubre de 2023) que el modelo atribuye al desempleo, no a la cosecha. Se dejó como ejemplo de que la comparación simple de curvas no separa cosecha y entorno.

## Transiciones

10. Los buckets son al corriente, 1-29, 30-59, 60-89 y 90+, más una columna de cerrado. Se miden a 1 y 12 meses, y por grupo (cosechas en alerta contra resto, desempleo alto contra normal).
11. Con cortes mensuales el bucket 1-29 no se queda igual: el crédito paga o pasa a 30-59. Es una propiedad del generador y se anota como límite.

## PD

12. Modelo de cuatro estados con regresiones logísticas para cierre, entrada a mora y empeoramiento desde cada bucket. El incumplimiento es 90+. Es el equivalente discreto de un modelo de riesgos con la antigüedad como base y el desempleo como covariable.
13. Las cosechas entran como efectos fijos en la entrada a mora y como un solo coeficiente sobre ese efecto en los empeoramientos, porque en los buckets con pocos casos 35 variables por cosecha serían ruido. El efecto se centra en la cosecha promedio.
14. El último mes de cada crédito se deja fuera del ajuste, porque lo decide el vencimiento y no el comportamiento.
15. La validación es dentro de muestra (observado contra esperado por cosecha a 12 y 24 meses, y AUC). No se apartó una muestra de prueba porque la cartera es una sola cadena de cohortes y el objetivo es ilustrar el método.

## LGD y EAD

16. Curva de recuperación por escalera con la recuperación mensual sobre la exposición en el incumplimiento, usando solo los incumplimientos que ya tienen ese mes observado. La LGD se descuenta a la tasa contractual por canal.
17. La sensibilidad de la recuperación al desempleo se estima con los incumplimientos de ventana completa (regresión del logaritmo de la recuperación total). Se usa para ajustar la LGD por escenario.
18. El rezago entre el primer pago fallido y el incumplimiento se eligió con los datos: el que deja la razón entre EAD estimada y observada más cerca de 1 (cuatro meses).
19. Los créditos con más de 24 meses en incumplimiento se consideran castigados y salen de la reserva.

## Reserva IFRS 9

20. SICR base: mora de 30 días o más (presunción de IFRS 9), o PD de vida remanente de al menos dos veces la que se habría estimado al origen para ese crédito al corriente a su edad actual, con un aumento mínimo de 0.5 puntos porcentuales. La PD al origen no incluye el efecto de cosecha, que el banco no conocía.
21. La clasificación usa el escenario base. La reserva por etapa usa los tres escenarios con pesos de 50%, 30% y 20%.
22. Etapa 1: pérdida esperada a 12 meses. Etapa 2: a vida. Etapa 3: saldo bruto menos lo cobrado menos el valor presente de lo que falta por recuperar según la curva y los meses transcurridos. No se modela cura de la etapa 3.
23. La pérdida esperada por crédito suma, mes a mes, la probabilidad de entrar a incumplimiento por la LGD por la EAD por el factor de descuento, con EAD de amortización contractual.
24. La matriz de etapas a 12 meses usa cinco cierres (diciembre de 2022 a diciembre de 2024) agrupados, con la clasificación recalculada en cada fecha.
25. Matriz de provisiones simplificada: incumplimiento a 12 meses histórico ponderado por saldo en cada bucket, por la LGD global. Para el bucket 90+ se aplica la LGD global al saldo bruto menos lo ya recuperado. Es una versión propia para comparar, no una norma.

## Sensibilidades

26. Escenarios por separado, peso del adverso de 0 a 60%, desplazamiento paralelo de la trayectoria base de menos 1 a más 3 puntos y una malla de SICR (mora desde 1 o 30 días, razón de PD de 1.25 a 3, aumento de 0.2 a 1 punto).
27. Se reporta por separado la reserva de etapas 1 y 2 porque la etapa 3 domina el total y esconde el efecto de las demás decisiones.
