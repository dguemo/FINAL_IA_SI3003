# Reporte del modelo clásico (generado por `models/train.py`)

Generado: 2026-11-06T11:55:31  
Atributos: `log_pm25_t, log_pm25_t1, log_pm25_t2, log_media3, delta_1d, sin_anual, cos_anual, sin_semestral, cos_semestral, log_precip_t, viento_t, hr_t`  
Filas: 6063 | estaciones: 24 | rango: 2023-08-02 a 2026-11-04

## Validación cruzada por bloques temporales (fuera de pliegue)

### 3 clases (Normal / Prevención / Crítico)

| Modelo | F1 macro | IC95 % F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| naive_bayes | 0.465 | [0.446, 0.483] | 0.951 | 0.434 | 0.536 |
| persistencia | 0.522 | [0.500, 0.541] | 0.977 | 0.522 | 0.521 |
| clase_mayoritaria | 0.329 | [0.328, 0.329] | 0.972 | 0.324 | 0.333 |

### Tarea operativa: alerta (≥ 38 µg/m³) al día siguiente

| Modelo | F1 alerta | IC95 % | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| naive_bayes | 0.424 | [0.368, 0.476] | 0.315 | 0.649 | 109 | 237 | 59 |
| persistencia | 0.585 | [0.523, 0.646] | 0.587 | 0.583 | 98 | 69 | 70 |
| clase_mayoritaria | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0 | 0 | 168 |

McNemar NB vs Persistencia: b=60, c=216, p=0.0000

Matriz de confusión Naive Bayes (filas=real, columnas=predicho; Normal, Prevención, Crítico):

```
[5658, 237, 0]
[59, 108, 0]
[0, 1, 0]
```
Soporte: {'Normal': 5895, 'Prevención': 167, 'Crítico': 1}

F1 macro por pliegue NB: [1.0, 0.499, 0.45, 0.61, 0.499] (media 0.612 ± 0.201)

## Holdout temporal (último 25 % de fechas; pocos días en alerta, ver soporte)

### 3 clases (Normal / Prevención / Crítico)

| Modelo | F1 macro | IC95 % F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| naive_bayes | 0.500 | [0.499, 0.500] | 0.998 | 0.499 | 0.500 |
| persistencia | 0.500 | [0.499, 0.500] | 0.998 | 0.499 | 0.500 |
| clase_mayoritaria | 0.500 | [0.499, 0.500] | 0.999 | 0.499 | 0.500 |

### Tarea operativa: alerta (≥ 38 µg/m³) al día siguiente

| Modelo | F1 alerta | IC95 % | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| naive_bayes | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0 | 1 | 2 |
| persistencia | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0 | 1 | 2 |
| clase_mayoritaria | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0 | 0 | 2 |

McNemar NB vs Persistencia: b=0, c=0, p=1.0000

Matriz de confusión Naive Bayes (filas=real, columnas=predicho; Normal, Prevención, Crítico):

```
[1591, 1, 0]
[2, 0, 0]
[0, 0, 0]
```
Soporte: {'Normal': 1592, 'Prevención': 2, 'Crítico': 0}

## Ablación de atributos (CV por bloques, hiperparámetros por defecto)

| Conjunto | F1 macro (media ± std) | F1 alerta ≥38 | Accuracy |
|---|---|---|---|
| solo_pm25 | 0.638 ± 0.193 | 0.405 | 0.923 |
| pm25+calendario | 0.613 ± 0.199 | 0.383 | 0.953 |
| completo | 0.611 ± 0.205 | 0.358 | 0.945 |

Modelo final — atributos seleccionados: `log_pm25_t, hr_t`; hiperparámetros: `{'priors': 'empirical', 'prior_power': 0.75, 'var_smoothing': 1e-09}`

Nota: los promedios macro se calculan sobre las clases presentes en la prueba (`clases_evaluadas` en metrics.json).