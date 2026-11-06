# Informe técnico — Agente de calidad del aire del Valle de Aburrá

**Curso:** SI3003 Inteligencia Artificial, Universidad EAFIT · **Equipo:** _[nombres]_ · **Repositorio:** github.com/dguemo/FINAL_IA_SI3003

## 1. Problema

El Valle de Aburrá es una cuenca estrecha: la baja altura de la capa de mezcla atrapa el
material particulado, sobre todo en las temporadas secas (febrero–abril y octubre–noviembre).
Objetivos: (i) **pronosticar la clase de PM2.5 del día siguiente** por estación con un
algoritmo propio y (ii) ofrecer un **agente conversacional** que combine datos medidos,
pronóstico y normativa citada. Clases (µg/m³, 24 h): Normal 0–37, Prevención 38–55 y
Crítico ≥ 56, coherentes con la Res. 2254/2017 (límite de 24 h y niveles de prevención/alerta).
El valor se redondea al entero (half-up) antes de clasificar.

## 2. Datos

| Fuente | Contenido |
|---|---|
| SIATA EntregaData1 | PM2.5 horario de 21 estaciones, 31-jul-2023 a 31-jul-2024 (186 141 horas), con bandera de calidad |
| Geoportal SIATA/AMVA | Catálogo oficial (coordenadas, municipio), últimas 72 h y 30 promedios diarios oficiales (oct–nov 2026) |
| Open-Meteo / ERA5 | Temperatura, humedad, lluvia, viento y altura de capa límite (diario, centro del valle) |

**Limpieza:** `-9999`, bandera SIATA > 2.5, valores < 0 o > 500 → faltante (75.1 % de horas
válidas); un día es válido con ≥ 18 h válidas (75 %). Donde existe el promedio diario oficial
de SIATA, prevalece sobre el calculado. **Resultado:** 6 544 días-estación en 24 estaciones;
6 063 filas con atributos completos. **Desbalance extremo:** objetivo Normal 5 895, Prevención
167, **Crítico 1**. Los días ≥ 38 se concentran en ene–abr 2024. La capa límite de ERA5 falta
en ene–jun 2024 (cobertura 61 %), por lo que se excluye automáticamente; las demás variables
meteorológicas se conservan. **Plan B:** `data/raw/` (CSV versionados) permite correr todo sin
internet.

## 3. Método clásico: Gaussian Naive Bayes desde cero (NumPy)

  log P(k|**x**) = log π_k − ½ Σ_j [ log(2πσ²_kj) + (x_j − μ_kj)²/σ²_kj ] − logsumexp_k(·)

- μ y σ² por clase, estimados por máxima verosimilitud; σ² + ε, con ε = 10⁻⁹·max Var(x_j).
- Priors π_k ∝ (n_k + 1)^λ. El exponente λ ∈ [0, 1] **aplana los priors** ante el desbalance
  (λ = 1 empírico, λ = 0 uniforme). Equivale a una regla de decisión de Bayes sensible al costo.
- Implementación vectorizada en `models/naive_bayes.py`, verificada con pruebas contra la regla
  de Bayes calculada a mano.

**Atributos candidatos** (día t → clase en t+1, sin información futura):
- log1p de PM2.5 en t, t−1 y t−2; media de 3 días; Δ diario.
- Armónicos anual y **semestral**, que capturan las dos temporadas críticas.
- Lluvia, viento y humedad del día t.

**Selección:** NB supone independencia condicional y los rezagos son redundantes. Por eso se
aplica **selección hacia adelante** de atributos y se elige λ con CV. El criterio es el **F1
macro agregado fuera de pliegue**: promediar el F1 por pliegue premiaba "nunca alertar" en los
pliegues sin días de alerta (Forman & Scholz, 2010).

**Modelo final:** atributos `log_pm25_t` y `hr_t` (humedad), con λ = 0.75. **Líneas base:**
Persistencia (mañana = hoy) y Clase Mayoritaria.

## 4. Evaluación del modelo

Protocolo sin fuga temporal:
- **CV por bloques** (5 bloques contiguos, purga de 3 días; Bergmeir & Benítez, 2012), con
  selección **anidada** de atributos y λ solo en la parte de entrenamiento de cada bloque.
  Es la evaluación principal, porque solo así los meses con episodios quedan en prueba.
- **Holdout temporal** (último 25 % de fechas, desde 2024-05-20), que se reporta como control.

Métricas:
- **3 clases:** F1 macro sobre las clases presentes, accuracy, precision y recall macro.
- **Tarea operativa "alerta":** ¿mañana ≥ 38 µg/m³? Se evalúa así porque Crítico tiene un solo
  ejemplo y no puede aprenderse.
- **Pruebas estadísticas:** IC bootstrap 95 % y McNemar exacto NB vs Persistencia.

**CV por bloques, fuera de pliegue (n = 6 063; 168 días de alerta):**

| Modelo | F1 macro [IC95] | Accuracy | F1 alerta [IC95] | Precision alerta | Recall alerta |
|---|---|---|---|---|---|
| Naive Bayes | 0.465 [0.446, 0.483] | 0.951 | 0.424 [0.368, 0.476] | 0.315 | **0.649** |
| Persistencia | **0.522** [0.500, 0.541] | 0.977 | **0.585** [0.523, 0.646] | 0.587 | 0.583 |
| Clase mayoritaria | 0.329 | 0.972 | 0.000 | — | 0.000 |

- **McNemar NB vs Persistencia:** p < 10⁻²¹. La diferencia es significativa a favor de
  Persistencia.
- **Holdout:** solo trae 2 días de alerta en 1 594 casos; los tres modelos empatan (F1 macro
  0.50) y no discrimina.
- **Ablación** (F1 macro media por pliegue / F1 alerta): solo PM2.5 0.638 / 0.405;
  + calendario 0.613 / 0.383; + meteorología 0.611 / 0.358.

**Interpretación.** Ambos modelos superan claramente a la clase mayoritaria, que tiene 97 % de
accuracy pero no detecta ninguna alerta: la accuracy engaña con este desbalance. Persistencia
es una línea base muy fuerte porque el PM2.5 diario está muy autocorrelacionado. NB **detecta
más días de alerta** (109 contra 98 aciertos; recall 0.65 contra 0.58) a costa de más falsas
alarmas (237 contra 69), así que su F1 es menor. Para una alerta de salud pública, ese balance
entre recall y precisión puede ser aceptable, pero con los datos disponibles NB no supera a
Persistencia en F1. La ablación muestra que el calendario y la meteorología de un solo punto
no aportan con un solo año de historia.

## 5. Agente LLM + RAG

- **RAG:**
  - Documentos: Res. 2254/2017 (11 págs.) y Plan de Acción POECA 2026 de Corantioquia (zona
    rural del valle, 22 págs.).
  - Extracción por página con PyMuPDF, sin los encabezados y pies que se repiten en cada página.
  - Fragmentos de ~900 caracteres con 150 de solapamiento, sin cruzar páginas (95 en total).
  - Embeddings `paraphrase-multilingual-MiniLM-L12-v2` y ChromaDB persistente con distancia
    coseno.
- **Herramientas:**
  - `get_pm25(estacion, fecha[, fecha_fin])`: SQL parametrizado en DuckDB sobre Parquet.
  - `predict_tomorrow(estacion)`: NB propio; devuelve probabilidades y advierte si el dato es
    antiguo.
  - `search_docs(query)`: top-k fragmentos con fuente, página y similitud.
  - Las estaciones se reconocen por código, código corto (MED-BEME) o nombre.
- **Loop manual** (`agent/loop.py`, T_max = 4):
  - Iteraciones 1–3 con herramientas.
  - En la 4 **no se envían herramientas** y se pide responder. Usar `tool_choice="none"` falló
    con `gpt-oss-120b`, que llamó una herramienta igual y provocó error 400.
  - Los errores de las herramientas vuelven como observación `{"error": …}`.
  - Los fallos del LLM se reintentan con *backoff* y nunca tumban la app.
- **Verificador de citas:** cada página citada debe estar entre los fragmentos recuperados en
  ese turno. Si la normativa no pudo consultarse y hay citas, la respuesta se bloquea; las
  páginas no recuperadas se marcan. Se añadió tras observar, en pruebas reales, que con el
  índice vacío el LLM inventaba páginas.
- **LLM:** `openai/gpt-oss-120b` vía Groq (configurable a NVIDIA NIM).

## 6. Evaluación del agente

Banco de **14 preguntas** (`evaluation/questions.json`) en 6 categorías: datos, pronóstico,
normativa, multi-herramienta, errores y fuera de alcance. Además, 6 consultas solo para el
recuperador. Las estaciones y fechas se instancian con datos reales.

| Métrica | Resultado |
|---|---|
| Herramientas esperadas invocadas | **14/14 (1.00)** |
| Citas [Fuente, p. N] en preguntas normativas | **6/6 (1.00)** |
| Errores manejados (estación inexistente, fecha fuera de rango) | **2/2 (1.00)** |
| Anclaje de cifras (cifras de la respuesta presentes en salidas de herramientas/contexto) | 0.95 |
| Respuestas con citas no verificadas / fallos del LLM | 0 / 0 |
| Iteraciones promedio / máximo | 2.36 / 4 |
| RAG hit@4 | **6/6 (1.00)** |
| Latencia promedio | 24.5 s (plan gratuito de Groq) |

El banco detectó y permitió corregir tres fallas reales: el error 400 en la iteración final,
códigos cortos de estación que el LLM descartaba y citas con la palabra "Fuente" en lugar del
documento. Hay además **34 pruebas automáticas** (`pytest`, con un LLM simulado) que verifican
la matemática del NB, el límite T_max, los errores como observación, la ausencia de fuga en
los atributos, el verificador de citas y la regla del 75 %.

## 7. Limitaciones

- **Historia abierta corta.** Hay un solo año (2023–2024) y fue limpio: Crítico tiene 1
  ejemplo, así que el modelo no puede aprenderlo, y el holdout final casi no tiene alertas.
  Más años (p. ej. el archivo de OpenAQ, o datos solicitados formalmente a SIATA) son la
  mejora prioritaria.
- **NB.** El supuesto naive persiste aunque la selección por CV lo reduce. Las probabilidades
  están mal calibradas (tiende a 0/1). No supera a Persistencia en F1.
- **Meteorología** de un solo punto, del día t (no el pronóstico de t+1), y sin capa límite
  en ene–jun 2024.
- **Agente.** Depende de un LLM externo con latencia y cuotas. El anclaje es heurístico (no
  reconoce cifras derivadas, como diferencias). El POECA indexado es el plan rural de
  Corantioquia, no el POECA urbano del AMVA. No hay OCR para PDF escaneados.

## 8. Referencias

- MinAmbiente (2017). *Resolución 2254 de 2017*.
- Corantioquia (2026). *Plan de Acción POECA – zona rural del Valle de Aburrá* (F-PGI-13).
- SIATA, datos de la red de calidad del aire.
- Bergmeir, C. & Benítez, J. M. (2012). On the use of cross-validation for time series
  predictor evaluation. *Information Sciences*, 191.
- Forman, G. & Scholz, M. (2010). Apples-to-apples in cross-validation studies. *SIGKDD
  Explorations*, 12(1).
- Open-Meteo / ERA5 (Copernicus).
