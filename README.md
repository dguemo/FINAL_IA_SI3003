# Agente de calidad del aire del Valle de Aburrá

Proyecto final de **SI3003 – Inteligencia Artificial (EAFIT)**. Es un MVP que integra:

1. **IA clásica propia**: un Gaussian Naive Bayes escrito desde cero con NumPy. Predice la
   clase de PM2.5 del día siguiente (Normal 0–37, Prevención 38–55, Crítico ≥56 µg/m³) y se
   compara contra las líneas base de Persistencia y Clase Mayoritaria.
2. **Agente LLM con RAG**: un loop manual de *tool calling* (máximo T_max = 4 iteraciones) con
   tres herramientas: `get_pm25`, `predict_tomorrow` y `search_docs`. Las búsquedas se hacen
   sobre la Resolución 2254 de 2017 y el Plan de Acción POECA 2026, con citas de fuente y página.
3. **Dashboard Streamlit**: mapa de estaciones, serie de tiempo de PM2.5, pronóstico y chat
   con el agente, todo en una sola página.

```
SIATA / AMVA ─┐                                   ┌─> get_pm25 ────────┐
Open-Meteo ───┼─> data/raw (Plan B) ─> Parquet ─> DuckDB              │
              │                         └─> Naive Bayes ─> predict_tomorrow ─> Agente LLM ─> Streamlit
PDF normativa ┴─> PyMuPDF ─> ChromaDB ──────────────> search_docs ─────┘   (Groq / NIM)
```

**Equipo:** David Guerra Morales, Eder Ceballos Quiroz · Profesor: Jorge Iván Padilla Buriticá ·
Idea elegida: 3.1 *Agente de calidad del aire del Valle de Aburrá*.

---

## Dónde está cada parte solicitada en la guía

### Requisitos obligatorios (sección 2 de la guía)

| # | Requisito | Dónde está en el repositorio |
|---|---|---|
| 1 | **Datos reales** con fuente identificada, documentada y citada, y **plan B** | Fuentes y URLs: sección [Datos](#datos) de este README. Descarga: [`data/download_siata.py`](data/download_siata.py). **Plan B** (históricos descargados, versionados): [`data/raw/`](data/raw/) y [`data/raw/README.md`](data/raw/README.md). Limpieza: [`data/clean.py`](data/clean.py). Consultas DuckDB: [`data/db.py`](data/db.py) |
| 2 | **Modelo de la Parte 1 implementado por el equipo** con métrica frente a línea base | Gaussian Naive Bayes desde cero (solo NumPy): [`models/naive_bayes.py`](models/naive_bayes.py). Líneas base (Persistencia, Clase Mayoritaria): [`models/baselines.py`](models/baselines.py). Atributos: [`models/features.py`](models/features.py). Entrenamiento y evaluación: [`models/train.py`](models/train.py), [`models/evaluate.py`](models/evaluate.py). **Resultados:** [`evaluation/results/model_report.md`](evaluation/results/model_report.md) |
| 3 | **Agente LLM con RAG** y **≥ 2 herramientas** | Loop agéntico manual (T_max = 4, errores como observación, verificador de citas): [`agent/loop.py`](agent/loop.py). **3 herramientas** `get_pm25`, `predict_tomorrow`, `search_docs`: [`agent/tools.py`](agent/tools.py). Prompt: [`agent/prompts.py`](agent/prompts.py). Cliente Groq/NIM: [`agent/llm_client.py`](agent/llm_client.py). RAG (PyMuPDF + embeddings + ChromaDB): [`rag/ingest.py`](rag/ingest.py), [`rag/retriever.py`](rag/retriever.py). Corpus: [`rag/docs/`](rag/docs/) |
| 4 | **≥ 2 visualizaciones** significativas | [`app.py`](app.py): mapa de estaciones (sección "Visualización 1") y serie de tiempo de PM2.5 con bandas por clase ("Visualización 2"); además, barras de probabilidad del pronóstico y tabla de desempeño del modelo |
| 5 | **Interacción**: seleccionar parámetros y conversar con el agente | [`app.py`](app.py): barra lateral (estaciones, rango de fechas, mapa con/sin conexión) y chat con el agente al final de la página. El chat recibe como contexto la estación y las fechas elegidas |
| 6 | **Funcionalidad reproducible** siguiendo el README | [Guía rápida](#guía-rápida-5-pasos) de este README. Orquestador: [`pipeline.py`](pipeline.py) (`--offline` usa el Plan B). Pruebas automáticas: [`tests/`](tests/) |
| 7 | **Repositorio** con código, README, `requirements.txt` y datos o instrucciones | Este README, [`requirements.txt`](requirements.txt), [`.env.example`](.env.example), datos en [`data/raw/`](data/raw/) |
| 8 | **Declaración de uso de IA generativa** | [`docs/declaracion_uso_ia.md`](docs/declaracion_uso_ia.md) |

### Entregables (sección 5 de la guía)

| # | Entregable | Dónde está |
|---|---|---|
| 1 | Repositorio con código, app corriendo desde el README e instrucciones para reproducir resultados | Este repositorio. Reproducir resultados: [Comandos adicionales](#comandos-adicionales) |
| 2 | Informe breve (máx. 4 páginas): problema, datos, método Parte 1, agente y RAG, resultados y limitaciones | [`docs/informe_tecnico.md`](docs/informe_tecnico.md) |
| 3 | Demostración funcional del dashboard | `streamlit run app.py` (paso 5 de la guía rápida) |
| 4 | **Evaluación del agente** con 10–15 preguntas, indicando si llamó la herramienta correcta y si citó la fuente | Banco de **14 preguntas**: [`evaluation/questions.json`](evaluation/questions.json). Script: [`evaluation/eval_agent.py`](evaluation/eval_agent.py). **Resultados por pregunta** (columnas "Herr. OK" y "Cita OK"): [`evaluation/results/agent_eval.md`](evaluation/results/agent_eval.md); detalle con respuestas completas: [`evaluation/results/agent_eval.json`](evaluation/results/agent_eval.json) |
| 5 | Declaración de uso de IA generativa | [`docs/declaracion_uso_ia.md`](docs/declaracion_uso_ia.md) |

### Rúbrica de evaluación (sección 6 de la guía)

| Criterio (peso) | Evidencia en el repositorio |
|---|---|
| Técnica de la Parte 1: implementación propia, corrección y métrica frente a línea base (20 %) | [`models/naive_bayes.py`](models/naive_bayes.py) (derivación matemática en el docstring), pruebas de corrección contra la regla de Bayes calculada a mano en [`tests/test_naive_bayes.py`](tests/test_naive_bayes.py), comparación con líneas base en [`evaluation/results/model_report.md`](evaluation/results/model_report.md) |
| Agente LLM: loop, RAG, ≥ 2 herramientas, manejo de errores y citas (20 %) | [`agent/loop.py`](agent/loop.py), [`agent/tools.py`](agent/tools.py), [`rag/`](rag/); manejo de errores y citas probado en [`tests/test_agent.py`](tests/test_agent.py) |
| Dashboard: visualización clara y significativa (10 %) | [`app.py`](app.py) |
| Interacción del usuario y funcionalidad (15 %) | [`app.py`](app.py) + [Guía rápida](#guía-rápida-5-pasos) |
| Rigor experimental: evaluación del modelo y del agente, limitaciones (15 %) | [`evaluation/results/model_report.md`](evaluation/results/model_report.md) (CV por bloques temporales, holdout, IC bootstrap 95 %, McNemar, ablación), [`evaluation/results/agent_eval.md`](evaluation/results/agent_eval.md), limitaciones en la sección 7 de [`docs/informe_tecnico.md`](docs/informe_tecnico.md) |
| Reproducibilidad: repositorio, README, dependencias, uso responsable de datos (10 %) | Este README, [`requirements.txt`](requirements.txt), [`pipeline.py`](pipeline.py), [Uso responsable de datos y alcance](#uso-responsable-de-datos-y-alcance) |
| Declaración y uso ético de IA generativa (5 %) | [`docs/declaracion_uso_ia.md`](docs/declaracion_uso_ia.md) |

---

## Guía rápida (5 pasos)

> **Requisitos previos**
> - **Python 3.12** (también funciona con 3.11 y 3.13). Compruébelo con `python --version`.
> - **Git**.
> - **Internet durante la instalación**: paquetes de pip y, la primera vez, el modelo de
>   embeddings (~470 MB) desde Hugging Face.
> - **Espacio en disco:** ~2 GB en Windows y macOS; en Linux ~6 GB, porque pip instala PyTorch
>   con CUDA (ver la nota del paso 2 para usar la versión CPU).
> - **Una clave gratuita de Groq**: créela en <https://console.groq.com/keys> → *Create API Key*.
>   Solo la necesita el chat; el mapa, la serie y el pronóstico funcionan sin ella.

### 1. Clonar el repositorio

```bash
git clone https://github.com/dguemo/FINAL_IA_SI3003.git
cd FINAL_IA_SI3003
```

### 2. Crear el entorno virtual e instalar dependencias

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Si PowerShell bloquea la activación, ejecute una vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y repita. En CMD use `.venv\Scripts\activate.bat`.

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Linux sin GPU (opcional, ahorra ~4 GB):** antes de `pip install -r requirements.txt`, instale
> PyTorch solo para CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

La instalación tarda entre 3 y 10 minutos según la conexión. Todos los comandos siguientes se
ejecutan **con el entorno activado**: el prompt muestra `(.venv)`.

### 3. Configurar la clave del LLM

Copie la plantilla:

| Sistema | Comando |
|---|---|
| Windows | `copy .env.example .env` |
| macOS / Linux | `cp .env.example .env` |

Abra el archivo `.env` con cualquier editor y pegue su clave después del `=`:

```
GROQ_API_KEY=gsk_su_clave_aqui
```

Guarde el archivo. `.env` está en `.gitignore` y **nunca se sube** al repositorio.

### 4. Preparar datos, modelo e índice RAG

```bash
python pipeline.py --offline
```

Este comando usa los datos reales ya incluidos en el repositorio (`data/raw/`, el **Plan B**) y:
1. limpia los datos y los exporta a Parquet;
2. entrena y evalúa el Naive Bayes;
3. construye el índice RAG de los PDF normativos. La primera vez descarga el modelo de
   embeddings (~470 MB).

Tarda unos 30 segundos, más la descarga inicial del modelo. Debe terminar con:

```
INFO pipeline: Listo. Ejecute:  streamlit run app.py
```

### 5. Abrir el dashboard

```bash
streamlit run app.py
```

Se abre el navegador en <http://localhost:8501>. Si no se abre solo, copie esa dirección en el
navegador. Para detener la app, pulse `Ctrl + C` en la terminal.

**Uso del dashboard:**
- **Barra lateral:** elija estaciones (la primera es la principal) y el rango de fechas.
- **Arriba:** indicadores, mapa de estaciones (color = PM2.5 promedio) y serie de tiempo con
  las bandas de cada clase.
- **"Detalle del pronóstico…":** probabilidades del Naive Bayes y su desempeño frente a las
  líneas base.
- **Chat (abajo):** escriba una pregunta o pulse un ejemplo. Bajo cada respuesta,
  "Herramientas usadas" muestra qué consultó el agente.

Preguntas de ejemplo:
- *¿Cuál fue el PM2.5 en Tráfico Centro el 2024-03-15?*
- *¿Qué calidad del aire se espera mañana en MED-BEME?*
- *¿Qué dice la Resolución 2254 sobre el límite de PM2.5 en 24 horas?*

Si en la siguiente sesión abre una terminal nueva, solo necesita activar el entorno (paso 2,
línea `activate`) y ejecutar `streamlit run app.py`. El paso 4 no hay que repetirlo.

---

## Comandos adicionales

| Qué | Comando | Resultado |
|---|---|---|
| Pruebas automáticas (no requieren internet ni clave) | `python -m pytest -q` | `34 passed` |
| Banco de evaluación del agente (14 preguntas, requiere clave; ~6 min) | `python -m evaluation.eval_agent` | `evaluation/results/agent_eval.md` |
| Evaluar solo el recuperador RAG (sin clave) | `python -m evaluation.eval_agent --solo-rag` | idem |
| Reentrenar y reevaluar el modelo | `python -m models.train` | `evaluation/results/model_report.md` |
| Reconstruir el índice RAG (tras cambiar PDF) | `python -m rag.ingest` | `rag/chroma_db/` |
| **Actualizar datos desde internet** (SIATA, geoportal AMVA, Open-Meteo) | `python pipeline.py` | descarga y fusiona en `data/raw/`, luego todo lo demás |

La actualización en línea tarda unos 2–3 minutos. Si una fuente falla, el pipeline sigue con
los datos que ya están en `data/raw/` (Plan B).

## Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| `python` no se reconoce / versión distinta | En Windows use `py -3.12`; en macOS/Linux, `python3.12`. Instale Python 3.12 desde python.org. |
| `ModuleNotFoundError` | El entorno virtual no está activado: active `.venv` (paso 2) y repita. |
| El dashboard dice "Falta …parquet. Ejecute: python pipeline.py" | Falta el paso 4: `python pipeline.py --offline`. |
| El chat responde "⚠️ No se pudo iniciar el LLM: Falta la variable GROQ_API_KEY" | El archivo `.env` no existe, no está en la carpeta del proyecto o la clave está vacía (paso 3). Reinicie Streamlit tras editarlo. |
| El chat responde "The model … does not exist" | Groq retiró el modelo. Elija otro de <https://console.groq.com/docs/models> que soporte *tool use* y póngalo en `.env` como `LLM_MODEL=...`. |
| Error 429 / "rate limit" en el chat | Límite del plan gratuito de Groq: espere un minuto y repita. |
| El mapa se ve en blanco | No hay acceso al mapa base. Desactive "Mapa base con calles" en la barra lateral. |
| Falla la descarga del modelo de embeddings en el paso 4 | Revise la conexión o el proxy (se descarga de huggingface.co) y repita `python -m rag.ingest`. |
| `python pipeline.py` (en línea) muestra "Falló Open-Meteo … 429" | Límite diario de Open-Meteo por IP. No afecta: se usan los datos de `data/raw/`. Para importarlos a mano, vea `data/raw/README.md`. |
| `streamlit` dice que el puerto 8501 está ocupado | `streamlit run app.py --server.port 8502`. |

## Datos

| Fuente | Contenido | Uso | Acceso |
|---|---|---|---|
| SIATA `EntregaData1` | PM2.5 horario por estación (histórico 2023-07 → 2024-07) con bandera de calidad | Entrenamiento | `https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json` |
| Geoportal SIATA / AMVA | Catálogo oficial de estaciones (coordenadas, municipio), últimas 72 h y últimos 30 promedios diarios oficiales | Mapa, datos recientes y pronóstico | `https://geoportal.siata.gov.co/fastgeoapi/geodata/` (rutas exactas en [`config.py`](config.py)) |
| Open-Meteo (ERA5) | Lluvia, viento, humedad, temperatura y altura de la capa límite (diario) | Atributos meteorológicos | `https://archive-api.open-meteo.com/v1/archive` y `https://api.open-meteo.com/v1/forecast` |
| PDF en `rag/docs/` | Resolución 2254 de 2017 (MinAmbiente); Plan de Acción POECA 2026 (Corantioquia, F-PGI-13) | RAG | Ver [`rag/docs/README.md`](rag/docs/README.md) |

**Plan B.** Cada descarga se fusiona en `data/raw/` (CSV versionados). Si una fuente falla, el
pipeline sigue con lo que hay en disco, y `build_dataset` solo lee `data/raw/`.

**Limpieza.** Se descartan los valores `-9999`, las banderas de calidad SIATA > 2.5 y los
valores negativos o mayores de 500 µg/m³. Un día solo es válido con al menos 18 horas válidas
(75 % de representatividad). Donde SIATA publica el promedio diario oficial, ese prevalece.

Créditos: datos de **SIATA – Área Metropolitana del Valle de Aburrá** y **Open-Meteo**.
Normativa del **Ministerio de Ambiente y Desarrollo Sostenible** y de **Corantioquia**.

## Uso responsable de datos y alcance

- **Datos públicos y agregados:** mediciones ambientales por estación; no se procesan datos
  personales. Las fuentes se acreditan aquí y en el dashboard.
- **Consultas moderadas:** el descargador llama una vez a cada endpoint por estación, con
  reintentos con espera y un `User-Agent` identificable ([`data/download_siata.py`](data/download_siata.py)).
- **Secretos fuera del repositorio:** la clave del LLM va en `.env`, que está en `.gitignore`.
- **Alcance:** es un proyecto académico. **El pronóstico no es un pronóstico oficial de SIATA
  ni constituye consejo médico, legal o de seguridad.** Para decisiones, consulte los canales
  oficiales de SIATA y del Área Metropolitana del Valle de Aburrá.
- **Privacidad del chat:** las preguntas se envían al proveedor del LLM (Groq o NVIDIA NIM)
  para generar la respuesta.

## Estructura

```
app.py                 Dashboard Streamlit (mapa, serie, pronóstico, chat)
config.py              Rutas, umbrales y parámetros centralizados
pipeline.py            Orquestador reproducible
data/                  download_siata.py · clean.py · build_dataset.py · db.py (DuckDB) · raw/ (Plan B)
models/                naive_bayes.py (NumPy) · baselines.py · features.py · train.py · evaluate.py
rag/                   ingest.py · retriever.py · embeddings.py · docs/ (PDF)
agent/                 llm_client.py · tools.py · prompts.py · loop.py (T_max = 4)
evaluation/            questions.json (14 preguntas + 6 RAG) · eval_agent.py · results/
tests/                 pruebas unitarias e integración con LLM simulado
docs/                  informe_tecnico.md · declaracion_uso_ia.md
```

## Decisiones clave

- **Naive Bayes propio.** Trabaja en log-espacio con *logsumexp*, suavizado de varianza y
  priors con Laplace y aplanamiento λ (por el desbalance de clases). scikit-learn se usa solo
  en `models/evaluate.py` para las métricas.
- **Evaluación sin fuga temporal.** Usa una CV por bloques con purga de 3 días y una CV
  anidada para elegir atributos e hiperparámetros, más un holdout temporal. Reporta IC
  bootstrap 95 % y la prueba de McNemar contra Persistencia. Además del F1 macro de 3 clases,
  evalúa la tarea operativa "alerta ≥ 38 µg/m³", porque Crítico tiene un solo ejemplo en los
  datos abiertos.
- **El LLM no inventa datos.** Todo número sale de una herramienta, y los errores de las
  herramientas vuelven al LLM como observación.
- **Verificador de citas.** Toda página citada debe estar entre los fragmentos recuperados; si
  la normativa no pudo consultarse, la respuesta con citas se bloquea.
- **Iteraciones acotadas.** En la iteración T_max no se envían herramientas, así el agente
  siempre responde dentro de 4 llamadas al LLM.

Resultados: [`evaluation/results/model_report.md`](evaluation/results/model_report.md) y
[`evaluation/results/agent_eval.md`](evaluation/results/agent_eval.md).

## Documentación

- [Informe técnico](docs/informe_tecnico.md)
- [Declaración de uso de IA generativa](docs/declaracion_uso_ia.md)
