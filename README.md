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

| Fuente | Contenido | Uso |
|---|---|---|
| SIATA `EntregaData1` | PM2.5 horario por estación (histórico 2023-07 → 2024-07) con bandera de calidad | Entrenamiento |
| Geoportal SIATA / AMVA | Catálogo oficial de estaciones (coordenadas, municipio), últimas 72 h y últimos 30 promedios diarios oficiales | Mapa, datos recientes y pronóstico |
| Open-Meteo (ERA5) | Lluvia, viento, humedad, temperatura y altura de la capa límite (diario) | Atributos meteorológicos |
| PDF en `rag/docs/` | Resolución 2254 de 2017 (MinAmbiente); Plan de Acción POECA 2026 (Corantioquia) | RAG |

**Plan B.** Cada descarga se fusiona en `data/raw/` (CSV versionados). Si una fuente falla, el
pipeline sigue con lo que hay en disco, y `build_dataset` solo lee `data/raw/`.

**Limpieza.** Se descartan los valores `-9999`, las banderas de calidad SIATA > 2.5 y los
valores negativos o mayores de 500 µg/m³. Un día solo es válido con al menos 18 horas válidas
(75 % de representatividad). Donde SIATA publica el promedio diario oficial, ese prevalece.

Créditos: datos de **SIATA – Área Metropolitana del Valle de Aburrá** y **Open-Meteo**.
Normativa del **Ministerio de Ambiente y Desarrollo Sostenible** y de **Corantioquia**.

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
