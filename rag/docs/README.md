# Documentos normativos para el RAG

Coloque aquí los PDF oficiales y versiónelos en el repositorio:

| Archivo sugerido | Documento | Origen |
|---|---|---|
| `Resolucion_2254_2017_MinAmbiente.pdf` | Resolución 2254 de 2017 – Norma de calidad del aire ambiente | Ministerio de Ambiente y Desarrollo Sostenible. `python -m rag.ingest` intenta descargarla si falta. |
| `POECA_2026_Plan_de_Accion_Corantioquia.pdf` | Plan de Acción POECA 2026 – zona rural del Valle de Aburrá (Corantioquia, F-PGI-13) | Incluido. |

El nombre del archivo define la fuente que se cita: si contiene `2254` se cita como
"Resolución 2254 de 2017 (MinAmbiente)"; si contiene `corantioquia`, como "Plan de Acción
POECA 2026 – Corantioquia (zona rural Valle de Aburrá)" (ver `SOURCE_NAMES` en `rag/ingest.py`).

Los PDF escaneados (sin texto) se omiten con una advertencia; use la versión digital.
Después de cambiar los PDF ejecute `python -m rag.ingest`.
