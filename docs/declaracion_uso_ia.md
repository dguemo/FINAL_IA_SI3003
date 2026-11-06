# Declaración de uso ético de IA generativa

**Proyecto:** Agente de calidad del aire del Valle de Aburrá (SI3003, EAFIT)
**Equipo:** _[nombres y correos]_ · **Fecha:** _[fecha]_

## 1. Herramientas de IA generativa usadas durante el desarrollo

| Herramienta | Para qué se usó | Qué hizo el equipo |
|---|---|---|
| Claude (Anthropic) | Propuesta de arquitectura modular; borradores de código de los módulos (datos, Naive Bayes en NumPy, RAG, loop agéntico, dashboard), pruebas automáticas y borradores de documentación. | Definió los requisitos y restricciones; revisó, ejecutó y corrigió el código; verificó la matemática del Naive Bayes con pruebas contra cálculos a mano; validó resultados con datos reales; redactó el análisis e interpretación. |
| _[otra herramienta, si aplica]_ | _[uso]_ | _[revisión]_ |

## 2. IA generativa como componente del sistema

- **LLM del agente:** `openai/gpt-oss-120b` (vía Groq) o el modelo configurado en NVIDIA NIM. Solo
  redacta respuestas y decide qué herramienta llamar. Las cifras y pronósticos salen siempre de
  las herramientas (datos SIATA y modelo propio), y la normativa sale de los PDF oficiales con
  cita de página.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers) para la
  búsqueda semántica.

## 3. Lo que NO se delegó a la IA

- El algoritmo de IA clásica (Gaussian Naive Bayes) está implementado por el equipo en NumPy,
  sin librerías de aprendizaje automático. scikit-learn se usa solo para métricas.
- La interpretación de resultados, las conclusiones y las limitaciones son responsabilidad del
  equipo.

## 4. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Alucinación de cifras o normas | Prompt restrictivo, datos solo vía herramientas, citas [Fuente, p. N], verificador automático de citas (bloquea páginas no recuperadas), métrica de anclaje en el banco de evaluación. |
| Uso del pronóstico como alerta oficial | El dashboard y el agente indican que es un modelo académico con incertidumbre; la fuente oficial es SIATA/AMVA. |
| Privacidad | No se procesan datos personales; las preguntas se envían al proveedor del LLM y no se almacenan en el repositorio. |
| Licencia de datos | Se acredita a SIATA/AMVA y Open-Meteo; los datos se usan con fines académicos. |

Declaramos que el contenido entregado fue revisado y comprendido por todos los integrantes,
quienes asumen la responsabilidad por su exactitud.
