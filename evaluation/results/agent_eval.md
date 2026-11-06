# Evaluación del agente y del RAG (generado por `evaluation/eval_agent.py`)

## Resumen

```json
{
  "agente": {
    "preguntas": 14,
    "tasa_herramientas_correctas": 1.0,
    "tasa_citas_en_normativas": 1.0,
    "anclaje_promedio_cifras": 0.951,
    "tasa_errores_manejados": 1.0,
    "fallos_llm": 0,
    "respuestas_con_citas_no_verificadas": 0,
    "iteraciones_promedio": 2.36,
    "max_iteraciones": 4,
    "latencia_promedio_s": 24.5
  },
  "rag": {
    "consultas": 6,
    "hit@4": 1.0
  }
}
```

## Agente por pregunta

| ID | Categoría | Herramientas llamadas | Herr. OK | Cita OK | Anclaje | Iter. |
|---|---|---|---|---|---|---|
| D1 | datos | get_pm25 | ✅ | — | 1.00 | 2 |
| D2 | datos | get_pm25 | ✅ | — | 1.00 | 2 |
| D3 | datos | get_pm25 | ✅ | — | 1.00 | 2 |
| P1 | pronóstico | predict_tomorrow | ✅ | — | 1.00 | 2 |
| P2 | pronóstico | predict_tomorrow, predict_tomorrow | ✅ | — | 0.71 | 3 |
| N1 | normativa | search_docs | ✅ | ✅ | 1.00 | 2 |
| N2 | normativa | search_docs | ✅ | ✅ | 1.00 | 2 |
| N3 | normativa | search_docs | ✅ | ✅ | 1.00 | 2 |
| N4 | normativa | search_docs, search_docs | ✅ | ✅ | 0.69 | 3 |
| M1 | multi-herramienta | get_pm25, search_docs, search_docs | ✅ | ✅ | 1.00 | 4 |
| M2 | multi-herramienta | predict_tomorrow, search_docs, search_docs | ✅ | ✅ | 1.00 | 4 |
| E1 | errores | get_pm25 | ✅ | — | — | 2 |
| E2 | errores | get_pm25 | ✅ | — | 1.00 | 2 |
| F1 | fuera de alcance | — | ✅ | — | — | 1 |

## Recuperador RAG

| ID | Consulta | hit@4 | Top-1 |
|---|---|---|---|
| R1 | nivel máximo permisible PM2.5 24 horas | ✅ | Resolución 2254 de 2017 (MinAmbiente) p.2 (sim 0.781) |
| R2 | niveles de prevención alerta y emergencia PM2.5 | ✅ | Resolución 2254 de 2017 (MinAmbiente) p.5 (sim 0.74) |
| R3 | índice de calidad del aire colores y rangos | ✅ | Resolución 2254 de 2017 (MinAmbiente) p.8 (sim 0.768) |
| R4 | representatividad temporal 75 por ciento de los datos | ✅ | Resolución 2254 de 2017 (MinAmbiente) p.5 (sim 0.498) |
| R5 | medidas del POECA restricción vehicular episodio | ✅ | Plan de Acción POECA 2026 – Corantioquia (zona rural Valle de Aburrá) p.16 (sim 0.536) |
| R6 | nivel máximo permisible PM2.5 anual | ✅ | Resolución 2254 de 2017 (MinAmbiente) p.2 (sim 0.667) |