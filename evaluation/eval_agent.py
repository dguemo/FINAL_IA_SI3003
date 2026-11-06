"""Banco de evaluación del agente y del recuperador RAG.

Métricas por pregunta del agente:
  * herramientas_ok : se invocaron todas las herramientas esperadas (o ninguna si no aplica)
  * cita_ok         : hay cita [Fuente, p. N] cuando la pregunta es normativa
  * anclaje         : % de cifras de la respuesta que aparecen en las salidas de las
                      herramientas (detector simple de cifras inventadas)
  * error_manejado  : en preguntas con fallo esperado, la herramienta devolvió error,
                      el loop no se cayó y la respuesta no inventa un PM2.5
  * iteraciones     : <= T_max
Métrica RAG (no requiere LLM): hit@k = algún fragmento top-k contiene la palabra clave.

Uso:  python -m evaluation.eval_agent            (agente + RAG; requiere API key)
      python -m evaluation.eval_agent --solo-rag (solo recuperador)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time

import pandas as pd

import config

log = logging.getLogger("eval")

NUM_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w])")
# Cifras que el agente puede decir sin herramienta: umbrales del prompt, año/número de norma, etc.
ALLOWED_NUMBERS = {"0", "1", "2", "3", "4", "24", "37", "38", "55", "56", "2.5", "2254", "2017", "100"}


def placeholders() -> dict:
    from data import db
    st = db.stations()
    d = pd.read_parquet(config.PM25_DAILY_PARQUET)
    counts = d.groupby("codigo").size().sort_values(ascending=False)
    c1, c2 = int(counts.index[0]), int(counts.index[min(1, len(counts) - 1)])
    s1 = d[d["codigo"] == c1].sort_values("fecha")
    alto = s1[s1["clase"] >= 1]
    fecha = (alto if len(alto) else s1).iloc[len(alto if len(alto) else s1) // 2]["fecha"]
    fin = s1["fecha"].max()
    name = lambda c: st.loc[st["codigo"] == c, "nombre"].iloc[0]  # noqa: E731
    return {"estacion": name(c1), "estacion2": name(c2), "fecha": str(pd.Timestamp(fecha).date()),
            "fecha_ini": str((fin - pd.Timedelta(days=30)).date()), "fecha_fin": str(fin.date())}


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUM_RE.findall(text or "")}


def grounding(answer: str, question: str, steps: list[dict], context: str = "") -> float | None:
    """Fracción de cifras de la respuesta presentes en la pregunta, en el contexto dado al
    agente (catálogo de estaciones) o en las salidas de herramientas."""
    tool_text = json.dumps([s["result"] for s in steps], ensure_ascii=False)
    raw = numbers_in(tool_text)
    tool_nums = set(raw)
    tool_nums |= {str(round(float(n))) for n in raw}                # 41.81 -> "42"
    tool_nums |= {f"{float(n):.1f}" for n in raw}                   # 41.81 -> "41.8"
    for n in raw:                                                    # 0.987 -> "99", "98.7"
        if float(n) <= 1:
            tool_nums |= {str(round(float(n) * 100)), f"{float(n) * 100:.1f}"}
    nums = numbers_in(answer) - numbers_in(question) - numbers_in(context) - ALLOWED_NUMBERS
    nums = {n for n in nums if not re.fullmatch(r"(19|20)\d\d", n)}  # años
    if not nums:
        return None
    return sum(n in tool_nums for n in nums) / len(nums)


def eval_agent(questions: list[dict], ph: dict) -> list[dict]:
    from agent.llm_client import LLMClient
    from agent.loop import CITATION_RE, _default_context, run_agent
    client = LLMClient()
    context = _default_context()[0]
    rows = []
    for q in questions:
        text = q["pregunta"].format(**ph)
        t0 = time.time()
        res = run_agent(text, client=client)
        dt = time.time() - t0
        called = res.tools_called
        if q.get("sin_herramientas"):
            tools_ok = len(called) == 0
        else:
            tools_ok = set(q.get("herramientas", [])) <= set(called)
        row = {
            "id": q["id"], "categoria": q["categoria"], "pregunta": text,
            "herramientas_llamadas": called, "herramientas_ok": tools_ok,
            "cita_ok": bool(CITATION_RE.search(res.answer)) if q.get("cita") else None,
            "anclaje": grounding(res.answer, text, res.steps, context),
            "citas_no_verificadas": res.unverified_pages,
            "iteraciones": res.iterations, "limite_alcanzado": res.stopped_by_limit,
            "error_llm": res.error, "latencia_s": round(dt, 1), "respuesta": res.answer,
        }
        if q.get("debe_reportar_error"):
            tool_failed = any(not s["ok"] for s in res.steps)
            row["error_manejado"] = tool_failed and res.error is None and row["anclaje"] in (None, 1.0)
        rows.append(row)
        log.info("%s tools=%s ok=%s cita=%s anclaje=%s it=%d %.1fs", q["id"], called, tools_ok,
                 row["cita_ok"], row["anclaje"], res.iterations, dt)
        time.sleep(1.5)  # respeta límites de tasa del proveedor gratuito
    return rows


def eval_rag(items: list[dict], k: int = config.RAG_TOP_K) -> list[dict]:
    from rag.retriever import search
    rows = []
    for it in items:
        res = search(it["consulta"], k)
        text = " ".join(r["texto"].lower() for r in res)
        hit = any(kw.lower() in text for kw in it["palabras_clave"])
        rows.append({"id": it["id"], "consulta": it["consulta"], f"hit@{k}": hit,
                     "top1": f"{res[0]['fuente']} p.{res[0]['pagina']} (sim {res[0]['similitud']})"})
    return rows


def summarize(agent_rows: list[dict], rag_rows: list[dict]) -> dict:
    s: dict = {}
    if agent_rows:
        df = pd.DataFrame(agent_rows)
        cita = df["cita_ok"].dropna()
        anc = df["anclaje"].dropna()
        err = df["error_manejado"].dropna() if "error_manejado" in df else pd.Series(dtype=bool)
        s["agente"] = {
            "preguntas": len(df),
            "tasa_herramientas_correctas": round(df["herramientas_ok"].mean(), 3),
            "tasa_citas_en_normativas": round(cita.mean(), 3) if len(cita) else None,
            "anclaje_promedio_cifras": round(anc.mean(), 3) if len(anc) else None,
            "tasa_errores_manejados": round(err.astype(bool).mean(), 3) if len(err) else None,
            "fallos_llm": int(df["error_llm"].notna().sum()),
            "respuestas_con_citas_no_verificadas": int(df["citas_no_verificadas"].map(bool).sum()),
            "iteraciones_promedio": round(df["iteraciones"].mean(), 2),
            "max_iteraciones": int(df["iteraciones"].max()),
            "latencia_promedio_s": round(df["latencia_s"].mean(), 1),
        }
    if rag_rows:
        key = next(k for k in rag_rows[0] if k.startswith("hit@"))
        s["rag"] = {"consultas": len(rag_rows), key: round(sum(r[key] for r in rag_rows) / len(rag_rows), 3)}
    return s


def write_outputs(agent_rows, rag_rows, summary) -> None:
    out = config.EVAL_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_eval.json").write_text(json.dumps(
        {"resumen": summary, "agente": agent_rows, "rag": rag_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    lines = ["# Evaluación del agente y del RAG (generado por `evaluation/eval_agent.py`)", "",
             "## Resumen", "", "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```", ""]
    if agent_rows:
        lines += ["## Agente por pregunta", "",
                  "| ID | Categoría | Herramientas llamadas | Herr. OK | Cita OK | Anclaje | Iter. |",
                  "|---|---|---|---|---|---|---|"]
        for r in agent_rows:
            fmt = lambda v: "—" if v is None else ("✅" if v is True else "❌" if v is False else f"{v:.2f}")  # noqa: E731
            lines.append(f"| {r['id']} | {r['categoria']} | {', '.join(r['herramientas_llamadas']) or '—'} | "
                         f"{fmt(r['herramientas_ok'])} | {fmt(r['cita_ok'])} | {fmt(r['anclaje'])} | {r['iteraciones']} |")
    if rag_rows:
        key = next(k for k in rag_rows[0] if k.startswith("hit@"))
        lines += ["", "## Recuperador RAG", "", f"| ID | Consulta | {key} | Top-1 |", "|---|---|---|---|"]
        for r in rag_rows:
            lines.append(f"| {r['id']} | {r['consulta']} | {'✅' if r[key] else '❌'} | {r['top1']} |")
    (out / "agent_eval.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-rag", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    bank = json.loads((config.EVAL_DIR / "questions.json").read_text(encoding="utf-8"))

    rag_rows = eval_rag(bank["rag"])
    agent_rows = []
    if not args.solo_rag:
        from agent.llm_client import LLMError
        try:
            agent_rows = eval_agent(bank["agente"], placeholders())
        except LLMError as err:
            log.error("No se pudo evaluar el agente: %s (use --solo-rag)", err)
            return 1
    summary = summarize(agent_rows, rag_rows)
    write_outputs(agent_rows, rag_rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
