"""Pruebas del loop agéntico con un LLM simulado (sin red ni API key)."""
import json

import config
from agent.llm_client import LLMError
from agent.loop import run_agent
from agent.tools import execute_tool


class ScriptedLLM:
    """Devuelve respuestas predefinidas y registra cada llamada."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools is not None})
        if tools is None:
            return {"content": "Respuesta final forzada.", "tool_calls": []}
        return self.replies.pop(0)


def call(name, args, i=1):
    return {"content": None, "tool_calls": [{"id": f"c{i}", "name": name, "arguments": json.dumps(args)}]}


def test_tools_direct(env):
    r = execute_tool("get_pm25", {"estacion": "10", "fecha": "ultima"})
    assert r["clase"] in config.CLASS_NAMES and r["pm25_promedio_24h"] > 0
    p = execute_tool("predict_tomorrow", {"estacion": "Sintetica 1"})
    assert abs(sum(p["probabilidades"].values()) - 1) < 1e-2 and "advertencia" in p
    d = execute_tool("search_docs", {"query": "alerta emergencia prevención"})
    assert d["resultados"][0]["pagina"] == 2 and "Resolución 2254" in d["resultados"][0]["fuente"]


def test_tool_errors_never_raise(env):
    assert "error" in execute_tool("get_pm25", {"estacion": "Marte", "fecha": "2024-01-01"})
    assert "error" in execute_tool("get_pm25", {"estacion": "10", "fecha": "ayer-ish"})
    assert "error" in execute_tool("get_pm25", "{malformado")
    assert "error" in execute_tool("predict_tomorrow", {})
    assert "error" in execute_tool("search_docs", {"query": ""})
    assert "error" in execute_tool("borrar_todo", {})


def test_loop_uses_tool_then_answers(env):
    llm = ScriptedLLM([call("get_pm25", {"estacion": "10", "fecha": "ultima"}),
                       {"content": "El PM2.5 fue X.", "tool_calls": []}])
    res = run_agent("¿PM2.5 reciente en 10?", client=llm)
    assert res.tools_called == ["get_pm25"] and res.steps[0]["ok"]
    assert res.iterations == 2 and res.answer.startswith("El PM2.5")
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "c1"


def test_tool_error_is_returned_as_observation(env):
    llm = ScriptedLLM([call("get_pm25", {"estacion": "Marte", "fecha": "2024-01-01"}),
                       {"content": "Esa estación no existe.", "tool_calls": []}])
    res = run_agent("PM2.5 en Marte", client=llm)
    assert not res.steps[0]["ok"] and res.error is None
    obs = json.loads([m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]["content"])
    assert "error" in obs


def test_t_max_is_enforced(env):
    llm = ScriptedLLM([call("get_pm25", {"estacion": "10", "fecha": "ultima"}, i) for i in range(10)])
    res = run_agent("bucle", client=llm)
    assert len(llm.calls) == config.T_MAX
    assert [c["tools"] for c in llm.calls] == [True] * (config.T_MAX - 1) + [False]
    assert res.stopped_by_limit
    assert len(res.steps) == config.T_MAX - 1 and res.answer


def test_citations_are_appended_when_missing(env):
    llm = ScriptedLLM([call("search_docs", {"query": "nivel máximo PM2.5 24 horas"}),
                       {"content": "El límite es el indicado en la norma.", "tool_calls": []}])
    res = run_agent("límite PM2.5", client=llm)
    assert "Fuentes consultadas" in res.answer and "p. " in res.answer


def test_llm_failure_is_graceful(env):
    class Broken:
        def chat(self, *a, **k):
            raise LLMError("cuota agotada")
    res = run_agent("hola", client=Broken())
    assert res.error and "cuota" in res.answer


def test_ui_context_and_history_reach_llm(env):
    llm = ScriptedLLM([{"content": "ok", "tool_calls": []}])
    run_agent("hola", client=llm, ui_context="estación 10",
              history=[{"role": "user", "content": "antes"}, {"role": "assistant", "content": "r"}])
    msgs = llm.calls[0]["messages"]
    assert msgs[0]["role"] == "system" and "Estacion Sintetica 1" in msgs[0]["content"]
    assert msgs[1]["content"] == "antes" and "estación 10" in msgs[-1]["content"]


def test_evaluator_grounding_and_rag(env):
    from evaluation.eval_agent import eval_rag, grounding, placeholders, summarize
    steps = [{"result": {"pm25_promedio_24h": 41.81, "probabilidades": {"Normal": 0.757}}}]
    assert grounding("Fue 41.8 µg/m³ con 76 % de probabilidad", "q", steps) == 1.0
    assert grounding("Fue 99.9 µg/m³", "q", steps) == 0.0
    assert grounding("Límite de 37 µg/m³ desde 2018", "q", []) is None
    rows = eval_rag([{"id": "R", "consulta": "alerta emergencia prevención", "palabras_clave": ["alerta"]}])
    assert rows[0]["hit@4"] is True
    assert summarize([], rows)["rag"]["hit@4"] == 1.0
    ph = placeholders()
    assert set(ph) == {"estacion", "estacion2", "fecha", "fecha_ini", "fecha_fin"}


def test_invented_citations_are_blocked_when_rag_fails(env, monkeypatch):
    from rag import retriever
    def boom(*a, **k):
        raise FileNotFoundError("índice vacío")
    monkeypatch.setattr(retriever, "search", boom)
    llm = ScriptedLLM([call("search_docs", {"query": "medidas"}),
                       {"content": "Se suspenden quemas (p. 13) y obras (p. 14).", "tool_calls": []}])
    res = run_agent("medidas", client=llm)
    assert "p. 13" not in res.answer and "no respondo" in res.answer and res.unverified_pages == [13, 14]


def test_citation_to_unretrieved_page_is_flagged(env):
    llm = ScriptedLLM([call("search_docs", {"query": "alerta emergencia prevención"}),
                       {"content": "Ver [Res. 2254, p. 2] y [Res. 2254, p. 9].", "tool_calls": []}])
    res = run_agent("niveles", client=llm)
    assert res.unverified_pages == [9] and "no verificadas" in res.answer
