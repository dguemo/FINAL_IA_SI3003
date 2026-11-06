"""Loop agéntico MANUAL de tool calling (sin frameworks).

Algoritmo (T_max = 4 llamadas al LLM):

    para t = 1..T_max:
        si t < T_max: se envían las herramientas (tool_choice="auto")
        si t = T_max: NO se envían herramientas y se pide responder con lo obtenido
                      (algunos modelos ignoran tool_choice="none" y el proveedor
                       devuelve error 400; sin el parámetro tools es imposible llamar una)
        respuesta = LLM(mensajes, herramientas)
        si no hay tool_calls -> respuesta final, terminar
        para cada tool_call:
            observación = execute_tool(...)     # errores -> {"error": ...}
            mensajes += observación (role="tool")

Garantías adicionales:
* Si el LLM falla (red, cuota, clave), se devuelve un mensaje de error, nunca una excepción.
* Si se consultó la normativa y la respuesta no trae citas [Fuente, p. N], se agrega
  automáticamente la lista de fuentes recuperadas (trazabilidad de citas).
* Verificación de citas: toda página citada debe estar entre los fragmentos que
  search_docs devolvió en ESTE turno. Si la normativa no se pudo consultar y aun así
  hay citas, la respuesta se reemplaza (el LLM estaría citando de memoria); si hay
  citas a páginas no recuperadas, se marcan como no verificadas.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import config
from agent.llm_client import LLMClient, LLMError
from agent.prompts import build_system_prompt
from agent.tools import TOOL_SCHEMAS, execute_tool

log = logging.getLogger("agent")

FINAL_NUDGE = ("Límite de pasos alcanzado: responde AHORA al usuario usando solo la información "
               "ya obtenida de las herramientas, sin llamar más herramientas.")
CITATION_RE = re.compile(r"(p\.|pág\.?|página)\s*\d+", re.IGNORECASE)
PAGE_RE = re.compile(r"(?:\bp\.|\bpág\.?|\bpágina)\s*(\d+)", re.IGNORECASE)


@dataclass
class AgentResult:
    answer: str
    steps: list[dict] = field(default_factory=list)   # traza de herramientas
    iterations: int = 0
    stopped_by_limit: bool = False
    error: str | None = None
    unverified_pages: list[int] = field(default_factory=list)

    @property
    def tools_called(self) -> list[str]:
        return [s["tool"] for s in self.steps]

    @property
    def sources(self) -> list[dict]:
        seen, out = set(), []
        for s in self.steps:
            if s["tool"] == "search_docs" and "resultados" in s["result"]:
                for r in s["result"]["resultados"]:
                    key = (r["fuente"], r["pagina"])
                    if key not in seen:
                        seen.add(key)
                        out.append(r)
        return out


def _default_context() -> tuple[str, str]:
    try:
        from data import db
        st = db.stations()
        lo, hi = db.date_range()
        return ("; ".join(f"{r.codigo}={r.nombre_corto} {r.nombre}" + (f" ({r.municipio})" if r.municipio else "")
                          for r in st.itertuples()),
                f"{lo.date()} a {hi.date()}")
    except Exception as err:  # noqa: BLE001
        log.warning("Sin contexto de datos: %s", err)
        return ("(no disponible)", "(no disponible)")


def verify_citations(answer: str, steps: list[dict], sources: list[dict]) -> tuple[str, list[int]]:
    """Comprueba que cada página citada provenga de los fragmentos recuperados."""
    cited = sorted({int(n) for n in PAGE_RE.findall(answer or "")})
    if not cited:
        return answer, []
    if not sources:
        errs = [s["result"].get("error", "") for s in steps if s["tool"] == "search_docs" and not s["ok"]]
        detalle = f" (detalle: {errs[0]})" if errs else ""
        return ("⚠️ No pude consultar la normativa en este momento" + detalle + ". Para no darte "
                "información sin respaldo, no respondo esa parte de memoria. Intenta de nuevo más "
                "tarde o consulta directamente la Resolución 2254 de 2017 y el POECA."), cited
    valid = {int(r["pagina"]) for r in sources}
    bad = [p for p in cited if p not in valid]
    if bad:
        answer = (answer.rstrip() + "\n\n⚠️ Citas no verificadas (no están entre los fragmentos "
                  f"recuperados): {', '.join(f'p. {p}' for p in bad)}.")
    return answer, bad


def _append_sources(answer: str, sources: list[dict]) -> str:
    if not sources or CITATION_RE.search(answer or ""):
        return answer
    lines = ["", "**Fuentes consultadas:**"]
    for r in sources[:4]:
        frag = " ".join(r["texto"].split())[:180]
        lines.append(f'- [{r["fuente"]}, p. {r["pagina"]}]: "{frag}…"')
    return (answer or "").rstrip() + "\n" + "\n".join(lines)


def run_agent(user_message: str, history: list[dict] | None = None,
              client: LLMClient | None = None, ui_context: str | None = None,
              t_max: int = config.T_MAX) -> AgentResult:
    """Ejecuta el agente sobre un mensaje. ``history``: [{'role','content'}] de turnos previos."""
    stations, data_range = _default_context()
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(stations, data_range)}]
    for m in (history or [])[-8:]:          # memoria corta: últimos 4 intercambios
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    content = user_message if not ui_context else f"{user_message}\n\n[Contexto del panel: {ui_context}]"
    messages.append({"role": "user", "content": content})

    result = AgentResult(answer="")
    try:
        client = client or LLMClient()
    except LLMError as err:
        result.answer, result.error = f"⚠️ No se pudo iniciar el LLM: {err}", str(err)
        return result

    for t in range(1, t_max + 1):
        result.iterations = t
        final = t == t_max
        try:
            if final:
                reply = client.chat(messages + [{"role": "system", "content": FINAL_NUDGE}], tools=None)
            else:
                reply = client.chat(messages, tools=TOOL_SCHEMAS, tool_choice="auto")
        except LLMError as err:
            result.answer, result.error = f"⚠️ {err}", str(err)
            return result

        calls = reply.get("tool_calls") or []
        if not calls or final:
            result.answer = reply.get("content") or ""
            result.stopped_by_limit = final and t > 1
            break

        messages.append({"role": "assistant", "content": reply.get("content") or "",
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for c in calls]})
        for c in calls:
            obs = execute_tool(c["name"], c["arguments"])
            try:
                args = json.loads(c["arguments"]) if isinstance(c["arguments"], str) else c["arguments"]
            except json.JSONDecodeError:
                args = {"_raw": c["arguments"]}
            result.steps.append({"iteration": t, "tool": c["name"], "args": args, "result": obs,
                                 "ok": "error" not in obs})
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(obs, ensure_ascii=False, default=str)})

    if not result.answer:
        result.answer = ("No logré completar la respuesta dentro del límite de pasos. "
                         "Intenta una pregunta más específica.")
    result.answer, result.unverified_pages = verify_citations(result.answer, result.steps, result.sources)
    result.answer = _append_sources(result.answer, result.sources)
    return result
