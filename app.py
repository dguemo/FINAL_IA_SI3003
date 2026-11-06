"""Dashboard: Agente de calidad del aire del Valle de Aburrá.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from data import db

st.set_page_config(page_title="Calidad del aire · Valle de Aburrá", page_icon="🌫️", layout="wide")

# Paleta: categórica en orden fijo para estaciones; colores de estado (con etiqueta)
# para las clases; secuencial de un solo tono (azul) para la magnitud en el mapa.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLORS = {"Normal": "#0ca30c", "Prevención": "#fab219", "Crítico": "#d03b3b"}
CLASS_ICON = {"Normal": "🟢", "Prevención": "🟠", "Crítico": "🔴"}
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GRID, MUTED = "#e1e0d9", "#898781"


# ----------------------------------------------------------------------------
# Datos (con caché)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_stations() -> pd.DataFrame:
    return db.stations()


@st.cache_data(show_spinner=False)
def load_range():
    return db.date_range()


@st.cache_data(show_spinner=False)
def load_series(codigos: tuple[int, ...], start, end) -> pd.DataFrame:
    return db.series(list(codigos), start, end)


@st.cache_data(show_spinner=False)
def load_summary(start, end) -> pd.DataFrame:
    return db.station_summary(start, end)


@st.cache_data(show_spinner=False)
def load_prediction(codigo: int) -> dict:
    from agent.tools import execute_tool
    return execute_tool("predict_tomorrow", {"estacion": str(codigo)})


def load_metrics() -> dict | None:
    if config.MODEL_METRICS.exists():
        return json.loads(config.MODEL_METRICS.read_text(encoding="utf-8"))
    return None


try:
    stations = load_stations()
    dmin, dmax = load_range()
except FileNotFoundError as err:
    st.error(f"{err}\n\nPrepare los datos con:  `python pipeline.py`")
    st.stop()

label = {int(r.codigo): f"{r.nombre} ({r.codigo})" for r in stations.itertuples()}

# ----------------------------------------------------------------------------
# Barra lateral: parámetros
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Parámetros")
    sel = st.multiselect("Estaciones", options=list(label), format_func=label.get,
                         default=[int(stations["codigo"].iloc[0])], max_selections=len(SERIES),
                         help="La primera estación es la principal (KPI, pronóstico y chat).")
    rango = st.date_input("Rango de fechas", value=(dmin.date(), dmax.date()),
                          min_value=dmin.date(), max_value=dmax.date())
    st.caption(f"Datos disponibles: {dmin.date()} a {dmax.date()}")
    mapa_online = st.toggle("Mapa base con calles (requiere internet)", value=True,
                            help="Desactívelo si no hay conexión: el mapa se dibuja sin teselas.")
    st.divider()
    st.caption("Fuentes: SIATA / Área Metropolitana del Valle de Aburrá (PM2.5), "
               "Open-Meteo (meteorología). Normativa: Res. 2254 de 2017 y POECA.")

if not sel:
    st.info("Seleccione al menos una estación en la barra lateral.")
    st.stop()
if not isinstance(rango, (list, tuple)) or len(rango) != 2:
    st.info("Seleccione fecha inicial y final.")
    st.stop()
start, end = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
principal = sel[0]

# Color estable por estación: el slot se asigna al seleccionarla y no cambia
# aunque se agreguen o quiten otras (el color sigue a la entidad, no a la posición).
slots: dict[int, int] = st.session_state.setdefault("color_slots", {})
for c in list(slots):
    if c not in sel:
        slots.pop(c)
for c in sel:
    if c not in slots:
        slots[c] = min(set(range(len(SERIES))) - set(slots.values()))
color_map = {label[c]: SERIES[slots[c]] for c in sel}

# ----------------------------------------------------------------------------
# Encabezado y KPIs de la estación principal
# ----------------------------------------------------------------------------
st.title("🌫️ Calidad del aire del Valle de Aburrá")
st.caption("PM2.5 promedio 24 h · Clases: Normal 0–37 · Prevención 38–55 · Crítico ≥56 µg/m³")

serie = load_series(tuple(sel), start, end)
sp = serie[serie["codigo"] == principal]
pred = load_prediction(principal)

k1, k2, k3, k4 = st.columns(4)
k1.metric("PM2.5 promedio del rango", f"{sp['pm25'].mean():.1f} µg/m³" if len(sp) else "—",
          help=label[principal])
k2.metric("Máximo diario", f"{sp['pm25'].max():.1f} µg/m³" if len(sp) else "—")
k3.metric("Días ≥ 38 µg/m³", f"{int((sp['clase'] >= 1).sum())} de {len(sp)}" if len(sp) else "—")
if "error" in pred:
    k4.metric("Pronóstico NB", "—", help=pred["error"])
else:
    c = pred["clase_predicha"]
    k4.metric(f"Pronóstico NB {pred['fecha_pronosticada']}",
              f"{CLASS_ICON[c]} {c} ({pred['probabilidades'][c]:.0%})",
              help="Clase más probable según el Naive Bayes del equipo y su probabilidad.")

# ----------------------------------------------------------------------------
# Visualización 1: mapa  ·  Visualización 2: serie de tiempo
# ----------------------------------------------------------------------------
col_map, col_ts = st.columns([2, 3], gap="large")

with col_map:
    st.subheader("Mapa de estaciones")
    summ = load_summary(start, end).dropna(subset=["latitud", "longitud"])
    summ["pm25_prom"] = summ["pm25_prom"].astype(float)
    summ["seleccionada"] = summ["codigo"].isin(sel).map({True: "Sí", False: "No"})
    summ["tam"] = summ["seleccionada"].map({"Sí": 16, "No": 10})
    fig_map = px.scatter_map(
        summ, lat="latitud", lon="longitud", color="pm25_prom", size="tam", size_max=16,
        color_continuous_scale=SEQUENTIAL, range_color=(0, 60),
        hover_name="nombre",
        hover_data={"municipio": True, "pm25_prom": ":.1f", "pm25_max": ":.1f",
                    "dias": True, "dias_sobre_37": True, "tam": False,
                    "latitud": False, "longitud": False, "seleccionada": True},
        labels={"pm25_prom": "PM2.5 prom. (µg/m³)", "pm25_max": "Máx. diario",
                "dias": "Días con dato", "dias_sobre_37": "Días ≥38"},
        zoom=10, center={"lat": summ["latitud"].mean(), "lon": summ["longitud"].mean()},
        map_style="carto-positron" if mapa_online else "white-bg", height=460)
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_colorbar=dict(title="µg/m³", thickness=10))
    st.plotly_chart(fig_map, width="stretch")
    st.caption("Color = PM2.5 promedio en el rango elegido. Marcadores grandes = estaciones seleccionadas.")

with col_ts:
    st.subheader("Serie de tiempo de PM2.5")
    if serie.empty:
        st.warning("No hay datos para las estaciones y el rango elegidos.")
    else:
        serie = serie.assign(
            estacion=serie["codigo"].map(label),
            detalle=[("promedio oficial SIATA" if pd.isna(h) else f"{int(h)} h válidas")
                     for h in serie["horas_validas"]])
        fig = go.Figure()
        top = max(80.0, float(serie["pm25"].max()) * 1.1)
        for y0, y1, name in ((0, 37.5, "Normal"), (37.5, 55.5, "Prevención"), (55.5, top, "Crítico")):
            fig.add_hrect(y0=y0, y1=y1, fillcolor=CLASS_COLORS[name], opacity=0.07, line_width=0,
                          annotation_text=name, annotation_position="top right",
                          annotation_font=dict(color=MUTED, size=11))
        for c in sel:
            s = serie[serie["codigo"] == c]
            if s.empty:
                continue
            # Calendario diario completo: los días sin dato quedan en NaN y la línea se corta
            # (no se dibuja una recta engañosa sobre huecos, p. ej. 2024-08 a 2026-10).
            s = s.set_index("fecha").asfreq("D").reset_index()
            fig.add_trace(go.Scatter(
                x=s["fecha"], y=s["pm25"], mode="lines", name=label[c],
                line=dict(width=2, color=color_map[label[c]]),
                customdata=s[["clase_nombre", "detalle"]],
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} µg/m³ · %{customdata[0]}"
                              "<br>%{customdata[1]}<extra>%{fullData.name}</extra>"))
        fig.add_hline(y=37.5, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_hline(y=55.5, line=dict(color=MUTED, width=1, dash="dot"))
        fig.update_layout(height=460, hovermode="x unified", plot_bgcolor="#fcfcfb",
                          margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(title="PM2.5 24 h (µg/m³)", range=[0, top], gridcolor=GRID),
                          xaxis=dict(gridcolor=GRID),
                          legend=dict(orientation="h", y=-0.15),
                          showlegend=len(sel) >= 2)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Ver tabla de datos"):
            st.dataframe(serie[["estacion", "fecha", "pm25", "clase_nombre", "detalle"]],
                         hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# Pronóstico y desempeño del modelo
# ----------------------------------------------------------------------------
with st.expander("🔮 Detalle del pronóstico y desempeño del modelo Naive Bayes"):
    if "error" in pred:
        st.warning(pred["error"])
    else:
        st.write(f"**{label[principal]}** · último dato {pred['fecha_base_ultimo_dato']} "
                 f"({pred['pm25_ultimo_dia']} µg/m³, {pred['clase_ultimo_dia']}) → "
                 f"pronóstico para **{pred['fecha_pronosticada']}**")
        if "advertencia" in pred:
            st.caption(f"⚠️ {pred['advertencia']}")
        probs = pd.DataFrame({"Clase": list(pred["probabilidades"]),
                              "Probabilidad": list(pred["probabilidades"].values())})
        fig_p = px.bar(probs, x="Probabilidad", y="Clase", orientation="h", text_auto=".0%",
                       color="Clase", color_discrete_map=CLASS_COLORS, height=180)
        fig_p.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                            xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=GRID),
                            plot_bgcolor="#fcfcfb")
        fig_p.update_traces(marker_line_width=0, textfont_color="#0b0b0b")
        st.plotly_chart(fig_p, width="stretch")
    metrics = load_metrics()
    if metrics:
        rows = []
        for sec, name in (("cv", "CV por bloques"), ("holdout", "Holdout temporal")):
            for m, v in metrics[sec]["modelos"].items():
                rows.append({"Evaluación": name, "Modelo": m, "F1 macro": round(v["f1_macro"], 3),
                             "IC95 %": f"[{v['f1_macro_ic95'][0]:.3f}, {v['f1_macro_ic95'][1]:.3f}]",
                             "Accuracy": round(v["accuracy"], 3),
                             "F1 alerta ≥38": round(v["binaria_38"]["f1"], 3),
                             "Recall alerta ≥38": round(v["binaria_38"]["recall"], 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Evaluación principal: CV por bloques temporales. 'Alerta ≥38' = el día "
                   "siguiente alcanza Prevención o Crítico. Reporte completo: "
                   "evaluation/results/model_report.md")

# ----------------------------------------------------------------------------
# Chat con el agente
# ----------------------------------------------------------------------------
st.divider()
st.subheader("💬 Conversa con AireBot")
st.caption("Pregunta por datos medidos, pronósticos o normativa. El agente usa herramientas y cita fuentes.")

if "chat" not in st.session_state:
    st.session_state.chat = []  # [{'role','content','steps'}]


def render_steps(steps: list[dict]) -> None:
    if not steps:
        return
    with st.expander(f"🔧 Herramientas usadas ({len(steps)})"):
        for s in steps:
            icon = "✅" if s["ok"] else "❌"
            st.markdown(f"{icon} **{s['tool']}** (iteración {s['iteration']}) — `{json.dumps(s['args'], ensure_ascii=False)}`")
            st.json(s["result"], expanded=False)


for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_steps(m.get("steps", []))

ejemplos = ["¿Cuál es el pronóstico para mañana en la estación seleccionada?",
            "¿Qué dice la Resolución 2254 sobre el límite de PM2.5 en 24 horas?",
            "¿Qué medidas contempla el POECA en nivel de prevención?"]
cols = st.columns(len(ejemplos))
clicked = next((q for q, c in zip(ejemplos, cols) if c.button(q, width="stretch")), None)
prompt = st.chat_input("Escribe tu pregunta…") or clicked

if prompt:
    from agent.loop import run_agent
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    ctx = (f"estación principal {label[principal]}; estaciones seleccionadas "
           f"{', '.join(label[c] for c in sel)}; rango {start.date()} a {end.date()}")
    with st.chat_message("assistant"):
        with st.spinner("Consultando herramientas…"):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[:-1]]
            res = run_agent(prompt, history=history, ui_context=ctx)
        st.markdown(res.answer)
        render_steps(res.steps)
        if res.stopped_by_limit:
            st.caption(f"Se alcanzó el límite de {config.T_MAX} iteraciones.")
    st.session_state.chat.append({"role": "assistant", "content": res.answer, "steps": res.steps})
