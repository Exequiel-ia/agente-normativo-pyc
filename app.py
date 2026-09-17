from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.llm_client import generate_answer
from src.rag_pipeline import build_knowledge_base, search_knowledge_base


ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "documentos_cmf"
LOGO = ROOT / "assets" / "logo_pyc.jpg"

st.set_page_config(
    page_title="Agente Normativo PyC",
    page_icon="📘",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1280px; padding-top: 1.4rem; }
      .pyc-title { color:#183a78; font-weight:800; margin:0; }
      .pyc-subtitle { color:#5d687a; margin-top:.25rem; }
      .status-ok { background:#e9f7ef; border-left:5px solid #238636;
                   padding:.8rem 1rem; border-radius:8px; }
      .source-card { background:white; border:1px solid #dce3ee;
                     padding:.8rem; border-radius:10px; margin:.4rem 0; }
      div[data-testid="stMetric"] { background:white; border:1px solid #dce3ee;
                                    padding:12px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

head_logo, head_text = st.columns([1, 5])
with head_logo:
    if LOGO.exists():
        st.image(str(LOGO), width=180)
with head_text:
    st.markdown('<h1 class="pyc-title">Agente Normativo PyC</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pyc-subtitle">Consulta asistida sobre normativa pública de la CMF</p>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_kb():
    return build_knowledge_base(DOCS_DIR)


with st.sidebar:
    st.header("Base de conocimiento")
    st.caption("Fuentes públicas descargadas desde la CMF")
    top_k = st.slider("Fragmentos a recuperar", 2, 8, 5)
    min_score = st.slider("Umbral mínimo de evidencia", 0.0, 0.5, 0.08, 0.01)
    st.divider()
    st.subheader("Modelo generativo")
    provider = st.selectbox("Proveedor", ["Gemini", "Modo demostración"])
    if provider == "Gemini":
        st.caption("La clave se configura como secreto; nunca se guarda en GitHub.")
        api_key_input = st.text_input("API key (opcional en esta sesión)", type="password")
    else:
        api_key_input = ""
    st.divider()
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "kb" not in st.session_state:
    progress = st.progress(0, text="Preparando documentos...")
    labels = [
        "Leyendo PDF",
        "Extrayendo texto",
        "Dividiendo por páginas y fragmentos",
        "Vectorizando contenido",
        "Construyendo índice",
    ]
    for i, label in enumerate(labels, 1):
        progress.progress(i * 18, text=label)
    st.session_state.kb = load_kb()
    progress.progress(100, text="Base de conocimiento disponible")

kb = st.session_state.kb
metrics = st.columns(4)
metrics[0].metric("Documentos", kb.document_count)
metrics[1].metric("Páginas con texto", kb.page_count)
metrics[2].metric("Fragmentos", len(kb.chunks))
metrics[3].metric("Índice", "Disponible")

st.markdown(
    '<div class="status-ok"><b>Agente preparado.</b> Las respuestas deben apoyarse en los documentos recuperados y mostrar sus fuentes.</div>',
    unsafe_allow_html=True,
)

with st.expander("Ver documentos incorporados"):
    for name, pages in kb.document_pages.items():
        st.write(f"• **{name}** — {pages} páginas con contenido")

st.subheader("Consulta normativa")
st.caption(
    "Ejemplos: ¿En qué moneda deben informarse los montos? · "
    "¿Cuál es la periodicidad del archivo D03? · ¿Qué contiene el archivo MB1?"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Escribe tu pregunta sobre la normativa...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Consultando la base de conocimiento...", expanded=True) as status:
            st.write("1. Vectorizando la pregunta")
            results = search_knowledge_base(kb, question, top_k=top_k)
            st.write("2. Recuperando fragmentos relevantes")
            valid = [r for r in results if r.score >= min_score]
            st.write(f"3. Evidencia encontrada: {len(valid)} fragmentos")

            if not valid:
                answer = (
                    "No encontré evidencia suficiente en la documentación disponible para "
                    "responder con seguridad. Prueba reformulando la pregunta o revisa si la "
                    "norma correspondiente está incorporada."
                )
                status.update(label="Sin evidencia suficiente", state="complete")
            else:
                st.write("4. Preparando contexto para el modelo generativo")
                secret_key = ""
                try:
                    secret_key = st.secrets.get("GEMINI_API_KEY", "")
                except Exception:
                    pass
                api_key = api_key_input or os.getenv("GEMINI_API_KEY", "") or secret_key
                answer, mode = generate_answer(question, valid, provider, api_key)
                st.write(f"5. Respuesta generada en modo: {mode}")
                status.update(label="Consulta completada", state="complete")

        st.markdown(answer)
        if valid:
            st.markdown("#### Fuentes recuperadas")
            for idx, item in enumerate(valid, 1):
                with st.expander(
                    f"{idx}. {item.source} · página {item.page} · relevancia {item.score:.1%}"
                ):
                    st.write(item.text)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption(
    "Prototipo demostrativo. Las respuestas no sustituyen la revisión de la normativa oficial vigente."
)
