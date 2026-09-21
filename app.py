from __future__ import annotations

import io
import json
import os
import re
import tempfile
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st

try:
    from src.ingestion_assistant import render_ingestion_assistant
except ModuleNotFoundError:
    from ingestion_assistant import render_ingestion_assistant

try:
    from src.llm_client import generate_answer
    from src.rag_pipeline import (
        build_knowledge_base,
        file_sha256,
        inspect_pdf,
        load_metadata,
        save_metadata,
        search_knowledge_base,
    )
except ModuleNotFoundError:  # Compatibilidad con el repositorio desplegado en formato plano.
    from llm_client import generate_answer
    from rag_pipeline import (
        build_knowledge_base,
        file_sha256,
        inspect_pdf,
        load_metadata,
        save_metadata,
        search_knowledge_base,
    )


ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "documentos_cmf" if (ROOT / "documentos_cmf").exists() else ROOT
LOGO = ROOT / "assets" / "logo_pyc.jpg"
if not LOGO.exists():
    LOGO = ROOT / "logo_pyc.jpg"
METADATA_PATH = ROOT / "knowledge_metadata.json"

st.set_page_config(page_title="Agente Normativo PyC", page_icon="📘", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1280px; padding-top: 1.4rem; }
      .pyc-title { color:#183a78; font-weight:800; margin:0; }
      .pyc-subtitle { color:#5d687a; margin-top:.25rem; }
      .status-ok { background:#e9f7ef; border-left:5px solid #238636;
                   padding:.8rem 1rem; border-radius:8px; }
      .status-info { background:#eef5ff; border-left:5px solid #2878d0;
                     padding:.8rem 1rem; border-radius:8px; }
      div[data-testid="stMetric"] { background:white; border:1px solid #dce3ee;
                                    padding:12px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def safe_pdf_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ._ -]", "_", Path(name).name)
    if not clean.lower().endswith(".pdf"):
        clean += ".pdf"
    return clean


@st.cache_resource(show_spinner=False)
def load_kb(max_chars: int = 1600, overlap: int = 300):
    return build_knowledge_base(DOCS_DIR, METADATA_PATH, max_chars=max_chars, overlap=overlap)


def package_knowledge_base() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for pdf in sorted(DOCS_DIR.glob("*.pdf")):
            archive.write(pdf, f"documentos_cmf/{pdf.name}")
        if METADATA_PATH.exists():
            archive.write(METADATA_PATH, "knowledge_metadata.json")
        governance_path = ROOT / "knowledge_governance.json"
        if governance_path.exists():
            archive.write(governance_path, "knowledge_governance.json")
        archive.writestr(
            "INSTRUCCIONES.txt",
            "Suba los PDF a documentos_cmf/ y knowledge_metadata.json a la raíz del repositorio.\n"
            "Luego Streamlit reconstruirá automáticamente la base de conocimiento.\n",
        )
    return buffer.getvalue()


head_logo, head_text = st.columns([1, 5])
with head_logo:
    if LOGO.exists():
        st.image(str(LOGO), width=180)
with head_text:
    st.markdown('<h1 class="pyc-title">Agente Normativo PyC</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pyc-subtitle">Consulta e ingesta asistida de normativa pública de la CMF</p>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("Configuración de consulta")
    top_k = st.slider("Fragmentos finales", 3, 10, 6)
    candidate_k = st.slider("Candidatos a revisar", 10, 30, 20)
    min_score = st.slider("Umbral mínimo de evidencia", 0.0, 0.5, 0.06, 0.01)
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
    started = time.perf_counter()
    with st.spinner("Preparando la base de conocimiento..."):
        st.session_state.kb = load_kb()
    st.session_state.kb_load_seconds = time.perf_counter() - started

kb = st.session_state.kb
query_tab, knowledge_tab, admin_tab, monitor_tab = st.tabs(
    ["💬 Consulta normativa", "📚 Base de conocimiento", "⚙️ Administración", "📊 Monitoreo"]
)


with query_tab:
    metrics = st.columns(4)
    metrics[0].metric("Documentos", kb.document_count)
    metrics[1].metric("Páginas con texto", kb.page_count)
    metrics[2].metric("Fragmentos", len(kb.chunks))
    metrics[3].metric("Carga inicial", f"{st.session_state.get('kb_load_seconds', 0):.1f} s")
    st.markdown(
        '<div class="status-ok"><b>Agente preparado.</b> La búsqueda combina texto, caracteres y prioridad para códigos normativos exactos.</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Consulta normativa")
    st.caption(
        "Ejemplos: ¿Qué contiene el archivo D10? · ¿Cuál es la periodicidad del D03? · "
        "¿En qué moneda deben informarse los montos?"
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
                search_started = time.perf_counter()
                candidates = search_knowledge_base(kb, question, top_k=candidate_k)
                valid = [result for result in candidates if result.score >= min_score][:top_k]
                search_seconds = time.perf_counter() - search_started
                st.write(f"Evidencia seleccionada: {len(valid)} fragmentos")
                if not valid:
                    answer = (
                        "No encontré evidencia suficiente en la documentación disponible para "
                        "responder con seguridad. Prueba reformulando la pregunta o confirma que "
                        "el documento correspondiente esté incorporado."
                    )
                    mode = "sin evidencia"
                else:
                    api_key = api_key_input or os.getenv("GEMINI_API_KEY", "") or get_secret("GEMINI_API_KEY")
                    llm_started = time.perf_counter()
                    answer, mode = generate_answer(question, valid, provider, api_key)
                    st.session_state.last_llm_seconds = time.perf_counter() - llm_started
                st.session_state.last_search_seconds = search_seconds
                st.session_state.last_question = question
                st.session_state.last_mode = mode
                status.update(label="Consulta completada", state="complete")

            st.markdown(answer)
            if valid:
                st.markdown("#### Fuentes recuperadas")
                for index, item in enumerate(valid, 1):
                    code_text = f" · códigos {', '.join(item.codes[:5])}" if item.codes else ""
                    with st.expander(
                        f"{index}. {item.source} · página {item.page} · puntaje {item.score:.3f}{code_text}"
                    ):
                        st.write(item.text)
        st.session_state.messages.append({"role": "assistant", "content": answer})


with knowledge_tab:
    st.subheader("Documentos incorporados")
    st.caption("Inventario actual utilizado por el agente.")
    metadata = load_metadata(METADATA_PATH)
    rows = []
    for name, pages in kb.document_pages.items():
        info = metadata.get(name, {})
        rows.append(
            {
                "Documento": name,
                "Páginas": pages,
                "Área": info.get("area", "Sin clasificar"),
                "Versión": info.get("version", "—"),
                "Vigencia": info.get("fecha_vigencia", "—"),
                "Estado": info.get("estado", "Vigente"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Probar recuperación sin utilizar Gemini")
    test_question = st.text_input("Pregunta o código de prueba", value="¿Qué contiene el archivo D10?")
    if st.button("Probar búsqueda", type="primary"):
        results = search_knowledge_base(kb, test_question, top_k=10)
        for index, result in enumerate(results, 1):
            st.write(
                f"**{index}. {result.source}, página {result.page}** — puntaje {result.score:.3f}"
            )
            st.caption(result.text[:420] + ("…" if len(result.text) > 420 else ""))


with admin_tab:
    st.subheader("Administración de conocimiento")
    st.caption("Gestiona el ciclo completo: borrador, análisis, curación, pruebas y publicación.")
    admin_secret = os.getenv("ADMIN_PASSWORD", "") or get_secret("ADMIN_PASSWORD")
    if not admin_secret:
        st.warning(
            "La administración está deshabilitada hasta configurar `ADMIN_PASSWORD` en los "
            "Secrets de Streamlit. Esta clave debe ser distinta de GEMINI_API_KEY."
        )
        st.code('ADMIN_PASSWORD = "una-clave-administrativa-segura"', language="toml")
    else:
        password = st.text_input("Clave de administración", type="password", key="admin_password")
        if password != admin_secret:
            st.info("Ingresa la clave administrativa para habilitar la carga y publicación.")
        else:
            st.success("Acceso administrativo habilitado.")
            def refresh_after_publish() -> None:
                load_kb.clear()
                st.session_state.pop("kb", None)
                st.rerun()

            admin_api_key = os.getenv("GEMINI_API_KEY", "") or get_secret("GEMINI_API_KEY")
            admin_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
            render_ingestion_assistant(
                documents_dir=DOCS_DIR,
                metadata_path=METADATA_PATH,
                api_key=admin_api_key,
                model=admin_model,
                on_publish=refresh_after_publish,
            )

            st.divider()
            st.markdown("#### Respaldo para publicación permanente")
            st.download_button(
                "Descargar paquete de conocimiento",
                data=package_knowledge_base(),
                file_name="base_conocimiento_pyc.zip",
                mime="application/zip",
                use_container_width=True,
            )


with monitor_tab:
    st.subheader("Monitoreo técnico")
    cols = st.columns(4)
    cols[0].metric("Carga de base", f"{st.session_state.get('kb_load_seconds', 0):.2f} s")
    cols[1].metric("Última búsqueda", f"{st.session_state.get('last_search_seconds', 0):.3f} s")
    cols[2].metric("Última llamada LLM", f"{st.session_state.get('last_llm_seconds', 0):.2f} s")
    cols[3].metric("Último modo", st.session_state.get("last_mode", "—"))
    st.write("**Última pregunta:**", st.session_state.get("last_question", "Aún no hay consultas."))
    st.info(
        "La carga inicial puede aumentar cuando Streamlit despierta después de estar inactivo. "
        "La búsqueda local debería mantenerse bajo un segundo; la mayor variación posterior corresponde al modelo generativo."
    )

st.divider()
st.caption("Prototipo demostrativo. Las respuestas no sustituyen la revisión de la normativa oficial vigente.")
