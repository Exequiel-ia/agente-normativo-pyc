from __future__ import annotations

import json
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import streamlit as st
from pypdf import PdfReader

try:
    from src.rag_pipeline import (
        build_knowledge_base,
        inspect_pdf,
        load_metadata,
        save_metadata,
        search_knowledge_base,
    )
except ModuleNotFoundError:
    from rag_pipeline import (
        build_knowledge_base,
        inspect_pdf,
        load_metadata,
        save_metadata,
        search_knowledge_base,
    )


CLASSIFICATIONS = [
    "Conocimiento principal",
    "Contexto complementario",
    "Metadato",
    "Referencia histórica",
    "Ejemplo",
    "Anexo técnico",
    "Excluir",
]


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ._ -]", "_", Path(name).name)
    return clean if clean.lower().endswith(".pdf") else f"{clean}.pdf"


def _extract_pages(pdf_bytes: bytes) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
        temp.write(pdf_bytes)
        temp.flush()
        reader = PdfReader(temp.name)
        return [
            {"page": number, "text": _clean(page.extract_text() or "")}
            for number, page in enumerate(reader.pages, 1)
        ]


def _detect_sections(pages: list[dict]) -> list[dict]:
    sections: list[dict] = []
    current_title = "Inicio del documento"
    current_pages: list[int] = []
    current_text: list[str] = []
    heading_pattern = re.compile(
        r"^(?:ARCHIVO\s+[A-Z]{1,4}[ -]?\d{1,4}|\d+(?:\.\d+)*\.?\s+[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ ]{4,}|[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ ]{8,})$",
        re.IGNORECASE,
    )

    def flush() -> None:
        nonlocal current_pages, current_text
        body = "\n".join(current_text).strip()
        if body:
            sections.append(
                {
                    "include": True,
                    "title": current_title[:160],
                    "pages": f"{min(current_pages)}-{max(current_pages)}" if current_pages else "—",
                    "classification": "Conocimiento principal",
                    "reason": "Contenido normativo detectado",
                    "preview": body[:900],
                }
            )
        current_pages, current_text = [], []

    for page in pages:
        text = page["text"]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        heading = next((line for line in lines[:12] if heading_pattern.match(line)), "")
        if heading and current_text:
            flush()
            current_title = heading
        current_pages.append(page["page"])
        current_text.append(text)
        if sum(len(item) for item in current_text) >= 9000:
            flush()
    flush()
    return sections[:80]


def _json_from_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("El modelo no devolvió un objeto JSON reconocible.")
    return json.loads(cleaned[start : end + 1])


def analyze_with_gemini(pages: list[dict], sections: list[dict], api_key: str, model: str) -> dict:
    from google import genai

    text_pages = [page for page in pages if page["text"]]
    per_page = max(500, min(2500, 76000 // max(1, len(text_pages))))
    source = "\n\n".join(
        f"[PÁGINA {page['page']}]\n{page['text'][:per_page]}" for page in text_pages
    )[:80000]
    section_catalog = "\n".join(
        f"S{index}: {section['title']} (páginas {section['pages']})"
        for index, section in enumerate(sections, 1)
    )
    prompt = f"""Analiza el siguiente documento para preparar su incorporación a una base de conocimiento RAG normativa.
No inventes información. Devuelve exclusivamente JSON válido con esta estructura:
{{
  "summary": "resumen ejecutivo",
  "purpose": "propósito",
  "scope": "alcance y a quién aplica",
  "processes": ["procesos afectados"],
  "actors": ["actores o responsables"],
  "obligations": [{{"rule":"obligación", "actor":"responsable", "source_page":1}}],
  "deadlines": [{{"rule":"plazo o periodicidad", "source_page":1}}],
  "exceptions": [{{"rule":"excepción o condición", "source_page":1}}],
  "key_concepts": ["conceptos y códigos importantes"],
  "relationships": ["otras normas o documentos mencionados"],
  "risks": ["ambigüedades, contradicciones o aspectos a revisar"],
  "section_guidance": [{{"section":1, "classification":"Conocimiento principal", "include":true, "reason":"justificación"}}],
  "suggested_questions": [{{"question":"pregunta de prueba", "expected":"idea esperada", "source_page":1}}]
}}

Las clasificaciones permitidas para section_guidance son: {", ".join(CLASSIFICATIONS)}.

CATÁLOGO DE SECCIONES:
{section_catalog}

DOCUMENTO:
{source}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return _json_from_response(response.text or "")


def _analysis_fallback(name: str, inspection, sections: list[dict]) -> dict:
    codes = list(inspection.codes[:40])
    questions = [
        {"question": f"¿Qué regula el documento {name}?", "expected": "Propósito general", "source_page": 1}
    ]
    questions.extend(
        {"question": f"¿Qué establece el código {code}?", "expected": "Definición y regla aplicable", "source_page": 1}
        for code in codes[:5]
    )
    return {
        "summary": "Análisis automático pendiente de Gemini.",
        "purpose": "Debe ser confirmado por el responsable.",
        "scope": "Debe ser confirmado por el responsable.",
        "processes": [],
        "actors": [],
        "obligations": [],
        "deadlines": [],
        "exceptions": [],
        "key_concepts": codes,
        "relationships": [],
        "risks": ["El análisis de contenido no fue generado por el modelo."],
        "section_guidance": [],
        "suggested_questions": questions,
        "section_count": len(sections),
    }


def _init_state() -> None:
    defaults = {
        "ingestion_step": 1,
        "draft_document": None,
        "draft_inspection": None,
        "draft_pages": None,
        "draft_sections": None,
        "draft_analysis": None,
        "draft_tests": [],
        "draft_metadata": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _records(value) -> list[dict]:
    if hasattr(value, "to_dict"):
        return value.to_dict(orient="records")
    return [dict(item) for item in value]


def _reset() -> None:
    for key in [
        "draft_document", "draft_inspection", "draft_pages", "draft_sections",
        "draft_analysis", "draft_tests", "draft_metadata",
    ]:
        st.session_state[key] = None if key != "draft_tests" else []
    st.session_state.ingestion_step = 1


def _show_analysis(analysis: dict) -> None:
    st.markdown("##### Síntesis propuesta")
    st.write(analysis.get("summary", "—"))
    left, right = st.columns(2)
    with left:
        st.markdown("**Propósito**")
        st.write(analysis.get("purpose", "—"))
        st.markdown("**Alcance**")
        st.write(analysis.get("scope", "—"))
        st.markdown("**Procesos**")
        st.write(", ".join(analysis.get("processes", [])) or "—")
        st.markdown("**Actores**")
        st.write(", ".join(analysis.get("actors", [])) or "—")
    with right:
        st.markdown("**Conceptos clave**")
        st.write(", ".join(analysis.get("key_concepts", [])) or "—")
        st.markdown("**Relaciones documentales**")
        st.write("\n".join(f"- {item}" for item in analysis.get("relationships", [])) or "—")
        st.markdown("**Riesgos o dudas para el experto**")
        st.write("\n".join(f"- {item}" for item in analysis.get("risks", [])) or "—")

    for label, key in [
        ("Obligaciones detectadas", "obligations"),
        ("Plazos y periodicidades", "deadlines"),
        ("Excepciones y condiciones", "exceptions"),
    ]:
        with st.expander(label):
            items = analysis.get(key, [])
            st.dataframe(items, use_container_width=True, hide_index=True) if items else st.info("No se detectaron elementos.")


def _apply_section_guidance(sections: list[dict], analysis: dict) -> list[dict]:
    updated = [dict(section) for section in sections]
    for guidance in analysis.get("section_guidance", []):
        try:
            index = int(guidance.get("section", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(updated):
            continue
        classification = guidance.get("classification", "Conocimiento principal")
        if classification not in CLASSIFICATIONS:
            classification = "Conocimiento principal"
        updated[index]["classification"] = classification
        updated[index]["include"] = bool(guidance.get("include", classification != "Excluir"))
        updated[index]["reason"] = str(guidance.get("reason", "Propuesta de Gemini"))[:300]
    return updated


def render_ingestion_assistant(
    documents_dir: Path,
    metadata_path: Path,
    api_key: str,
    model: str,
    on_publish: Callable[[], None],
) -> None:
    _init_state()
    st.markdown("### Asistente de incorporación de conocimiento")
    st.caption("La IA propone; el responsable humano revisa; el sistema registra y publica.")
    labels = ["Carga", "Validación", "Análisis IA", "Curación", "Pruebas", "Publicación"]
    current = st.session_state.ingestion_step
    st.progress((current - 1) / 5, text=f"Paso {current} de 6 · {labels[current - 1]}")

    if current == 1:
        st.markdown("#### 1. Carga y contexto del documento")
        upload = st.file_uploader("Documento PDF", type=["pdf"], accept_multiple_files=False)
        left, right = st.columns(2)
        with left:
            organism = st.text_input("Organismo emisor", value="CMF")
            area = st.text_input("Área, sistema o proceso", value="")
            document_type = st.selectbox("Tipo documental", ["Manual normativo", "Circular", "Norma", "Instructivo", "Procedimiento", "Otro"])
            owner = st.text_input("Responsable de la carga", value="Exequiel Catalán")
        with right:
            version = st.text_input("Versión", value=date.today().isoformat())
            effective_date = st.date_input("Fecha de vigencia", value=date.today())
            confidentiality = st.selectbox("Clasificación", ["Público", "Uso interno", "Confidencial"])
            replacement = st.selectbox("¿Reemplaza otro documento?", ["No", "Sí"])
            replaced_name = st.text_input("Documento reemplazado", disabled=replacement == "No")
        if st.button("Guardar borrador y validar", type="primary", disabled=upload is None):
            name = _safe_name(upload.name)
            st.session_state.draft_document = {"name": name, "bytes": upload.getvalue()}
            st.session_state.draft_metadata = {
                "organismo": organism, "area": area, "tipo_documental": document_type,
                "version": version, "fecha_vigencia": effective_date.isoformat(),
                "clasificacion": confidentiality, "responsable": owner,
                "reemplaza": replaced_name if replacement == "Sí" else "",
                "estado": "Borrador",
            }
            st.session_state.ingestion_step = 2
            st.rerun()

    elif current == 2:
        document = st.session_state.draft_document
        if not st.session_state.draft_inspection:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
                temp.write(document["bytes"]); temp.flush()
                inspection = inspect_pdf(Path(temp.name))
            pages = _extract_pages(document["bytes"])
            st.session_state.draft_inspection = inspection
            st.session_state.draft_pages = pages
            st.session_state.draft_sections = _detect_sections(pages)
        inspection = st.session_state.draft_inspection
        st.markdown("#### 2. Validación técnica")
        cols = st.columns(4)
        cols[0].metric("Páginas", inspection.total_pages)
        cols[1].metric("Con texto", inspection.pages_with_text)
        cols[2].metric("Caracteres", f"{inspection.characters:,}")
        cols[3].metric("Códigos", len(inspection.codes))
        existing_hashes = {}
        for pdf in documents_dir.glob("*.pdf"):
            try:
                existing_hashes[inspect_pdf(pdf).sha256] = pdf.name
            except Exception:
                pass
        duplicate = existing_hashes.get(inspection.sha256)
        if duplicate:
            st.error(f"El documento es idéntico a `{duplicate}`. No debe publicarse como una copia nueva.")
        if inspection.warnings:
            for warning in inspection.warnings:
                st.warning(warning)
        else:
            st.success("El PDF contiene texto extraíble en todas sus páginas.")
        st.write("**Códigos detectados:**", ", ".join(inspection.codes[:80]) or "Ninguno")
        st.write("**Secciones preliminares:**", len(st.session_state.draft_sections))
        back, forward = st.columns(2)
        if back.button("Volver a carga", use_container_width=True):
            st.session_state.ingestion_step = 1; st.rerun()
        if forward.button("Aprobar validación y analizar", type="primary", use_container_width=True, disabled=bool(duplicate) or not inspection.pages_with_text):
            st.session_state.ingestion_step = 3; st.rerun()

    elif current == 3:
        st.markdown("#### 3. Comprensión del documento con IA")
        if not st.session_state.draft_analysis:
            if api_key:
                if st.button("Analizar documento con Gemini", type="primary"):
                    with st.spinner("Gemini está identificando propósito, obligaciones, plazos y conceptos..."):
                        try:
                            analysis = analyze_with_gemini(
                                st.session_state.draft_pages,
                                st.session_state.draft_sections,
                                api_key,
                                model,
                            )
                            st.session_state.draft_analysis = analysis
                            st.session_state.draft_sections = _apply_section_guidance(
                                st.session_state.draft_sections, analysis
                            )
                        except Exception as exc:
                            st.error(f"No fue posible completar el análisis: {type(exc).__name__}")
            else:
                st.warning("No se encontró GEMINI_API_KEY. Se puede continuar con un análisis básico.")
                if st.button("Generar análisis básico"):
                    st.session_state.draft_analysis = _analysis_fallback(
                        st.session_state.draft_document["name"],
                        st.session_state.draft_inspection,
                        st.session_state.draft_sections,
                    )
        if st.session_state.draft_analysis:
            _show_analysis(st.session_state.draft_analysis)
            expert_notes = st.text_area(
                "Observaciones o correcciones del experto",
                value=st.session_state.draft_analysis.get("expert_notes", ""),
                placeholder="Ej.: La sección de carátula también debe considerarse conocimiento principal...",
            )
            st.session_state.draft_analysis["expert_notes"] = expert_notes
            st.info("Revisa especialmente vigencia, obligaciones, plazos y excepciones. El modelo puede proponer; el experto debe confirmar.")
            back, forward = st.columns(2)
            if back.button("Volver a validación", use_container_width=True):
                st.session_state.ingestion_step = 2; st.rerun()
            if forward.button("Aceptar análisis y curar secciones", type="primary", use_container_width=True):
                st.session_state.ingestion_step = 4; st.rerun()

    elif current == 4:
        st.markdown("#### 4. Curación del conocimiento")
        st.write("Decide qué se incorporará. Puedes cambiar la clasificación o excluir secciones.")
        edited = st.data_editor(
            st.session_state.draft_sections,
            use_container_width=True,
            hide_index=True,
            column_config={
                "include": st.column_config.CheckboxColumn("Incluir"),
                "title": st.column_config.TextColumn("Sección"),
                "pages": st.column_config.TextColumn("Páginas", disabled=True),
                "classification": st.column_config.SelectboxColumn("Clasificación", options=CLASSIFICATIONS),
                "reason": st.column_config.TextColumn("Justificación"),
                "preview": st.column_config.TextColumn("Vista previa", disabled=True, width="large"),
            },
            key="curation_editor",
        )
        edited_records = _records(edited)
        st.session_state.draft_sections = edited_records
        included = sum(1 for row in edited_records if row.get("include") and row.get("classification") != "Excluir")
        st.metric("Secciones que se incorporarán", included)
        back, forward = st.columns(2)
        if back.button("Volver al análisis", use_container_width=True):
            st.session_state.ingestion_step = 3; st.rerun()
        if forward.button("Guardar curación y preparar pruebas", type="primary", use_container_width=True, disabled=included == 0):
            st.session_state.ingestion_step = 5; st.rerun()

    elif current == 5:
        st.markdown("#### 5. Pruebas antes de publicar")
        suggestions = st.session_state.draft_analysis.get("suggested_questions", [])
        default_questions = "\n".join(item.get("question", "") for item in suggestions if item.get("question"))
        questions_text = st.text_area(
            "Una pregunta por línea. Puedes corregir las propuestas o agregar preguntas propias.",
            value=default_questions,
            height=180,
        )
        if st.button("Ejecutar pruebas de recuperación", type="primary"):
            questions = [line.strip() for line in questions_text.splitlines() if line.strip()]
            with tempfile.TemporaryDirectory() as temp_dir:
                staging = Path(temp_dir)
                for pdf in documents_dir.glob("*.pdf"):
                    (staging / pdf.name).write_bytes(pdf.read_bytes())
                (staging / st.session_state.draft_document["name"]).write_bytes(st.session_state.draft_document["bytes"])
                with st.spinner("Construyendo índice de prueba..."):
                    test_kb = build_knowledge_base(staging, metadata_path)
                results = []
                for question in questions:
                    retrieved = search_knowledge_base(test_kb, question, top_k=5)
                    uses_new = any(item.source == st.session_state.draft_document["name"] for item in retrieved[:3])
                    results.append(
                        {
                            "question": question,
                            "status": "Aprobada" if uses_new else "Revisar",
                            "top_source": retrieved[0].source if retrieved else "Sin resultado",
                            "top_page": retrieved[0].page if retrieved else "—",
                            "new_document_found": uses_new,
                        }
                    )
                st.session_state.draft_tests = results
        if st.session_state.draft_tests:
            st.dataframe(st.session_state.draft_tests, use_container_width=True, hide_index=True)
            approved = sum(item["status"] == "Aprobada" for item in st.session_state.draft_tests)
            total = len(st.session_state.draft_tests)
            st.metric("Pruebas con el nuevo documento entre los 3 primeros resultados", f"{approved}/{total}")
            if approved < total:
                st.warning("Hay preguntas que requieren revisar la curación, fragmentación o formulación antes de publicar.")
        back, forward = st.columns(2)
        if back.button("Volver a curación", use_container_width=True):
            st.session_state.ingestion_step = 4; st.rerun()
        can_publish = bool(st.session_state.draft_tests)
        if forward.button("Continuar a aprobación", type="primary", use_container_width=True, disabled=not can_publish):
            st.session_state.ingestion_step = 6; st.rerun()

    elif current == 6:
        st.markdown("#### 6. Aprobación y publicación")
        inspection = st.session_state.draft_inspection
        included = [
            row for row in st.session_state.draft_sections
            if row.get("include") and row.get("classification") != "Excluir"
        ]
        approved = sum(item["status"] == "Aprobada" for item in st.session_state.draft_tests)
        summary = {
            "Documento": st.session_state.draft_document["name"],
            "Páginas procesadas": f"{inspection.pages_with_text}/{inspection.total_pages}",
            "Códigos detectados": len(inspection.codes),
            "Secciones incluidas": len(included),
            "Secciones excluidas": len(st.session_state.draft_sections) - len(included),
            "Pruebas aprobadas": f"{approved}/{len(st.session_state.draft_tests)}",
            "Responsable": st.session_state.draft_metadata.get("responsable", "—"),
        }
        st.dataframe([summary], use_container_width=True, hide_index=True)
        version_id = datetime.now().strftime("%Y.%m.%d-%H%M")
        st.code(f"Versión de ingesta propuesta: {version_id}")
        expert_confirm = st.checkbox("Confirmo que revisé el análisis, la vigencia, los plazos y las excepciones.")
        publish_confirm = st.checkbox("Autorizo publicar este documento en la base de conocimiento.")
        if approved < len(st.session_state.draft_tests):
            st.warning("La publicación contiene pruebas marcadas para revisión. Solo continúa si comprendes y aceptas el riesgo.")
        back, publish = st.columns(2)
        if back.button("Volver a pruebas", use_container_width=True):
            st.session_state.ingestion_step = 5; st.rerun()
        if publish.button("Publicar conocimiento", type="primary", use_container_width=True, disabled=not (expert_confirm and publish_confirm)):
            document = st.session_state.draft_document
            (documents_dir / document["name"]).write_bytes(document["bytes"])
            metadata = load_metadata(metadata_path)
            record = dict(st.session_state.draft_metadata)
            record.update(
                {
                    "estado": "Vigente",
                    "fecha_ingesta": datetime.now(timezone.utc).isoformat(),
                    "version_ingesta": version_id,
                    "sha256": inspection.sha256,
                    "secciones_incluidas": str(len(included)),
                    "pruebas_aprobadas": f"{approved}/{len(st.session_state.draft_tests)}",
                    "resumen_ia": st.session_state.draft_analysis.get("summary", ""),
                }
            )
            metadata[document["name"]] = record
            save_metadata(metadata_path, metadata)
            governance_path = metadata_path.with_name("knowledge_governance.json")
            try:
                governance = json.loads(governance_path.read_text(encoding="utf-8")) if governance_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                governance = {}
            governance[document["name"]] = {
                "version_ingesta": version_id,
                "metadata": record,
                "analysis": st.session_state.draft_analysis,
                "curation": st.session_state.draft_sections,
                "tests": st.session_state.draft_tests,
                "approvals": {
                    "expert_review_confirmed": True,
                    "publication_authorized": True,
                    "approved_by": record.get("responsable", ""),
                    "approved_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            governance_path.write_text(json.dumps(governance, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success("Conocimiento publicado. La base será reconstruida.")
            _reset()
            on_publish()
