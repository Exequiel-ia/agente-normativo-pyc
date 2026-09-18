from __future__ import annotations

from rag_pipeline import SearchResult


SYSTEM_INSTRUCTION = """Eres el Agente Normativo PyC.
Responde exclusivamente con el contexto proporcionado.
No inventes normas, fechas, plazos, campos ni obligaciones.
Si existe información contradictoria, indícalo.
Si la evidencia no basta, responde que no encontraste evidencia suficiente.
Redacta en español claro y profesional.
Incluye al final una sección 'Fuentes' con documento y página.
No digas que consultaste internet; solo utilizas los documentos recuperados.
"""


def _context(results: list[SearchResult]) -> str:
    blocks = []
    for i, result in enumerate(results, 1):
        blocks.append(
            f"[FUENTE {i}: {result.source}, página {result.page}]\n{result.text}"
        )
    return "\n\n".join(blocks)


def _extractive_answer(results: list[SearchResult]) -> str:
    lines = [
        "**Modo demostración sin LLM:** estos son los fragmentos con mayor relación con la pregunta:",
        "",
    ]
    for item in results[:3]:
        snippet = item.text[:700].strip()
        lines.append(f"- **{item.source}, página {item.page}:** {snippet}")
    lines.extend(
        [
            "",
            "> Para obtener una respuesta redactada y sintetizada, configura la clave del modelo generativo.",
        ]
    )
    return "\n\n".join(lines)


def generate_answer(
    question: str,
    results: list[SearchResult],
    provider: str,
    api_key: str,
) -> tuple[str, str]:
    if provider != "Gemini" or not api_key:
        return _extractive_answer(results), "demostración extractiva"

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"PREGUNTA DEL USUARIO:\n{question}\n\n"
            f"CONTEXTO DOCUMENTAL:\n{_context(results)}"
        )
        response = client.models.generate_content(
           model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("El modelo no devolvió contenido.")
        return text, "Gemini con RAG"

    except Exception as exc:
        fallback = _extractive_answer(results)
        return (
            f"No fue posible utilizar el modelo generativo en esta consulta. "
            f"Se muestra la evidencia recuperada.\n\n{fallback}\n\n"
            f"Detalle técnico: `{type(exc).__name__}: {str(exc)}`",
            "respaldo extractivo",
        )
