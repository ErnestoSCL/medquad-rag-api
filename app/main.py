from fastapi import FastAPI
import gradio as gr

from app.schemas import AskRequest, AskResponse
from app.rag_chain import answer_question
from app.guardrails import (
    apply_guardrails,
    apply_clinical_guardrails,
    contains_toxicity,
    INSUFFICIENT_INFO_MSG,
)

app = FastAPI(title="Asistente Médico RAG API")


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    guard = apply_guardrails(payload.question)
    if not guard["allowed"]:
        return AskResponse(
            answer=f"No puedo procesar esta pregunta (motivo: {guard['reason']}).",
            citations=[],
        )

    # Se usa search_question (PII eliminado), no safe_question (PII sustituido
    # por placeholders): los placeholders degradan tanto la búsqueda como la
    # generación. safe_question queda disponible en `guard` para logs.
    raw_answer, docs = answer_question(guard["search_question"])

    is_toxic, terms = contains_toxicity(raw_answer)
    if is_toxic:
        final_answer = f"Respuesta bloqueada por el filtro de toxicidad (términos: {terms})."
    else:
        final_answer = apply_clinical_guardrails(raw_answer)

    # Si el modelo no encontró la respuesta, los chunks recuperados no
    # respaldan nada: mostrarlos como fuentes sería contradictorio. Pasa con
    # preguntas ajenas al corpus, donde el corte relativo de rag_chain no
    # filtra nada porque todos los resultados son igual de malos.
    if final_answer.startswith(INSUFFICIENT_INFO_MSG):
        docs = []

    citations = []
    for d in docs:
        m = d.metadata
        frag = f" (fragmento {m['chunk_id'] + 1} de {m['n_chunks']})" if m.get("n_chunks", 1) > 1 else ""
        citations.append(f"[{m.get('document_source')}] {m.get('question_focus')}{frag} — {m.get('document_url')}")

    return AskResponse(answer=final_answer, citations=citations)


# --- Interfaz Gradio, montada sobre el mismo FastAPI ---
def gradio_ask(question):
    result = ask(AskRequest(question=question))
    sources_md = "\n".join(f"- {c}" for c in result.citations) or "_(sin fuentes)_"
    return result.answer, sources_md


demo = gr.Interface(
    fn=gradio_ask,
    inputs=gr.Textbox(label="Tu pregunta médica", placeholder="Ej: What are the symptoms of Bell's palsy?"),
    outputs=[gr.Textbox(label="Respuesta"), gr.Markdown(label="Fuentes")],
    title="🩺 Asistente Médico RAG — MedQuAD",
)

app = gr.mount_gradio_app(app, demo, path="/")
