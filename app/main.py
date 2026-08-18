from fastapi import FastAPI
import gradio as gr

from app.schemas import AskRequest, AskResponse
from app.rag_chain import answer_question
from app.guardrails import apply_guardrails, apply_clinical_guardrails, contains_toxicity

app = FastAPI(title="Asistente Médico RAG API")


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    guard = apply_guardrails(payload.question)
    if not guard["allowed"]:
        return AskResponse(
            answer=f"No puedo procesar esta pregunta (motivo: {guard['reason']}).",
            citations=[],
        )

    raw_answer, docs = answer_question(guard["safe_question"], guard["search_question"])

    is_toxic, terms = contains_toxicity(raw_answer)
    if is_toxic:
        final_answer = f"Respuesta bloqueada por el filtro de toxicidad (términos: {terms})."
    else:
        final_answer = apply_clinical_guardrails(raw_answer)

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
