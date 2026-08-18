from fastapi import FastAPI
import gradio as gr

from app.schemas import AskRequest, AskResponse
from app.rag_chain import answer_question
from app.memory import (
    cargar_historial,
    guardar_interaccion,
    nuevo_session_id,
)
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

    session_id = payload.session_id
    historial = cargar_historial(session_id) if session_id else []

    # Se usa search_question (PII eliminado), no safe_question (PII sustituido
    # por placeholders): los placeholders degradan tanto la búsqueda como la
    # generación. safe_question queda disponible en `guard` para logs.
    pregunta_limpia = guard["search_question"]
    raw_answer, docs, consulta = answer_question(pregunta_limpia, historial)

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

    # Se guarda la pregunta ya sin PII: el historial vive en la base de datos y
    # no debe contener datos personales.
    if session_id:
        guardar_interaccion(session_id, pregunta_limpia, final_answer)

    return AskResponse(answer=final_answer, citations=citations, search_query=consulta)


# --- Interfaz Gradio (vista de chat), montada sobre el mismo FastAPI ---

def responder_chat(mensaje, history, session_id):
    """
    `history` lo administra Gradio para pintar la conversación; la fuente de
    verdad del backend es Supabase, que se consulta dentro de /ask con el
    session_id. Así el historial sobrevive a un refresco de la página.
    """
    resultado = ask(AskRequest(question=mensaje, session_id=session_id))

    respuesta = resultado.answer
    if resultado.citations:
        fuentes = "\n".join(f"- {c}" for c in resultado.citations)
        respuesta += f"\n\n---\n**Fuentes:**\n{fuentes}"
    return respuesta


def asegurar_session_id(session_id):
    """
    Genera el identificador la primera vez que alguien abre la página.

    Vive en gr.BrowserState, o sea en el localStorage del navegador: sobrevive
    a recargas y distingue usuarios sin recurrir a la IP, que detrás de un NAT
    es la misma para toda una red y además es un dato personal.
    """
    return session_id or nuevo_session_id()


with gr.Blocks(title="Asistente Médico RAG — MedQuAD") as demo:
    session_state = gr.BrowserState(None, storage_key="medquad_session_id")

    gr.Markdown(
        "# 🩺 Asistente Médico RAG — MedQuAD\n"
        "Preguntá en español sobre síntomas, tratamientos o enfermedades. "
        "Las respuestas se basan únicamente en el corpus MedQuAD de los "
        "Institutos Nacionales de Salud de EE. UU., y se citan las fuentes.\n\n"
        "Recuerda las últimas 3 interacciones, así que podés repreguntar: "
        "*«¿Qué es la parálisis de Bell?»* y luego *«¿y cómo se trata?»*.\n\n"
        "Ejemplos: ¿Cuáles son los síntomas de la parálisis de Bell? · "
        "¿Cómo se trata el asma? · ¿Qué causa los dolores de cabeza? · "
        "¿Qué es la enfermedad de Wilson?\n\n"
        "*Herramienta con fines educativos. No sustituye el criterio de un "
        "profesional médico colegiado.*"
    )

    # Sin `examples`: ChatInterface los exige como listas de listas cuando hay
    # additional_inputs, lo que obligaría a fijar session_id=None en cada
    # ejemplo y dejaría esas consultas sin memoria. Van arriba como texto.
    gr.ChatInterface(
        fn=responder_chat,
        additional_inputs=[session_state],
        textbox=gr.Textbox(
            placeholder="Escribí tu pregunta médica…",
            show_label=False,
            autofocus=True,
        ),
    )

    # Al cargar la página se asegura que exista un session_id persistente.
    demo.load(asegurar_session_id, inputs=[session_state], outputs=[session_state])

app = gr.mount_gradio_app(app, demo, path="/")
