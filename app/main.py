from fastapi import FastAPI
import gradio as gr

from app.schemas import AskRequest, AskResponse
from app.rag_chain import answer_question
from app.citations import formatear_cita, fuentes_html
from app.conversacion import es_conversacional, respuesta_conversacional
from app.ui import construir
from app.memory import (
    cargar_historial,
    crear_conversacion,
    guardar_interaccion,
    listar_conversaciones,
    nuevo_id,
    titular_conversacion,
)
from app.guardrails import (
    apply_guardrails,
    apply_clinical_guardrails,
    contains_toxicity,
    INSUFFICIENT_INFO_MSG,
)

app = FastAPI(title="Asistente Médico RAG API")


def _resolver(pregunta: str, conversation_id: str | None):
    """
    Núcleo compartido por la API y la interfaz.

    Devuelve (respuesta_final, respuesta_cruda, docs, consulta_de_busqueda,
    pregunta_sin_pii). La respuesta cruda se devuelve aparte porque es la que
    va al historial: si se guardara la final, con el disclaimer, el LLM vería
    esa nota en los turnos previos y la repetiría.
    """
    guard = apply_guardrails(pregunta)
    if not guard["allowed"]:
        msg = f"No puedo procesar esta pregunta (motivo: {guard['reason']})."
        return msg, None, [], None, None

    # Saludos y preguntas sobre el propio asistente no van al RAG: no hay nada
    # que recuperar y terminaban en "No hay información suficiente", que es una
    # respuesta desconcertante para el primer mensaje de cualquier usuario.
    # Se responden después de los guardrails para que una inyección disfrazada
    # de saludo siga bloqueándose.
    if es_conversacional(pregunta):
        return respuesta_conversacional(), None, [], None, None

    historial = cargar_historial(conversation_id) if conversation_id else []
    pregunta_limpia = guard["search_question"]
    cruda, docs, consulta = answer_question(pregunta_limpia, historial)

    es_toxica, terminos = contains_toxicity(cruda)
    if es_toxica:
        final = f"Respuesta bloqueada por el filtro de toxicidad (términos: {terminos})."
        return final, None, [], consulta, pregunta_limpia

    final = apply_clinical_guardrails(cruda)

    # Si el modelo no encontró la respuesta, los chunks no respaldan nada:
    # mostrarlos como fuentes sería contradictorio.
    if final.startswith(INSUFFICIENT_INFO_MSG):
        docs = []

    return final, cruda, docs, consulta, pregunta_limpia


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    cid = payload.session_id
    final, cruda, docs, consulta, limpia = _resolver(payload.question, cid)

    citations = [formatear_cita(d.metadata) for d in docs]

    if cid and cruda is not None:
        guardar_interaccion(cid, cid, limpia, cruda)

    return AskResponse(answer=final, citations=citations, search_query=consulta)


# ---------------------------------------------------------------- interfaz

def _iniciar_sesion(user_id):
    """Primera carga: asegura id de usuario y abre una conversación."""
    user_id = user_id or nuevo_id()
    opciones = listar_conversaciones(user_id)
    cid = opciones[0][1] if opciones else crear_conversacion(user_id)
    return user_id, cid, listar_conversaciones(user_id)


def _responder(mensaje, historial_ui, user_id, conversation_id):
    """
    Un turno de chat. Devuelve (entrada_vaciada, historial_ui, opciones).

    `historial_ui` es lo que pinta Gradio; la fuente de verdad del backend es
    Supabase, que se consulta con el conversation_id.
    """
    mensaje = (mensaje or "").strip()
    if not mensaje:
        return gr.skip(), gr.skip(), gr.skip()

    historial_ui = list(historial_ui or [])
    if not conversation_id:
        conversation_id = crear_conversacion(user_id)

    final, cruda, docs, _, limpia = _resolver(mensaje, conversation_id)

    respuesta = final + fuentes_html([d.metadata for d in docs])

    historial_ui.append({"role": "user", "content": mensaje})
    historial_ui.append({"role": "assistant", "content": respuesta})

    if cruda is not None:
        guardar_interaccion(user_id, conversation_id, limpia, cruda)
        titular_conversacion(conversation_id, mensaje)

    opciones = listar_conversaciones(user_id)
    return "", historial_ui, gr.update(choices=opciones, value=conversation_id)


demo = construir(_responder, _iniciar_sesion)
app = gr.mount_gradio_app(app, demo, path="/")
