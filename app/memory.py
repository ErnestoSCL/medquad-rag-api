"""Memoria conversacional por sesión, persistida en Supabase.

Deliberadamente NO importa de rag_chain: crea su propio cliente de Supabase.
Así el módulo queda libre de langchain y se puede testear sin instalarlo.

Identificación del usuario: un UUID que el navegador guarda en localStorage
(gr.BrowserState), no la IP. La IP se descartó porque detrás de un NAT todos
los usuarios de una misma red comparten IP pública —en una demo con varios
evaluadores en el mismo wifi compartirían historial—, cambia si es dinámica,
en Render llega la del proxy y no la del usuario, y además es un dato personal:
guardarla sería incoherente con tener un guardrail que enmascara PII.
"""
import logging
import uuid

from supabase import create_client

from app.config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

TABLE = "chat_history"
DEFAULT_TURNS = 3

_client = create_client(SUPABASE_URL, SUPABASE_KEY)


def nuevo_session_id() -> str:
    return str(uuid.uuid4())


def cargar_historial(session_id: str, turnos: int = DEFAULT_TURNS):
    """
    Devuelve las últimas `turnos` interacciones en orden cronológico, como
    lista de dicts {"role": "user"|"assistant", "content": str}.

    Un turno son dos mensajes (pregunta + respuesta), así que se piden
    turnos*2 filas. Se ordena descendente para quedarse con las más recientes
    y luego se invierte, porque el LLM necesita la conversación en orden.

    Nunca lanza: si la consulta falla, se devuelve historial vacío. Perder la
    memoria degrada la respuesta, pero no debe impedir contestar.
    """
    if not session_id:
        return []
    try:
        r = (
            _client.table(TABLE)
            .select("role, content")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .order("id", desc=True)
            .limit(turnos * 2)
            .execute()
        )
        return [{"role": m["role"], "content": m["content"]} for m in reversed(r.data)]
    except Exception as exc:
        logger.warning("no se pudo cargar el historial de %s: %s", session_id, exc)
        return []


def guardar_interaccion(session_id: str, pregunta: str, respuesta: str) -> bool:
    """
    Guarda el par pregunta/respuesta. Devuelve True si se persistió.

    `pregunta` debe venir ya sin PII (guardrails.strip_pii): el historial se
    almacena en la base, así que no debe contener datos personales.

    Nunca lanza, por el mismo motivo que cargar_historial: la respuesta ya se
    le entregó al usuario y no tiene sentido fallar después de eso.
    """
    if not session_id:
        return False
    try:
        _client.table(TABLE).insert([
            {"session_id": session_id, "role": "user", "content": pregunta},
            {"session_id": session_id, "role": "assistant", "content": respuesta},
        ]).execute()
        return True
    except Exception as exc:
        logger.warning("no se pudo guardar la interacción de %s: %s", session_id, exc)
        return False


def borrar_historial(session_id: str) -> bool:
    """Borra el historial de una sesión. Lo usan los tests para limpiar."""
    if not session_id:
        return False
    try:
        _client.table(TABLE).delete().eq("session_id", session_id).execute()
        return True
    except Exception as exc:
        logger.warning("no se pudo borrar el historial de %s: %s", session_id, exc)
        return False


def formatear_para_prompt(historial) -> str:
    """
    Convierte el historial en texto plano para insertarlo en un prompt.
    Cadena vacía si no hay historial, para que el prompt pueda omitir la
    sección entera.
    """
    if not historial:
        return ""
    etiquetas = {"user": "Usuario", "assistant": "Asistente"}
    return "\n".join(
        f"{etiquetas.get(m['role'], m['role'])}: {m['content']}" for m in historial
    )
