"""Memoria conversacional, persistida en Supabase.

Dos niveles de identificación:

  session_id       el usuario. Un UUID que el navegador guarda en localStorage.
                   No se usa la IP: detrás de un NAT todos los usuarios de una
                   red comparten IP pública —en una demo con varios evaluadores
                   en el mismo wifi compartirían historial—, cambia si es
                   dinámica, en Render llega la del proxy, y además es un dato
                   personal, incoherente con tener un guardrail que enmascara
                   PII.

  conversation_id  cada chat individual de ese usuario, para poder listarlos en
                   la barra lateral y retomarlos.

Deliberadamente NO importa de rag_chain: crea su propio cliente de Supabase, así
el módulo queda libre de langchain y se puede testear sin instalarlo.
"""
import logging
import uuid

from supabase import create_client

from app.config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

TABLA_MENSAJES = "chat_history"
TABLA_CONVERSACIONES = "conversations"
DEFAULT_TURNS = 3
TITULO_MAX = 60


def nuevo_id() -> str:
    return str(uuid.uuid4())


# Alias histórico: main.py y los tests lo usan para el id de usuario.
nuevo_session_id = nuevo_id


# --------------------------------------------------------------- mensajes

def cargar_historial(conversation_id: str, turnos: int = DEFAULT_TURNS):
    """
    Últimas `turnos` interacciones de una conversación, en orden cronológico,
    como lista de dicts {"role", "content"}.

    Un turno son dos mensajes, así que se piden turnos*2 filas: se ordena
    descendente para quedarse con las más recientes y luego se invierte, porque
    el LLM necesita la conversación en orden.

    Nunca lanza: sin memoria la respuesta es peor, pero se responde igual.
    """
    if not conversation_id:
        return []
    try:
        r = (
            _cliente().table(TABLA_MENSAJES)
            .select("role, content")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=True)
            .order("id", desc=True)
            .limit(turnos * 2)
            .execute()
        )
        return [{"role": m["role"], "content": m["content"]} for m in reversed(r.data)]
    except Exception as exc:
        logger.warning("no se pudo cargar el historial de %s: %s", conversation_id, exc)
        return []


def cargar_conversacion_completa(conversation_id: str):
    """
    Todos los mensajes de una conversación, para repintar el chat cuando el
    usuario la retoma desde la barra lateral. A diferencia de cargar_historial,
    que acota a los últimos turnos para el LLM, acá se quiere todo.
    """
    if not conversation_id:
        return []
    try:
        r = (
            _cliente().table(TABLA_MENSAJES)
            .select("role, content")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .order("id")
            .execute()
        )
        return [{"role": m["role"], "content": m["content"]} for m in r.data]
    except Exception as exc:
        logger.warning("no se pudo cargar la conversación %s: %s", conversation_id, exc)
        return []


def guardar_interaccion(session_id: str, conversation_id: str,
                        pregunta: str, respuesta: str) -> bool:
    """
    Guarda el par pregunta/respuesta.

    `pregunta` debe venir sin PII (guardrails.strip_pii) y `respuesta` sin el
    disclaimer clínico: el historial se le pasa al LLM como turnos previos, y si
    las respuestas guardadas terminaran con la nota legal, el modelo imitaría
    ese patrón y la repetiría. El disclaimer es capa de presentación.

    Nunca lanza: la respuesta ya se le entregó al usuario, no tiene sentido
    fallar después de eso.
    """
    if not conversation_id:
        return False
    try:
        _cliente().table(TABLA_MENSAJES).insert([
            {"session_id": session_id, "conversation_id": conversation_id,
             "role": "user", "content": pregunta},
            {"session_id": session_id, "conversation_id": conversation_id,
             "role": "assistant", "content": respuesta},
        ]).execute()
        return True
    except Exception as exc:
        logger.warning("no se pudo guardar la interacción de %s: %s", conversation_id, exc)
        return False


def borrar_historial(identificador: str) -> bool:
    """
    Borra los mensajes de una conversación o de un usuario entero (acepta
    cualquiera de los dos ids). Lo usan los tests para limpiar.
    """
    if not identificador:
        return False
    try:
        c = _cliente()
        c.table(TABLA_MENSAJES).delete().eq("conversation_id", identificador).execute()
        c.table(TABLA_MENSAJES).delete().eq("session_id", identificador).execute()
        c.table(TABLA_CONVERSACIONES).delete().eq("id", identificador).execute()
        c.table(TABLA_CONVERSACIONES).delete().eq("session_id", identificador).execute()
        return True
    except Exception as exc:
        logger.warning("no se pudo borrar %s: %s", identificador, exc)
        return False


# ---------------------------------------------------------- conversaciones

def crear_conversacion(session_id: str, titulo: str = "Nueva conversación") -> str:
    """Registra una conversación nueva y devuelve su id."""
    cid = nuevo_id()
    if not session_id:
        return cid
    try:
        _cliente().table(TABLA_CONVERSACIONES).insert({
            "id": cid, "session_id": session_id, "title": titulo[:TITULO_MAX],
        }).execute()
    except Exception as exc:
        logger.warning("no se pudo crear la conversación: %s", exc)
    return cid


def listar_conversaciones(session_id: str, limite: int = 30):
    """
    Conversaciones del usuario, de la más reciente a la más antigua, como
    lista de (titulo, id) — el formato que espera un gr.Radio.
    """
    if not session_id:
        return []
    try:
        r = (
            _cliente().table(TABLA_CONVERSACIONES)
            .select("id, title")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limite)
            .execute()
        )
        return [(c["title"], c["id"]) for c in r.data]
    except Exception as exc:
        logger.warning("no se pudieron listar las conversaciones: %s", exc)
        return []


def titular_conversacion(conversation_id: str, primera_pregunta: str) -> None:
    """
    Usa la primera pregunta como título, igual que hacen la mayoría de los
    chats. Solo se aplica si el título sigue siendo el genérico, para no
    renombrar una conversación en curso.
    """
    if not conversation_id or not primera_pregunta:
        return
    titulo = " ".join(primera_pregunta.split())[:TITULO_MAX]
    try:
        c = _cliente()
        actual = (
            c.table(TABLA_CONVERSACIONES).select("title")
            .eq("id", conversation_id).limit(1).execute()
        )
        if actual.data and actual.data[0]["title"] != "Nueva conversación":
            return
        c.table(TABLA_CONVERSACIONES).update({"title": titulo}).eq("id", conversation_id).execute()
    except Exception as exc:
        logger.warning("no se pudo titular la conversación %s: %s", conversation_id, exc)


def borrar_conversacion(conversation_id: str) -> bool:
    if not conversation_id:
        return False
    try:
        c = _cliente()
        c.table(TABLA_MENSAJES).delete().eq("conversation_id", conversation_id).execute()
        c.table(TABLA_CONVERSACIONES).delete().eq("id", conversation_id).execute()
        return True
    except Exception as exc:
        logger.warning("no se pudo borrar la conversación %s: %s", conversation_id, exc)
        return False


# ------------------------------------------------------------------ varios

def formatear_para_prompt(historial) -> str:
    """
    Historial como texto plano para insertarlo en un prompt. Cadena vacía si no
    hay nada, para que el prompt pueda omitir la sección entera.
    """
    if not historial:
        return ""
    etiquetas = {"user": "Usuario", "assistant": "Asistente"}
    return "\n".join(
        f"{etiquetas.get(m['role'], m['role'])}: {m['content']}" for m in historial
    )


_conn = None


def _cliente():
    """Cliente perezoso: evita abrir la conexión al importar el módulo."""
    global _conn
    if _conn is None:
        _conn = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _conn
