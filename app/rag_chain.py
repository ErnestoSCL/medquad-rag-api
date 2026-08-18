import json
import logging

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

from app.config import SUPABASE_URL, SUPABASE_KEY
from app.memory import formatear_para_prompt

logger = logging.getLogger(__name__)

# `dimensions` debe coincidir con vector(512) en sql/schema.sql y con
# EMBEDDING_DIMENSIONS en scripts/ingest_to_supabase.py: si las tres no están
# alineadas, pgvector rechaza la consulta por dimensión incompatible.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)
TOP_K = 5

# Corte relativo: se descartan los chunks cuya similitud sea menor al 80% de
# la del mejor resultado de esa misma consulta.
#
# Por qué relativo y no un umbral fijo: los chunks irrelevantes no tienen una
# similitud baja en términos absolutos (llegan a 0.565), y los relevantes
# pueden ser bajos (bajan a 0.498) — los rangos se solapan, así que un umbral
# fijo perdería resultados buenos. Lo que sí distingue es la caída respecto al
# mejor de la propia consulta.
#
# QUÉ HACE Y QUÉ NO, según la evaluación con LLM-as-judge (ver eval/judge.py):
# no es el mecanismo que garantiza la calidad de las citas — de eso se encarga
# el reranker, que por sí solo deja 0% de chunks inútiles. Con reranker activo,
# quitar este corte da prácticamente el mismo resultado (42 chunks útiles vs 41,
# 0% inútiles en ambos casos).
#
# Se mantiene por dos razones que el reranker no cubre:
#   · Acota su entrada. Sin el corte el reranker recibe los 20 candidatos
#     completos (~2000 tokens) en vez de los 5-10 que sobreviven, lo que agrega
#     latencia y costo a cada consulta.
#   · Es determinista. El reranker es un LLM y puede variar aun con
#     temperature=0; este filtro es aritmética pura y siempre descarta lo mismo.
#
# Sin reranker (RERANK_ENABLED = False) sí pasa a ser el filtro principal, y ahí
# su aporte es grande: reduce los chunks inútiles del 28% al 12%.
RELATIVE_CUTOFF = 0.80

# Reranking con LLM: se recuperan RERANK_CANDIDATES por similitud y el modelo
# elige cuáles pasan al contexto, hasta TOP_K.
#
# Es el mecanismo que realmente controla la calidad de las citas. Evaluado con
# LLM-as-judge sobre 15 preguntas (ver eval/judge.py), juzgando el CONTENIDO de
# cada chunk y no su etiqueta:
#
#   configuración          citas   útiles   inútiles
#   solo similitud            75      51%       28%
#   + corte relativo          64      59%       12%
#   + reranking               51      80%        0%
#
# Elimina por completo los chunks inútiles, algo que el corte relativo por sí
# solo no logra, porque distingue utilidad de pertenencia temática: un fragmento
# sobre el tratamiento de Bell's palsy es del tema correcto pero no sirve para
# responder por los síntomas, y el reranker lo descarta.
#
# Costo medido: +0.4 s por consulta. Para desactivarlo basta con
# RERANK_ENABLED = False; el pipeline vuelve a similitud + corte relativo.
RERANK_ENABLED = True
RERANK_CANDIDATES = 20

# Se le pasa el chunk completo al reranker, no un recorte. Con 300 caracteres
# descartaba pasajes válidos: el chunk de "Caffey disease" es el fragmento 3 de
# 6 y su definición no entraba en el recorte, así que el reranker lo juzgaba
# inútil y dejaba el contexto vacío. Los chunks rondan los 424 caracteres, así
# que el ahorro de tokens no justificaba el riesgo.
RERANK_SNIPPET_CHARS = 1200

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# El reranker trabaja en inglés a propósito: juzga pasajes en inglés con una
# consulta ya traducida. Pasarle texto en español empeoraría la comparación.
RERANK_PROMPT = (
    "You are ranking retrieved passages for a medical question.\n"
    "Select ONLY the passages that contain information useful to answer it, "
    "ordered most useful first, at most {top_k}.\n"
    "Be strict: it is better to return two good passages than five padded with "
    "loosely related ones. If none are useful, return an empty list.\n"
    "Reply with ONLY a JSON array of passage numbers, e.g. [3, 1, 7]."
)


def parse_rerank_response(respuesta: str, n_candidatos: int) -> list[int]:
    """
    Traduce la respuesta cruda del reranker a índices 0-based válidos.

    Está separada de _rerank —y sin dependencias de langchain— para poder
    testear el parseo sin red ni modelo: es la parte frágil (el LLM devuelve
    texto libre) y la que más vale la pena cubrir con tests.

    Lanza ValueError si la respuesta no es una lista JSON. Los índices fuera de
    rango o de tipo incorrecto se descartan en silencio, que es lo razonable:
    un elemento inválido no debería invalidar los demás.
    """
    limpia = respuesta.strip()

    # El modelo a veces envuelve el JSON en ```json ... ```
    if limpia.startswith("```"):
        limpia = limpia.strip("`").removeprefix("json").strip()

    elegidos = json.loads(limpia)
    if not isinstance(elegidos, list):
        raise ValueError(f"se esperaba una lista, llegó {type(elegidos).__name__}")

    return [
        n - 1
        for n in elegidos
        if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= n_candidatos
    ]


def _rerank(question, scored):
    """
    Recibe [(Document, score), ...] y devuelve los Documents que el LLM
    considera útiles, hasta TOP_K.

    Ante cualquier respuesta inesperada del modelo cae a los primeros TOP_K por
    similitud: el reranking es una mejora opcional, nunca un punto de fallo.
    """
    if not scored:
        return []

    pasajes = "\n\n".join(
        f"[{i}] {doc.page_content[:RERANK_SNIPPET_CHARS]}"
        for i, (doc, _) in enumerate(scored, 1)
    )
    try:
        respuesta = llm.invoke([
            {"role": "system", "content": RERANK_PROMPT.format(top_k=TOP_K)},
            {"role": "user", "content": f"Question:\n{question}\n\nPassages:\n{pasajes}"},
        ]).content

        indices = parse_rerank_response(respuesta, len(scored))
        return [scored[i][0] for i in indices][:TOP_K]

    except Exception as exc:
        logger.warning("reranking fallido (%s), se usa el orden por similitud", exc)
        return [doc for doc, _ in scored[:TOP_K]]


# ------------------------------------------------------------------
# Reformulación: traducir al inglés y resolver referencias al historial
# ------------------------------------------------------------------
# El corpus (MedQuAD) está íntegramente en inglés, pero los usuarios preguntan
# en español. Buscar directo con la pregunta en español degrada la
# recuperación: medido sobre 8 pares equivalentes, recupera 3.0/5 de los chunks
# que devuelve el inglés (y en dos casos, 0/5), con la similitud cayendo de
# 0.706 a 0.568. Traduciendo primero sube a 4.9/5 y 0.702 — prácticamente la
# calidad nativa. Costo: +0.76 s por consulta.
#
# La misma llamada resuelve las preguntas de seguimiento, que necesitan lo
# mismo (el historial) y de otro modo exigirían una segunda llamada:
#   historial: "¿Qué es la parálisis de Bell?" -> "..."
#   pregunta : "¿y cómo se trata?"
#   salida   : "How is Bell's palsy treated?"
#
# La regla sobre la primera persona no es cosmética. Sin ella, "¿Qué causa mis
# dolores de cabeza?" se traducía como "What causes MY headaches?", y con ese
# posesivo el reranker descartaba los 20 pasajes —ninguno habla de los dolores
# de cabeza de esa persona en particular— dejando el contexto vacío y forzando
# la abstención. Es el mismo problema que ya se había corregido en el generador,
# reaparecido una etapa antes. Una consulta a un índice documental tiene que ser
# impersonal; la respuesta sí se redacta contra la pregunta original.
REFORMULATE_PROMPT = (
    "You rewrite a user's medical question into a standalone English search "
    "query for a document retrieval system.\n"
    "Rules:\n"
    "- Always output English, regardless of the input language.\n"
    "- Resolve pronouns and ellipsis using the conversation history: if the "
    "user asks a follow-up like \"and how is it treated?\", name the condition "
    "explicitly.\n"
    "- Make it impersonal. Drop first-person framing: \"what causes my "
    "headaches?\" becomes \"what causes headaches?\", \"I have migraines, what "
    "can I take?\" becomes \"migraine treatment\".\n"
    "- Keep it a question or a short noun phrase. Do not answer it.\n"
    "- Do not add information that the user did not provide.\n"
    "Reply with ONLY the rewritten query."
)


def reformular(pregunta: str, historial=None) -> str:
    """
    Convierte la pregunta del usuario (normalmente en español, posiblemente
    incompleta) en una consulta de búsqueda en inglés autocontenida.

    Si la llamada falla se devuelve la pregunta original: buscar en español da
    peor recuperación, pero es mejor que no responder.
    """
    contexto = formatear_para_prompt(historial or [])
    entrada = (
        f"Conversation so far:\n{contexto}\n\nNew question:\n{pregunta}"
        if contexto
        else f"Question:\n{pregunta}"
    )
    try:
        salida = llm.invoke([
            {"role": "system", "content": REFORMULATE_PROMPT},
            {"role": "user", "content": entrada},
        ]).content.strip()
        return salida or pregunta
    except Exception as exc:
        logger.warning("reformulación fallida (%s), se busca con la pregunta original", exc)
        return pregunta


# ------------------------------------------------------------------
# Generación de la respuesta
# ------------------------------------------------------------------
# El contexto llega en inglés (el corpus lo está) pero la respuesta debe salir
# en español, que es el idioma del usuario.
#
# El párrafo sobre la primera persona no es decorativo. Con el prompt corto
# original ("Answer ONLY using provided context..."), gpt-4o-mini interpretaba
# "what causes MY headaches?" como un pedido de diagnóstico individual —que
# efectivamente no está en el contexto— y se abstenía, aunque el contexto
# contuviera las causas generales. Como la gente pregunta así de forma natural
# ("¿por qué me duelen las articulaciones?"), eso rompía buena parte de las
# consultas reales.
SYSTEM_PROMPT = (
    "Eres un asistente médico informativo. Respondes SIEMPRE en español.\n"
    "Usa ÚNICAMENTE la información del contexto proporcionado, que está en "
    "inglés: tradúcela al español al responder. No uses conocimiento externo "
    "ni agregues datos que no estén en el contexto.\n"
    "Las preguntas en primera persona (\"me duele la cabeza\", \"tengo "
    "migrañas\") piden la información médica general del contexto, no un "
    "diagnóstico personal: respóndelas con esa información general.\n"
    "Si el contexto no contiene la respuesta, responde exactamente: No lo sé."
)


def answer_question(question: str, historial=None):
    """
    `question` debe venir ya libre de PII (ver guardrails.strip_pii).

    `historial` es la lista de mensajes previos de la sesión (ver app.memory).
    Se usa para dos cosas: resolver referencias al reformular la búsqueda, y
    darle continuidad conversacional al modelo que redacta la respuesta.

    Devuelve (respuesta, docs, consulta_busqueda). `docs` puede tener menos de
    TOP_K elementos: tanto el corte relativo como el reranking descartan lo que
    no aporta, en vez de rellenar hasta completar k. `consulta_busqueda` se
    devuelve para poder depurar y testear la traducción.
    """
    # La búsqueda va en inglés (el corpus lo está); la generación responde en
    # español usando la pregunta original del usuario.
    consulta = reformular(question, historial)

    k = RERANK_CANDIDATES if RERANK_ENABLED else TOP_K
    scored = vector_store.similarity_search_with_relevance_scores(consulta, k=k)

    # Corte relativo primero: acota la entrada del reranker (ver arriba por qué
    # sigue acá aunque no sea lo que decide la calidad).
    if scored:
        best = scored[0][1]
        scored = [(doc, s) for doc, s in scored if s >= RELATIVE_CUTOFF * best]

    # El reranker recibe la consulta en inglés, no la pregunta en español: opera
    # sobre chunks en inglés y compararlos contra texto español lo empeoraría.
    if RERANK_ENABLED:
        docs = _rerank(consulta, scored)
    else:
        docs = [doc for doc, _ in scored[:TOP_K]]

    # Sin contexto no se llama al LLM. El system prompt le pide abstenerse,
    # pero no es una garantía: con el contexto vacío gpt-4o-mini respondía de
    # memoria (se detectó con "What is Caffey disease?", que contestó correcto
    # y sin ninguna fuente). En un asistente médico, una respuesta sin
    # respaldo verificable es peor que no responder — así que la abstención se
    # fuerza en código, no se delega al modelo.
    if not docs:
        return "No lo sé.", [], consulta

    # El historial va como turnos reales de conversación, no embutido en el
    # texto del usuario: así el modelo mantiene la coherencia del diálogo
    # ("¿y en niños?") sin confundir el historial con el contexto recuperado.
    context = "\n\n".join(d.page_content for d in docs)
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    mensajes += historial or []
    mensajes.append(
        {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta:\n{question}"}
    )

    response = llm.invoke(mensajes)
    return response.content, docs, consulta
