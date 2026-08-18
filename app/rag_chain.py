import json
import logging

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

from app.config import SUPABASE_URL, SUPABASE_KEY

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
# El ruido aparece cuando el tema tiene pocos chunks en el corpus y pedir k=5
# obliga a rellenar: "Caffey disease" tiene 1 solo chunk, así que los otros 4
# eran relleno que además se mostraba al usuario como fuentes citadas.
#
# Medido sobre 12 consultas (60 chunks): descarta 11 de 12 irrelevantes sin
# perder ninguno de los 48 relevantes. Precisión 90% -> 98%, sin llamadas
# extra ni latencia. Se aplica antes del reranking, para no gastarle tokens en
# candidatos que ya se sabe que están lejos.
RELATIVE_CUTOFF = 0.80

# Reranking con LLM: se recuperan RERANK_CANDIDATES por similitud y el modelo
# elige cuáles pasan al contexto, hasta TOP_K.
#
# Medición honesta de su aporte: sobre 12 consultas, solo 1 mejoraba al ampliar
# el pool de 5 a 20 candidatos (+1 chunk). Donde el corpus tiene material de
# sobra el top-5 ya venía completo, y donde el top-5 tenía huecos era porque no
# existen más chunks del tema.
#
# Se mantiene activo porque aporta algo que el corte relativo no puede dar:
# juzga relevancia semántica, no solo distancia vectorial — afina las citas
# (Bell's palsy pasa de 5 fuentes a 3, todas del tema). Costo real medido:
# +0.4 s por consulta (1.8 s -> 2.2 s). Para desactivarlo basta con
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

RERANK_PROMPT = (
    "You are ranking retrieved passages for a medical question.\n"
    "Select ONLY the passages that contain information useful to answer it, "
    "ordered most useful first, at most {top_k}.\n"
    "Be strict: it is better to return two good passages than five padded with "
    "loosely related ones. If none are useful, return an empty list.\n"
    "Reply with ONLY a JSON array of passage numbers, e.g. [3, 1, 7]."
)


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
        ]).content.strip()

        # El modelo a veces envuelve el JSON en ```json ... ```
        if respuesta.startswith("```"):
            respuesta = respuesta.strip("`").removeprefix("json").strip()

        elegidos = json.loads(respuesta)
        if not isinstance(elegidos, list):
            raise ValueError(f"se esperaba una lista, llegó {type(elegidos).__name__}")

        docs = [
            scored[n - 1][0]
            for n in elegidos
            if isinstance(n, int) and 1 <= n <= len(scored)
        ]
        return docs[:TOP_K]

    except Exception as exc:
        logger.warning("reranking fallido (%s), se usa el orden por similitud", exc)
        return [doc for doc, _ in scored[:TOP_K]]


# El párrafo sobre la primera persona no es decorativo. Con el prompt corto
# original ("Answer ONLY using provided context..."), gpt-4o-mini interpretaba
# "what causes MY headaches?" como un pedido de diagnóstico individual —que
# efectivamente no está en el contexto— y se abstenía, aunque el contexto
# contuviera las causas generales. Como la gente pregunta así de forma natural
# ("why do I have joint pain?", "what can I take for my migraines?"), eso
# rompía buena parte de las consultas reales.
# Verificado: con este prompt las preguntas en primera persona se responden, y
# las que están fuera del corpus se siguen rechazando igual que antes.
SYSTEM_PROMPT = (
    "Answer ONLY using the provided context. Do not use outside knowledge.\n"
    "Questions written in the first person (\"my headaches\", \"I have...\") are "
    "requests for the general medical information in the context, not for a "
    "personal diagnosis. Answer them with that general information.\n"
    "If the context does not contain the answer, say 'I don't know'."
)


def answer_question(question: str):
    """
    `question` debe venir ya libre de PII (ver guardrails.strip_pii): el mismo
    texto se usa para buscar y para generar.

    Devuelve (respuesta, docs). `docs` puede tener menos de TOP_K elementos:
    tanto el corte relativo como el reranking descartan lo que no aporta, en vez
    de rellenar hasta completar k.
    """
    k = RERANK_CANDIDATES if RERANK_ENABLED else TOP_K
    scored = vector_store.similarity_search_with_relevance_scores(question, k=k)

    # Corte relativo primero: descarta lo que está claramente lejos del mejor
    # resultado, para no gastar tokens del reranker en candidatos malos.
    if scored:
        best = scored[0][1]
        scored = [(doc, s) for doc, s in scored if s >= RELATIVE_CUTOFF * best]

    if RERANK_ENABLED:
        docs = _rerank(question, scored)
    else:
        docs = [doc for doc, _ in scored[:TOP_K]]

    # Sin contexto no se llama al LLM. El system prompt le pide abstenerse,
    # pero no es una garantía: con el contexto vacío gpt-4o-mini respondía de
    # memoria (se detectó con "What is Caffey disease?", que contestó correcto
    # y sin ninguna fuente). En un asistente médico, una respuesta sin
    # respaldo verificable es peor que no responder — así que la abstención se
    # fuerza en código, no se delega al modelo.
    if not docs:
        return "I don't know.", []

    context = "\n\n".join(d.page_content for d in docs)
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ])
    return response.content, docs
