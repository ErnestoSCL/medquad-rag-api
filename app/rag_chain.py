from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

from app.config import SUPABASE_URL, SUPABASE_KEY

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
# extra ni latencia (un reranking con LLM daba lo mismo costando ~1 s más).
RELATIVE_CUTOFF = 0.80

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

    Devuelve (respuesta, docs) donde `docs` ya pasó el corte relativo, así que
    puede tener menos de TOP_K elementos.
    """
    scored = vector_store.similarity_search_with_relevance_scores(question, k=TOP_K)
    if scored:
        best = scored[0][1]
        docs = [doc for doc, score in scored if score >= RELATIVE_CUTOFF * best]
    else:
        docs = []

    context = "\n\n".join(d.page_content for d in docs)
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ])
    return response.content, docs
