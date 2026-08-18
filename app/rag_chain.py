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
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

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
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ])
    return response.content, docs
