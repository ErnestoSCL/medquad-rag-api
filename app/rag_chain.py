from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

from app.config import SUPABASE_URL, SUPABASE_KEY

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPT = (
    "Answer ONLY using provided context. If the answer is not in the "
    "context, say 'I don't know'."
)


def answer_question(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ])
    return response.content, docs
