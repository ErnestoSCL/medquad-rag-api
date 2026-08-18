"""El hueco del analisis anterior: hay chunks relevantes en las posiciones
6-20 que hoy se pierden por usar k=5? Si los hay, el reranking (k=20 -> 5)
aporta algo que el corte relativo NO puede dar."""
import os, json
from openai import OpenAI
from supabase import create_client

oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

CASOS = [
    ("What are the symptoms of Bell's palsy?",        "bell"),
    ("What is the treatment for migraine?",           "migrain"),
    ("How is asthma treated?",                        "asthma"),
    ("What is diabetes?",                             "diabet"),
    ("What are the symptoms of Parkinson's disease?", "parkinson"),
    ("What is Wilson disease?",                       "wilson"),
    ("What is WAGR syndrome?",                        "wagr"),
    ("What is colpocephaly?",                         "colpocephaly"),
    ("What causes headaches?",                        "headache"),
    ("What is cystic fibrosis?",                      "cystic"),
    ("What is triploidy?",                            "triploidy"),
    ("What is Caffey disease?",                       "caffey"),
]

print(f"{'pregunta':44} {'rel en 1-5':>11} {'rel en 6-20':>12}   ganancia potencial")
print("=" * 92)

tot_top5 = tot_cola = 0
casos_con_ganancia = 0

for pregunta, clave in CASOS:
    emb = oa.embeddings.create(model="text-embedding-3-small",
                               input=pregunta, dimensions=512).data[0].embedding
    r = sb.rpc("match_documents", {"query_embedding": emb, "match_count": 20}).execute()

    rel_flags = []
    for d in r.data:
        m = d["metadata"] if isinstance(d["metadata"], dict) else json.loads(d["metadata"])
        rel_flags.append(clave in (m.get("question_focus") or "").lower())

    en_top5 = sum(rel_flags[:5])
    en_cola = sum(rel_flags[5:20])
    tot_top5 += en_top5
    tot_cola += en_cola

    # Solo hay ganancia si el top-5 tiene huecos QUE la cola podria llenar
    huecos = 5 - en_top5
    ganancia = min(huecos, en_cola)
    if ganancia:
        casos_con_ganancia += 1
    marca = f"+{ganancia} chunk(s)" if ganancia else "-"
    print(f"  {pregunta[:42]:44} {en_top5:>7}/5 {en_cola:>10}/15   {marca}")

print(f"\n{'='*92}")
print(f"relevantes en top-5      : {tot_top5}")
print(f"relevantes en 6-20       : {tot_cola}")
print(f"consultas donde reranking podria sumar: {casos_con_ganancia}/{len(CASOS)}")
