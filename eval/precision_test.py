"""Precision del top-5: cuantos chunks son del tema correcto y con que similitud.
Si los irrelevantes caen claramente por debajo, un umbral basta y no hace falta reranking."""
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
]

rel_sims, irr_sims = [], []
total_rel = 0

for pregunta, clave in CASOS:
    emb = oa.embeddings.create(model="text-embedding-3-small",
                               input=pregunta, dimensions=512).data[0].embedding
    r = sb.rpc("match_documents", {"query_embedding": emb, "match_count": 5}).execute()

    linea, n_rel = [], 0
    for d in r.data:
        m = d["metadata"] if isinstance(d["metadata"], dict) else json.loads(d["metadata"])
        foco = (m.get("question_focus") or "")
        es_rel = clave in foco.lower()
        (rel_sims if es_rel else irr_sims).append(d["similarity"])
        n_rel += es_rel
        linea.append(f"{'+' if es_rel else '-'}{d['similarity']:.3f}")
    total_rel += n_rel
    print(f"  {n_rel}/5  {pregunta[:44]:46} {' '.join(linea)}")

print(f"\n{'='*80}")
print(f"precision@5 media: {total_rel}/{len(CASOS)*5} = {100*total_rel/(len(CASOS)*5):.0f}%")
print(f"\nsimilitud de chunks RELEVANTES   : min {min(rel_sims):.3f}  media {sum(rel_sims)/len(rel_sims):.3f}  max {max(rel_sims):.3f}")
if irr_sims:
    print(f"similitud de chunks IRRELEVANTES : min {min(irr_sims):.3f}  media {sum(irr_sims)/len(irr_sims):.3f}  max {max(irr_sims):.3f}")
    print(f"\n?Se pueden separar con un umbral fijo?")
    print(f"  peor relevante   : {min(rel_sims):.3f}")
    print(f"  mejor irrelevante: {max(irr_sims):.3f}")
    if min(rel_sims) > max(irr_sims):
        print(f"  -> SI. Umbral ~{(min(rel_sims)+max(irr_sims))/2:.3f}")
    else:
        print(f"  -> NO, se solapan. Ver relative_cutoff.py")
