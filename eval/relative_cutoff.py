"""Corte RELATIVO: descartar chunks cuya similitud sea < ratio * (mejor similitud).
Gratis: sin llamadas extra, sin latencia. Comparar contra el reranking."""
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

# Recuperar una sola vez y evaluar varios ratios sobre lo mismo
datos = []
for pregunta, clave in CASOS:
    emb = oa.embeddings.create(model="text-embedding-3-small",
                               input=pregunta, dimensions=512).data[0].embedding
    r = sb.rpc("match_documents", {"query_embedding": emb, "match_count": 5}).execute()
    filas = []
    for d in r.data:
        m = d["metadata"] if isinstance(d["metadata"], dict) else json.loads(d["metadata"])
        filas.append((d["similarity"], clave in (m.get("question_focus") or "").lower()))
    datos.append((pregunta, filas))

print(f"{'ratio':>6} {'irrelevantes filtrados':>24} {'relevantes perdidos':>22}   precision resultante")
print("=" * 86)
for ratio in (0.70, 0.75, 0.80, 0.85, 0.90):
    filtrados = perdidos = quedan_rel = quedan_tot = 0
    for _, filas in datos:
        top = filas[0][0]
        for sim, es_rel in filas:
            if sim >= ratio * top:
                quedan_tot += 1
                quedan_rel += es_rel
            elif es_rel:
                perdidos += 1
            else:
                filtrados += 1
    tot_irr = sum(1 for _, f in datos for _, r_ in f if not r_)
    tot_rel = sum(1 for _, f in datos for _, r_ in f if r_)
    prec = 100 * quedan_rel / quedan_tot if quedan_tot else 0
    print(f"{ratio:>6} {filtrados:>10}/{tot_irr:<13} {perdidos:>10}/{tot_rel:<11}   {prec:.0f}%")

print("\n--- detalle con ratio 0.80 (el implementado) ---")
for pregunta, filas in datos:
    top = filas[0][0]
    marcas = []
    for sim, es_rel in filas:
        simbolo = ("+" if es_rel else "-") + f"{sim:.3f}"
        marcas.append(simbolo if sim >= 0.80 * top else f"[{simbolo}]")
    print(f"  {pregunta[:44]:46} {' '.join(marcas)}")
print("\n  (+ relevante, - irrelevante, [x] = descartado por el corte)")
