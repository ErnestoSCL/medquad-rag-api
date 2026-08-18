"""LLM-as-judge sobre el CONTENIDO de los chunks recuperados.

Reemplaza la metrica de `precision_test.py`, que solo comparaba la clave contra
`question_focus` y por lo tanto media coincidencia de etiquetas, no utilidad.

Compara cuatro configuraciones para responder dos preguntas:
  - El corte relativo aporta por si solo?           -> A vs B
  - Sigue aportando cuando hay reranker, o estorba? -> C vs D

  A  similitud sola (k=5)
  B  similitud + corte relativo
  C  similitud (k=20) + reranker
  D  similitud (k=20) + corte relativo + reranker   <- la actual

Guarda el detalle crudo en judge_output.json para poder revisar casos a mano y
calibrar si el juez esta acertando.
"""
import json
import os
import time

from openai import OpenAI
from supabase import create_client

oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

TOP_K = 5
RERANK_CANDIDATES = 20
RELATIVE_CUTOFF = 0.80
MODEL = "gpt-4o-mini"

PREGUNTAS = [
    "What are the symptoms of Bell's palsy?",
    "What is the treatment for migraine?",
    "How is asthma treated?",
    "What is diabetes?",
    "What are the symptoms of Parkinson's disease?",
    "What is Wilson disease?",
    "What is WAGR syndrome?",
    "What is colpocephaly?",
    "What causes headaches?",
    "What is cystic fibrosis?",
    "What is triploidy?",
    "What is Caffey disease?",
    "What is Hermansky-Pudlak syndrome?",
    "How is Peyronie's disease treated?",
    "What causes copper to build up in the liver and brain?",
]

# --------------------------------------------------------------------------
# Recuperacion
# --------------------------------------------------------------------------

def buscar(pregunta, k):
    emb = oa.embeddings.create(model="text-embedding-3-small",
                               input=pregunta, dimensions=512).data[0].embedding
    r = sb.rpc("match_documents", {"query_embedding": emb, "match_count": k}).execute()
    salida = []
    for d in r.data:
        m = d["metadata"] if isinstance(d["metadata"], dict) else json.loads(d["metadata"])
        salida.append({"content": d["content"], "focus": m.get("question_focus"),
                       "sim": d["similarity"]})
    return salida


def corte_relativo(chunks):
    if not chunks:
        return []
    tope = chunks[0]["sim"]
    return [c for c in chunks if c["sim"] >= RELATIVE_CUTOFF * tope]


RERANK_PROMPT = (
    "You are ranking retrieved passages for a medical question.\n"
    "Select ONLY the passages that contain information useful to answer it, "
    "ordered most useful first, at most {top_k}.\n"
    "Be strict: it is better to return two good passages than five padded with "
    "loosely related ones. If none are useful, return an empty list.\n"
    "Reply with ONLY a JSON array of passage numbers, e.g. [3, 1, 7]."
)


def rerank(pregunta, chunks):
    if not chunks:
        return []
    pasajes = "\n\n".join(f"[{i}] {c['content'][:1200]}" for i, c in enumerate(chunks, 1))
    try:
        txt = oa.chat.completions.create(
            model=MODEL, temperature=0,
            messages=[
                {"role": "system", "content": RERANK_PROMPT.format(top_k=TOP_K)},
                {"role": "user", "content": f"Question:\n{pregunta}\n\nPassages:\n{pasajes}"},
            ],
        ).choices[0].message.content.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").removeprefix("json").strip()
        idx = json.loads(txt)
        return [chunks[n - 1] for n in idx
                if isinstance(n, int) and 1 <= n <= len(chunks)][:TOP_K]
    except Exception as e:
        print(f"    [rerank fallo: {e}]")
        return chunks[:TOP_K]


CONFIGS = {
    "A_base":            lambda q: buscar(q, TOP_K),
    "B_corte":           lambda q: corte_relativo(buscar(q, TOP_K)),
    "C_rerank":          lambda q: rerank(q, buscar(q, RERANK_CANDIDATES)),
    "D_corte_rerank":    lambda q: rerank(q, corte_relativo(buscar(q, RERANK_CANDIDATES))),
}

# --------------------------------------------------------------------------
# Juez: lee el CONTENIDO, no la etiqueta
# --------------------------------------------------------------------------

JUEZ_PROMPT = """You are evaluating passages retrieved by a medical Q&A system.

For EACH passage, decide whether it would help a physician answer the question:

  2 = directly useful: contains information that answers part of the question
  1 = related: same condition or topic, but does not answer the question
  0 = not useful: different topic, would not help at all

Judge the passage TEXT on its own merits. A passage about a different named
condition can still score 2 if its content genuinely addresses the question.

Reply with ONLY a JSON array of integers, one per passage, in order.
Example for 3 passages: [2, 0, 1]"""


def juzgar(pregunta, chunks):
    if not chunks:
        return []
    pasajes = "\n\n".join(f"[{i}] {c['content'][:1200]}" for i, c in enumerate(chunks, 1))
    for intento in range(3):
        try:
            txt = oa.chat.completions.create(
                model=MODEL, temperature=0,
                messages=[
                    {"role": "system", "content": JUEZ_PROMPT},
                    {"role": "user", "content": f"Question:\n{pregunta}\n\nPassages:\n{pasajes}"},
                ],
            ).choices[0].message.content.strip()
            if txt.startswith("```"):
                txt = txt.strip("`").removeprefix("json").strip()
            notas = json.loads(txt)
            if isinstance(notas, list) and len(notas) == len(chunks):
                return [n for n in notas if isinstance(n, int)]
        except Exception as e:
            if intento == 2:
                print(f"    [juez fallo: {e}]")
            time.sleep(1)
    return []


# --------------------------------------------------------------------------

def main():
    crudo = []
    resumen = {k: {"utiles": 0, "relacionados": 0, "inutiles": 0, "citas": 0}
               for k in CONFIGS}

    for pregunta in PREGUNTAS:
        print(f"\n{pregunta}")
        for nombre, fn in CONFIGS.items():
            chunks = fn(pregunta)
            notas = juzgar(pregunta, chunks)
            if len(notas) != len(chunks):
                print(f"  {nombre:16} juez no devolvio notas alineadas, se omite")
                continue

            r = resumen[nombre]
            r["citas"] += len(chunks)
            r["utiles"] += sum(1 for n in notas if n == 2)
            r["relacionados"] += sum(1 for n in notas if n == 1)
            r["inutiles"] += sum(1 for n in notas if n == 0)

            print(f"  {nombre:16} {len(chunks)} citas  notas={notas}")
            crudo.append({
                "pregunta": pregunta, "config": nombre,
                "chunks": [{"focus": c["focus"], "sim": round(c["sim"], 3),
                            "nota": n, "texto": c["content"][:400]}
                           for c, n in zip(chunks, notas)],
            })

    print(f"\n\n{'='*94}")
    print(f"{'config':18} {'citas':>6} {'utiles(2)':>10} {'relac(1)':>9} {'inutiles(0)':>12} "
          f"{'% util':>8} {'% inutil':>9}")
    print("=" * 94)
    for nombre, r in resumen.items():
        t = r["citas"] or 1
        print(f"{nombre:18} {r['citas']:>6} {r['utiles']:>10} {r['relacionados']:>9} "
              f"{r['inutiles']:>12} {100*r['utiles']/t:>7.0f}% {100*r['inutiles']/t:>8.0f}%")

    print("\nB vs A  -> aporta el corte relativo por si solo?")
    print("D vs C  -> el corte sigue aportando con reranker, o estorba?")

    destino = os.path.join(os.path.dirname(__file__), "judge_output.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(crudo, f, ensure_ascii=False, indent=2)
    print(f"\ndetalle crudo -> {destino}")


if __name__ == "__main__":
    main()
