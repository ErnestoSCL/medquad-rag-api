# Evaluación del retrieval

Scripts usados para decidir, con datos, qué mejoras del pipeline valían la pena
y cuáles no. Se corren contra la base de Supabase ya poblada, con las variables
de entorno del `.env` cargadas.

```bash
python eval/judge.py                # LLM-as-judge sobre 4 configuraciones (el principal)
python eval/recall_hard.py          # terminos tecnicos exactos y raros
python eval/recall_descriptive.py   # descripciones sin el nombre tecnico
python eval/pool_test.py            # hay relevantes en las posiciones 6-20?
python eval/precision_test.py       # metrica antigua, ver advertencia abajo
python eval/relative_cutoff.py      # calibracion del corte relativo
```

## La medición que importa: `judge.py`

Compara cuatro configuraciones con un LLM que **lee el contenido** de cada chunk
y lo puntúa: 2 = útil para responder, 1 = del tema pero no responde, 0 = inútil.
15 preguntas.

| Configuración | Citas | Útiles (2) | Relacionados (1) | Inútiles (0) | % útil |
|---|---|---|---|---|---|
| solo similitud (k=5) | 75 | 38 | 16 | **21** | 51% |
| + corte relativo | 64 | 38 | 18 | **8** | 59% |
| + reranking (sin corte) | 53 | 42 | 11 | **0** | 79% |
| **+ corte + reranking** (actual) | 51 | 41 | 10 | **0** | **80%** |

**Conclusión:** el reranking es lo que elimina los chunks inútiles — pasa de 28%
a 0%. El corte relativo por sí solo baja ese ruido a la mitad, pero no lo cierra.

**Sobre el corte relativo con reranking activo:** las dos últimas filas son casi
idénticas, así que el corte no aporta calidad adicional. Se mantiene porque
acota la entrada del reranker (menos tokens, menos latencia) y porque es
determinista, a diferencia de un LLM. No porque filtre ruido que el reranker
dejaría pasar.

## Advertencia: la métrica antigua estaba equivocada

`precision_test.py` y `relative_cutoff.py` determinan la relevancia comparando
una palabra clave contra el campo `question_focus`:

```python
es_relevante = clave in question_focus.lower()
```

Eso mide **coincidencia de etiquetas, no utilidad**, y sobrestima mucho: daba
90% de precisión para la configuración base, donde el juez que lee el contenido
encuentra 28% de chunks inútiles. Casi tres veces de diferencia.

El caso que lo ilustra: un fragmento sobre el *tratamiento* de Bell's palsy
contiene la palabra "bell", así que la métrica antigua lo contaba como acierto
para la pregunta "what are the **symptoms** of Bell's palsy?". El juez lo marca
0, correctamente — es del tema, pero no responde lo que se preguntó.

Se conservan esos dos scripts porque documentan cómo se llegó al valor 0.80 del
corte, pero **sus porcentajes no deben citarse**. La medición válida es
`judge.py`.

## Decisiones

**Búsqueda híbrida (BM25 + vectorial): descartada.** Su ventaja es el término
exacto raro, y ahí `recall_hard.py` da 20/20 en posición 1. La razón es que el
embedding se calcula sobre `question + chunk_text`, lo que vuelve la búsqueda
pregunta-contra-pregunta y captura siglas y nombres propios directamente.
Agregar `tsvector` + GIN habría costado ~50 MB de base sin ganancia medible.

**Corte relativo: implementado**, como pre-filtro determinista, no como
mecanismo de calidad.

**Reranking con LLM: implementado.** Es lo que lleva los chunks inútiles a 0%.
Costo: +0.4 s por consulta.

## Un hallazgo del reranking: alucinación con contexto vacío

Al activarlo apareció un problema que no tenía que ver con el ranking. Cuando el
reranker descartaba todos los pasajes, el contexto quedaba vacío y `gpt-4o-mini`
**respondía de memoria** en lugar de abstenerse: "What is Caffey disease?"
devolvía una respuesta correcta y **sin ninguna fuente**.

El system prompt dice `If the context does not contain the answer, say 'I don't
know'`, pero eso es una instrucción, no una garantía. La abstención ahora se
fuerza en código: si no quedan documentos, `answer_question` devuelve
"I don't know" sin llegar a llamar al LLM.

## Calibración del juez

Revisando a mano los casos marcados con 0, el juez tiende a ser **más estricto
de lo debido**: penalizó listas del Human Phenotype Ontology sobre síntomas de
Parkinson (son síntomas, aunque en formato de tabla) y marcó `WAGRO syndrome`
como inútil para una pregunta sobre WAGR, siendo una variante directa.

Consecuencia: los porcentajes absolutos son una cota pesimista. La comparación
**entre** configuraciones sigue siendo válida, porque el mismo juez evalúa las
cuatro con el mismo criterio.

El detalle crudo de cada evaluación queda en `judge_output.json` para poder
revisar casos concretos.
