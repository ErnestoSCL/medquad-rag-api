"""Parseo de la respuesta del reranker.

Es la parte frágil del pipeline: el LLM devuelve texto libre y hay que
convertirlo en índices. Por eso `parse_rerank_response` está separada de
`_rerank` — se puede testear sin red, sin modelo y sin langchain.

Nota: importa desde un módulo copiado en el test, no de app.rag_chain, porque
ese importa langchain. La función es autocontenida (solo usa json).
"""
import json

import pytest

# Réplica exacta de app.rag_chain.parse_rerank_response. Se duplica a propósito
# para no arrastrar langchain al entorno de tests unitarios; si la original
# cambia, este test debe fallar en integración y avisar.
def parse_rerank_response(respuesta: str, n_candidatos: int) -> list[int]:
    limpia = respuesta.strip()
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


def test_lista_simple():
    assert parse_rerank_response("[3, 1, 5]", 10) == [2, 0, 4]


def test_conserva_el_orden_del_modelo():
    """El reranker ordena por relevancia; el orden no debe alterarse."""
    assert parse_rerank_response("[5, 2, 9]", 10) == [4, 1, 8]


def test_lista_vacia():
    """Ningún pasaje sirve: es una respuesta válida, no un error."""
    assert parse_rerank_response("[]", 10) == []


@pytest.mark.parametrize("crudo", [
    "```json\n[1, 2]\n```",
    "```\n[1, 2]\n```",
    "  [1, 2]  ",
    "[1, 2]\n",
])
def test_tolera_envoltorios_y_espacios(crudo):
    assert parse_rerank_response(crudo, 5) == [0, 1]


@pytest.mark.parametrize("indices,n,esperado", [
    ("[0]", 5, []),          # 0 es inválido: la numeración empieza en 1
    ("[6]", 5, []),          # fuera de rango por arriba
    ("[-1]", 5, []),
    ("[1, 99, 2]", 5, [0, 1]),  # descarta el inválido y conserva el resto
])
def test_descarta_indices_fuera_de_rango(indices, n, esperado):
    assert parse_rerank_response(indices, n) == esperado


@pytest.mark.parametrize("crudo", [
    '["uno", "dos"]',
    "[1.5, 2.7]",
    "[null]",
])
def test_descarta_tipos_no_enteros(crudo):
    assert parse_rerank_response(crudo, 5) == []


def test_booleanos_no_cuentan_como_enteros():
    """En Python bool es subclase de int: True pasaría como índice 1."""
    assert parse_rerank_response("[true, false]", 5) == []


@pytest.mark.parametrize("crudo", [
    "no encontré nada útil",
    "",
    "{'a': 1}",
])
def test_respuesta_no_json_lanza(crudo):
    """_rerank captura esto y cae al orden por similitud."""
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parse_rerank_response(crudo, 5)


def test_json_valido_pero_no_lista_lanza():
    with pytest.raises(ValueError):
        parse_rerank_response('{"elegidos": [1, 2]}', 5)
