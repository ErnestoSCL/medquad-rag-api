"""Detección de saludos y meta-preguntas.

El riesgo de esta capa no es dejar pasar un saludo, sino lo contrario:
interceptar una consulta médica real y devolver la bienvenida en lugar de
buscarla. Por eso hay más casos negativos que positivos.
"""
import pytest

from app.conversacion import BIENVENIDA, es_conversacional, respuesta_conversacional


@pytest.mark.parametrize("mensaje", [
    "hola",
    "Hola!",
    "buenas",
    "buenos días",
    "qué tal",
    "hey",
    "gracias",
    "hola, ¿cómo estás?",
    "",
    "   ",
])
def test_saludos_se_interceptan(mensaje):
    assert es_conversacional(mensaje)


@pytest.mark.parametrize("mensaje", [
    "hola en que me pedes ayudar",
    "¿en qué me podés ayudar?",
    "¿qué puedes hacer?",
    "¿cómo funcionas?",
    "¿quién sos?",
    "¿para qué sirves?",
    "necesito ayuda",
    "¿qué tipo de preguntas puedo hacer?",
    "¿de dónde sacas la información?",
    "what can you do?",
])
def test_meta_preguntas_se_interceptan(mensaje):
    assert es_conversacional(mensaje)


@pytest.mark.parametrize("mensaje", [
    "¿Cuáles son los síntomas de la parálisis de Bell?",
    "¿Cómo se trata el asma?",
    "¿Qué causa los dolores de cabeza?",
    "me duele la cabeza",
    "tengo migrañas desde hace una semana",
    "¿Qué es la diabetes?",
    "¿y cómo se trata?",
    "What are the symptoms of Bell's palsy?",
])
def test_consultas_medicas_no_se_interceptan(mensaje):
    assert not es_conversacional(mensaje), "debería ir al RAG"


@pytest.mark.parametrize("mensaje", [
    "hola, ¿qué es la diabetes?",
    "buenas, me duele la cabeza hace días",
    "gracias, ¿y cuáles son las causas del asma?",
])
def test_saludo_con_consulta_va_al_rag(mensaje):
    """
    El caso que más importa: si al quitar el saludo queda una consulta real,
    hay que responderla en vez de devolver la bienvenida.
    """
    assert not es_conversacional(mensaje)


def test_la_bienvenida_explica_y_da_ejemplos():
    r = respuesta_conversacional()
    assert r == BIENVENIDA
    assert "MedQuAD" in r
    assert "síntomas" in r
    assert "-" in r, "debería listar ejemplos"
