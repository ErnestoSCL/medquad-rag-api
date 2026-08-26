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


@pytest.mark.parametrize("mensaje", [
    "asma",
    "diabetes",
    "anemia",
    "migraña",
])
def test_una_sola_palabra_es_consulta(mensaje):
    """
    Regresión: el umbral de longitud mínima se aplicaba aunque no hubiera
    ningún saludo que recortar, así que escribir solo "asma" devolvía el
    mensaje de bienvenida en vez de información sobre el asma.
    """
    assert not es_conversacional(mensaje), "una palabra suelta es una consulta"


def test_consulta_larga_con_saludo_no_dispara_por_subcadena():
    """
    Regresión: el patrón `qué haces?` matcheaba dentro de "porque hace unos
    días" y una consulta médica extensa recibía la bienvenida. Los patrones
    ahora exigen límite de palabra.
    """
    mensaje = (
        "hola, quería consultar algo porque hace unos días vengo notando que "
        "me cuesta respirar cuando hago ejercicio y a veces me silba el pecho, "
        "sobre todo de noche, ¿qué puede ser?"
    )
    assert not es_conversacional(mensaje)


@pytest.mark.parametrize("mensaje", [
    "me duele la cabeza porque hace mucho calor",
    "tengo tos desde hace una semana",
    "no sé qué hacer con mi alergia",
])
def test_texto_corriente_no_se_confunde_con_meta_pregunta(mensaje):
    """Palabras como "hace" o "qué" en medio de una frase no son meta-preguntas."""
    assert not es_conversacional(mensaje)


def test_la_bienvenida_explica_y_da_ejemplos():
    r = respuesta_conversacional()
    assert r == BIENVENIDA
    assert "MedQuAD" in r
    assert "síntomas" in r
    assert "-" in r, "debería listar ejemplos"
