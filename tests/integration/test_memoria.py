"""Memoria conversacional: persistencia, ventana, aislamiento y repreguntas."""
import pytest

pytestmark = pytest.mark.integration


def test_repregunta_se_resuelve_con_el_historial(preguntar, sesion_limpia):
    """
    El caso central: "¿y cómo se trata?" no tiene sentido sola. El reformulador
    debe resolver la referencia usando el turno anterior.
    """
    preguntar("¿Qué es la parálisis de Bell?", sesion_limpia)
    r = preguntar("¿y cómo se trata?", sesion_limpia)

    assert "bell" in r["search_query"].lower(), (
        f"la repregunta debía resolverse a Bell's palsy, salió: {r['search_query']}"
    )
    assert "No hay información suficiente" not in r["answer"]


def test_sin_historial_la_repregunta_queda_ambigua(preguntar, sesion_limpia):
    """Control del test anterior: sin contexto previo no hay nada que resolver."""
    r = preguntar("¿y cómo se trata?", sesion_limpia)
    assert "bell" not in r["search_query"].lower()


def test_la_interaccion_se_persiste(preguntar, sesion_limpia):
    from app.memory import cargar_historial

    preguntar("¿Qué es la diabetes?", sesion_limpia)
    historial = cargar_historial(sesion_limpia)

    assert len(historial) == 2
    assert historial[0]["role"] == "user"
    assert historial[1]["role"] == "assistant"
    assert "diabetes" in historial[0]["content"].lower()


def test_el_historial_no_guarda_pii(preguntar, sesion_limpia):
    """
    El historial vive en la base: no debe contener datos personales, coherente
    con el guardrail que los elimina.
    """
    from app.memory import cargar_historial

    preguntar("Soy José Martínez, ¿qué es el asma?", sesion_limpia)
    historial = cargar_historial(sesion_limpia)

    guardado = " ".join(m["content"] for m in historial)
    assert "José" not in guardado
    assert "Martínez" not in guardado


def test_sesiones_distintas_no_comparten_historial(preguntar, sesion_limpia):
    import uuid

    from app.memory import borrar_historial, cargar_historial

    otra = f"test-{uuid.uuid4()}"
    try:
        preguntar("¿Qué es la parálisis de Bell?", sesion_limpia)
        assert cargar_historial(otra) == []

        r = preguntar("¿y cómo se trata?", otra)
        assert "bell" not in r["search_query"].lower(), (
            "una sesión no puede ver el contexto de otra"
        )
    finally:
        borrar_historial(otra)


def test_ventana_de_tres_turnos(sesion_limpia):
    """
    Se recuerdan las últimas 3 interacciones: 6 mensajes, y el más viejo queda
    fuera al agregar el cuarto.
    """
    from app.memory import cargar_historial, guardar_interaccion

    guardar_interaccion(sesion_limpia, "primera", "r1")
    guardar_interaccion(sesion_limpia, "segunda", "r2")
    guardar_interaccion(sesion_limpia, "tercera", "r3")
    guardar_interaccion(sesion_limpia, "cuarta", "r4")

    historial = cargar_historial(sesion_limpia, turnos=3)
    contenidos = [m["content"] for m in historial]

    assert len(historial) == 6
    assert "primera" not in contenidos, "la interacción más vieja debe caer"
    assert "cuarta" in contenidos
    assert contenidos.index("segunda") < contenidos.index("cuarta"), "orden cronológico"


def test_session_id_vacio_no_rompe():
    from app.memory import cargar_historial, guardar_interaccion

    assert cargar_historial("") == []
    assert guardar_interaccion("", "p", "r") is False
