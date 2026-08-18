"""Configuración compartida de la suite.

Separación deliberada entre unit e integration: los unitarios corren sin red ni
servidor (solo importan `app.guardrails`, que no depende de nada externo), y
los de integración hablan por HTTP con el contenedor. Así se puede validar la
lógica de los guardrails sin instalar langchain ni gradio.
"""
import os
import sys

import pytest

# Permite `from app.guardrails import ...` sin instalar el paquete
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

BASE_URL = os.environ.get("MEDQUAD_URL", "http://localhost:7860")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requiere el servidor levantado y credenciales"
    )


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def servidor_activo(base_url):
    """
    Salta los tests de integración si el servidor no responde, en vez de
    fallarlos: no tenerlo levantado no significa que el código esté roto.
    """
    import httpx

    try:
        httpx.get(f"{base_url}/docs", timeout=5)
    except Exception as exc:
        pytest.skip(f"servidor no disponible en {base_url}: {exc}")
    return base_url


@pytest.fixture
def preguntar(servidor_activo):
    """Hace POST /ask y devuelve el JSON. Timeout amplio: hay 2-3 llamadas al LLM."""
    import httpx

    def _preguntar(pregunta, session_id=None, timeout=180):
        payload = {"question": pregunta}
        if session_id:
            payload["session_id"] = session_id
        r = httpx.post(f"{servidor_activo}/ask", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    return _preguntar


@pytest.fixture
def sesion_limpia():
    """
    Session id único por test, con borrado al terminar para no dejar basura en
    la tabla compartida de Supabase.
    """
    import uuid

    sid = f"test-{uuid.uuid4()}"
    yield sid
    try:
        from app.memory import borrar_historial

        borrar_historial(sid)
    except Exception:
        pass  # la limpieza es best-effort; no debe hacer fallar el test
