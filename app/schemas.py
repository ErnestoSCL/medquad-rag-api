from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    # Identificador de conversación. Si no viene, la consulta se responde sin
    # memoria: no se carga historial previo ni se guarda esta interacción.
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    # Consulta en inglés con la que realmente se buscó, tras traducir y
    # resolver las referencias al historial. Se expone para poder depurar por
    # qué se recuperó lo que se recuperó.
    search_query: str | None = None
