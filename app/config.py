import os

from dotenv import load_dotenv

# Carga el .env para desarrollo local. En producción no existe ese archivo y la
# llamada no hace nada: Render y Docker inyectan las variables en el entorno,
# que es de donde se leen abajo.
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
