from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import google.generativeai as genai
import os

app = FastAPI()

# CORS (necesario para frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Variables globales (NO cargar al inicio)
embedder = None
collection = None


def init_db():
    global embedder, collection

    if embedder is None:
        print("Inicializando modelo y base de datos...")

        from sentence_transformers import SentenceTransformer
        import chromadb

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path="./db")
        collection = client.get_or_create_collection(name="materias")

        print("Modelo y DB cargados")


# Materias
MATERIAS = {
    "1": "Algoritmos y Estructura de Datos",
    "2": "Base de Datos",
    "algoritmos": "Algoritmos y Estructura de Datos",
    "algoritmos y estructura de datos": "Algoritmos y Estructura de Datos",
    "aed": "Algoritmos y Estructura de Datos",
    "base de datos": "Base de Datos",
    "bd": "Base de Datos"
}


def detectar_materia(texto: str):
    return MATERIAS.get(texto.strip().lower())


def mensaje_bienvenida_materia(nombre_materia: str):
    return {
        "respuesta": (
            f"Elegiste: {nombre_materia}.\n\n"
            "Podés consultarme sobre:\n"
            "- Plan de estudios\n"
            "- Horarios\n"
            "- Contenidos\n"
            "- Correlativas\n"
            "- Modalidad de cursado\n"
            "- Programa de la materia"
        )
    }


# Sirve la web
@app.get("/")
def home():
    return FileResponse("index.html")


# Endpoint menú
@app.get("/menu")
def menu():
    return {
        "mensaje": (
            "Hola, soy el asistente virtual de las siguientes asignaturas:\n\n"
            "1. Algoritmos y Estructura de Datos\n"
            "2. Base de Datos\n\n"
            "Podés elegir escribiendo el número o el nombre de la materia."
        )
    }


# Endpoint chat
@app.get("/chat")
def chat(q: str):

    # Inicializar DB SOLO cuando se usa
    init_db()

    # Detectar materia
    materia = detectar_materia(q)
    if materia:
        return mensaje_bienvenida_materia(materia)

    # Buscar en base vectorial
    query_embedding = embedder.encode([q]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documentos = results.get("documents", [[]])
    contextos = documentos[0] if documentos and len(documentos) > 0 else []

    context = "\n\n".join(contextos)

    # Prompt
    prompt = f"""
Sos un asistente académico.

Respondé solamente usando la información del contexto.
Si la respuesta no está en el contexto, decí:
"No encontré esa información en los documentos cargados."

Contexto:
{context}

Pregunta:
{q}
"""

    response = model.generate_content(prompt)

    return {"respuesta": response.text}
