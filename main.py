from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import chromadb
import os
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection(name="materias")

MATERIAS = {
    "1": "Algoritmos y Estructura de Datos",
    "2": "Base de Datos",
    "algoritmos": "Algoritmos y Estructura de Datos",
    "algoritmos y estructura de datos": "Algoritmos y Estructura de Datos",
    "aed": "Algoritmos y Estructura de Datos",
    "base de datos": "Base de Datos",
    "bd": "Base de Datos"
}

def menu_inicial():
    return {
        "mensaje": (
            "Hola, soy el asistente virtual de las siguientes asignaturas:\n\n"
            "1. Algoritmos y Estructura de Datos\n"
            "2. Base de Datos\n\n"
            "Podés elegir escribiendo el número o el nombre de la materia.\n\n"
            "Ejemplos:\n"
            "- 1\n"
            "- Algoritmos y Estructura de Datos\n"
            "- 2\n"
            "- Base de Datos"
        )
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

@app.get("/")
def home():
    return menu_inicial()

@app.get("/chat")
def chat(q: str):
    materia = detectar_materia(q)
    if materia:
        return mensaje_bienvenida_materia(materia)

    query_embedding = embedder.encode([q]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documentos = results.get("documents", [[]])
    contextos = documentos[0] if documentos and len(documentos) > 0 else []
    context = "\n\n".join(contextos)

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