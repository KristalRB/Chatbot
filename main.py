from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MATERIAS = {
    "1": "Algoritmos y Estructura de Datos",
    "2": "Base de Datos",
}

@app.get("/")
def home():
    return FileResponse("index.html")

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

@app.get("/chat")
def chat(q: str):
    if q.strip() in MATERIAS:
        materia = MATERIAS[q.strip()]
        return {
            "respuesta": f"Elegiste {materia}. Podés preguntarme sobre contenidos, horarios o plan de estudios."
        }

    return {"respuesta": f"Recibí tu mensaje: {q}"}
