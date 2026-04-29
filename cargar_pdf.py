import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./db")

collection = client.get_or_create_collection(name="materias")

folder = "pdfs"

def detectar_materia_desde_nombre(filename: str):
    nombre = filename.lower()

    if "algoritmos" in nombre or "estructura de datos" in nombre or "aed" in nombre:
        return "Algoritmos y Estructura de Datos"

    if "base de datos" in nombre or "bd" in nombre:
        return "Base de Datos"

    return "General"

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        path = os.path.join(folder, filename)
        print(f"Procesando {filename}...")

        reader = PdfReader(path)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

        if not chunks:
            print(f"No se pudo extraer texto de {filename}")
            continue

        embeddings = model.encode(chunks).tolist()
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        materia = detectar_materia_desde_nombre(filename)

        metadatas = [
            {
                "source": filename,
                "materia": materia
            }
            for _ in chunks
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

print("Todos los PDFs procesados y guardados en ./db ✅")