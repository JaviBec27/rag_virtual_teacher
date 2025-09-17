# src/ingestion/ingest_documents.py

import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def process_and_store_pdf_in_vectordb(
    pdf_path: str, FOLDER_INDEX_base: str, api_key: str
):
    """
    Lee un único documento PDF, crea su base de datos vectorial (FAISS)
    y la guarda en disco.
    """
    if not api_key:
        print("Proceso detenido. No se pudo cargar la API key de Google.")
        return False

    file_name = os.path.basename(pdf_path)
    base_name, _ = os.path.splitext(file_name)
    output_path = os.path.join(FOLDER_INDEX_base, f"faiss_index_{base_name}")

    print("-" * 50)
    print(f"Iniciando procesamiento para: '{file_name}'")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            raise ValueError("El PDF está vacío o no se pudo leer.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_key
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(output_path)
        print(f"¡Éxito! Índice guardado en: '{output_path}'")

        return True

    except Exception as e:
        print(f"ERROR al procesar el archivo {os.path.basename(pdf_path)}: {e}")
        return False
