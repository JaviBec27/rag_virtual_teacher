# app.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.retrieval.rag_agent import create_rag_chain

# Carga la API key desde .env al principio del script
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("No se encontró la GOOGLE_API_KEY en las variables de entorno.")

app = FastAPI()

class ChatRequest(BaseModel):
    user_message: str
    knowledge_domain: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para interactuar con el agente RAG.
    La solicitud debe incluir el mensaje del usuario y el dominio de conocimiento.
    """
    try:
        # 1. Crea la cadena RAG dinámicamente de forma asíncrona
        rag_chain = await create_rag_chain(
            api_key=GOOGLE_API_KEY,
            knowledge_domain=request.knowledge_domain
        )
        
        # 2. Invoca la cadena de forma asíncrona con el mensaje del usuario
        response = rag_chain.invoke(request.user_message)
        
        return {"response": response}
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error inesperado: {e}")