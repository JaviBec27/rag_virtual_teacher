# src/retrieval/rag_agent.py

import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- Configuración del logger ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cargar las variables de entorno al principio del script
load_dotenv()

# Ahora sí se puede leer la variable de entorno
FAISS_BASE_PATH = os.getenv("FAISS_BASE_PATH")
LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL")
TEMPERATURE = os.getenv("TEMPERATURE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


async def create_rag_chain(api_key: str, knowledge_domain: str):
    """
    Configura y devuelve la cadena RAG completa de LangChain para un dominio dado.
    """
    # Verificamos si la ruta base del FAISS está disponible
    if not FAISS_BASE_PATH:
        logger.error("La variable de entorno 'FAISS_BASE_PATH' no está definida.")
        raise ValueError("FAISS_BASE_PATH no está configurado.")

    index_path = os.path.join(FAISS_BASE_PATH, f"faiss_index_{knowledge_domain}")

    logger.info(f"Buscando el índice de conocimiento en: {index_path}")

    if not os.path.exists(index_path):
        logger.error(
            f"Índice de conocimiento no encontrado para el dominio '{knowledge_domain}'."
        )
        raise FileNotFoundError(
            f"Índice de conocimiento no encontrado: {knowledge_domain}"
        )

    try:
        logger.info("Cargando embeddings de Google Generative AI...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=api_key
        )
        logger.info(f"Cargando la base de datos vectorial FAISS desde: {index_path}")
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever()
        logger.info("Base de datos vectorial cargada y retriever creado.")

    except Exception as e:
        logger.exception("Ocurrió un error al cargar la base de datos vectorial.")
        raise RuntimeError("Error al cargar la base de datos vectorial.") from e

    llm = ChatGoogleGenerativeAI(
        model=LANGUAGE_MODEL, temperature=TEMPERATURE, google_api_key=api_key
    )
    logger.info(f"Modelo LLM '{llm.model}' inicializado.")

    system_template = """Eres un asistente virtual que se comporta como un profesor de {knowledge_domain} experto.
    Tu nombre es \"IAsistente de {knowledge_domain}\". Eres amable, didáctico y te encanta {knowledge_domain}. 
    REGLA ESTRICTA: Solo puedes responder preguntas relacionadas con {knowledge_domain} basándote en el contexto proporcionado. Si la pregunta no está relacionada con estos temas, 
    responde: \"Lo siento, mi especialidad es {knowledge_domain}. No puedo responder preguntas sobre otros temas.\""""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{user_question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chat_prompt_template = chat_prompt_template.partial(
        knowledge_domain=knowledge_domain
    )

    rag_chain = (
        {"context": retriever, "user_question": RunnablePassthrough()}
        | chat_prompt_template
        | llm
        | StrOutputParser()
    )
    logger.info("Cadena RAG construida exitosamente.")

    return rag_chain
