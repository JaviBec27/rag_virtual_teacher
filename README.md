# 📚 RAG Virtual Teacher  

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green)](https://www.langchain.com/)  
[![Status](https://img.shields.io/badge/status-experimental-orange)]()  

> **Asistente virtual educativo basado en RAG.**  
> Responde únicamente con información de documentos cargados por los docentes.  
> Proyecto modular y experimental, independiente del LLM o vector store usado.  

---

## 🎯 Objetivo  

Construir un **asistente virtual para profesores** que:  
- Integre técnicas de **Retrieval-Augmented Generation (RAG)**.  
- Permita **consultar únicamente la literatura y documentos proporcionados por docentes**.  
- Sea **modular y flexible**, para experimentar con distintos LLMs, vector stores y frameworks.  

---

## 🛠️ Tecnologías (fase actual)  

- **Framework**: [LangChain](https://www.langchain.com/)  
- **Vector Store**: FAISS (con posibilidad de migrar a ChromaDB, Milvus, etc.)  
- **Embeddings / LLM**: Google Generative AI (Gemini)  
- **Lenguaje**: Python 3.10+  

> ⚠️ Nota: El stack es experimental. El diseño permite cambiar componentes sin modificar la arquitectura general.  

---

## 📂 Estructura del proyecto  

```
rag_virtual_teacher/
├── notebooks/                  # Experimentos en .ipynb
│   ├── 01_experimento_loader.ipynb
│   ├── 02_experimento_retriever.ipynb
│   ├── 03_chat_demo.ipynb
│
├── src/
│   └── rag_virtual_teacher/
│       ├── __init__.py
│       ├── config.py           # (antes estaba en la raíz)
│       ├── chatbot.py          # (antes era main.py + pipeline.py)
│       ├── loader.py           # (antes document_loader.py + text_splitter.py)
│       ├── retriever.py        # (antes vectorstore.py + retrievers.py)
│       ├── llm.py              # (antes llms.py)
│       ├── prompts/
│       │   ├── base_prompts.py
│       │   ├── rag_prompts.py
│       │   └── prompt_utils.py
│       └── utils.py            # nuevo: helpers varios
│
├── tests/
│   └── test_loader.py
│
├── data/
│   ├── input/
│   └── processed/
│
├── requirements.txt
└── README.md

```

---

## 🚀 Instalación y uso  

1. Clona este repositorio:  

```bash
git clone https://github.com/tuusuario/rag_virtual_teacher.git
cd rag_virtual_teacher


python -m venv venv
source venv/bin/activate   # En Linux/Mac
venv\Scripts\activate      # En Windows

pip install -r requirements.txt
