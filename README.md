# Conversational RAG Q&A with PDF Uploads and Chat History

A **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs and ask questions interactively. Built with **Streamlit**, **LangChain**, **Groq AI**, and **HuggingFace embeddings**, it retrieves relevant content from documents and provides context-aware answers while maintaining **chat history**.

---

## Features

- Upload multiple PDF files for analysis.  
- Ask questions and get answers based only on PDF content.  
- Maintains chat history per session.  
- Uses **Groq LLM** and **HuggingFace embeddings** for intelligent retrieval.  
- Web interface with **Streamlit** for ease of use.

---

## Tech Stack

- Python 3.10+  
- Streamlit (Web Interface)  
- LangChain (RAG Pipelines)  
- Chroma (Vector Database)  
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)  
- Groq AI (`ChatGroq`)  
- dotenv (Environment Variables)

---

## Installation

1. Clone the repository:  

git clone https://github.com/yourusername/rag-pdf-chat.git
cd rag-pdf-chat
Create a virtual environment:


python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
Install dependencies:

pip install -r requirements.txt
Create a .env file with:

env
Copy code
HF_TOKEN=your_huggingface_api_token
Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Enter your Groq API key.

Upload PDF files and enter a Session ID.

Ask questions in the input box — answers and chat history will appear.

How It Works
PDFs are loaded via PyPDFLoader.

Text is split into chunks and embedded using HuggingFace.

Chroma stores embeddings for similarity search.

History-aware retriever reformulates questions using chat history.

RAG chain retrieves relevant chunks and generates answers with ChatGroq.

Chat messages are saved per session for context-aware responses.

Notes
Valid Groq API key and HuggingFace token required.

Processing large PDFs may take a few moments.

License
MIT License

Acknowledgements
LangChain

Streamlit

Chroma

HuggingFace

Groq AI

pgsql
Copy code

---

If you want, I can **also create a ready-to-use `requirements.txt`** for this project so anyone can set it up with one command. This will make your GitHub repo fully functional out of the box.  

Do you want me to do that?
