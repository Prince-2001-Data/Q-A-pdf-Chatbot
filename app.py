## RAG Q&A Conversation with PDF including chat history
import streamlit as st 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()

# HuggingFace Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Setup streamlit
st.title("Conversational Q&A RAG with PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")


## Displays app title and description.
api_key = st.text_input("Enter your Groq API key:", type="password")


# Check if Groq API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    ## Chat Interface
    session_id = st.text_input("SessionID", value="default_session")


    # User can enter session ID (to separate chats).
    if "store" not in st.session_state:
        st.session_state.store = {}


    #Initialize session state for storing chat history.
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)


    #Process Uploaded PDFs
    documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())


            #Saves uploaded PDF file temporarily for processing.
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)


    #Vector Store & Retriever
    if documents:
        # Split and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        #Splits documents into smaller overlapping chunks.
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()


        #Contextualize Query Prompt
        contextualize_q_system_prompt = """You are an intelligent assistant that helps users query 
        information from uploaded PDF documents. Reformulate the latest user question into a standalone 
        query for the retriever. Use chat history if needed, but do not answer the question yourself."""


        #Prompt for making ambiguous user queries standalone.
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )


        #Template for contextualizing queries with chat history.
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        ## Answer Question (QA System Prompt)
        system_prompt = """You are an intelligent assistant that helps users answer 
        questions based on the content of their uploaded PDF documents.

        Use ONLY the provided retrieved document excerpts to answer.
        - If the answer is not in the documents, say you don’t know. 
        - Be concise but complete in your explanation. 
        - Never make up facts. 
        - If helpful, summarize the information.

        Context:
        {context}
        """


        #System instructions for answering user queries from retrieved chunks.
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )


        #Build RAG Chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


        #Session History Management
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        #User Interaction
        user_input = st.text_input("Your question:")

        #Input box for user questions.
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            

            #Sends user input through the chain → gets answer.
            st.write("Assistant:", response["answer"])
            st.write("Chat history:", session_history.messages)


#API Key Not Provided Case
else:
    st.write("Please enter the Groq API key")
