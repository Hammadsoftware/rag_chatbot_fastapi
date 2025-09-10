from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

retrieval_chain = None
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def rag_pipeline(file_path: str):
    global retrieval_chain

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split PDF into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create embeddings and vector database
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever()

    # LLM setup
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY # Replace with your key
    )

    # Chat prompt for PDF conversations
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant that helps Hammad explore and understand the contents of a PDF document. 
    Always respond in a clear, conversational way, as if chatting naturally.  

    Guidelines:
    - Use only the information available in the provided PDF context.  
    - If the context is not sufficient, politely say: "I don’t know based on the document."  
    - Keep answers concise but informative.  
    - When useful, summarize or explain passages in simple terms.  
    - Maintain continuity of the conversation and remember earlier questions if possible.  
    - Do not make up facts outside the PDF.  

    Context from PDF:
    {context}

    User Question:
    {input}

    Assistant Answer:
    """)

    # Build document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return "RAG pipeline ready!"


def query_pipe_line(question: str):
    global retrieval_chain

    if retrieval_chain is None:
        return {"answer": "No PDF loaded yet. Please upload a PDF first."}

    res = retrieval_chain.invoke({"input": question})["answer"]

    return {"answer": res}
