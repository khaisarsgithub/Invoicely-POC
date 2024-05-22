from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

print("Loading data...")
loader = TextLoader('./buisiness_invoicely.txt', encoding='utf-8')
data = loader.load()

print("Splitting data...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(data)
documents = [split.page_content for split in splits]

print("Vectorizing data...")
encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(documents)

print("Creating FAISS index...")
dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

llm = OpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))

st.title("Invoicely Chat Support")
question = st.text_input("Search: ")
if st.button("Submit"):
    st.session_state.chat_history.append("You: " + question)
    vec = encoder.encode(question).reshape(1, -1)
    D, I = index.search(vec, 4)
    context = [documents[i] for i in I[0]]
    
    prompt = f"""
    Answer the question in the format.
    Step 1: ...\n
    Step 2: ...\n
    Question: {question}
    Context: {' '.join(context)}
    """
    response = llm.invoke(prompt)
    st.write(response)
    st.session_state.chat_history.append("AI: " + response)

st.title("Chat History")
for chat in st.session_state.chat_history:
    st.write(chat)
