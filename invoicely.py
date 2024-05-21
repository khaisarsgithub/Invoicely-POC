from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

if os.path.exists("./invoicely_business.pkl"):
    print("Loading vector store from pickle file...")
    with open("./invoicely_business.pkl", "rb") as f:
        serialized_data = pickle.load(f)
        documents = serialized_data['documents']
        vectors = serialized_data['embeddings']

else:
    with open("./invoicely_business.pkl", "wb") as f:
        print("Loading...")
        text = './buisiness_invoicely.txt'
        loader = TextLoader('./buisiness_invoicely.txt', encoding='utf-8')
        data = loader.load()

        print("Splitting...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(data)

        # convert to array format
        data = []
        for i in range(len(splits)):
            data.append(splits[i].page_content)


        encoder = SentenceTransformer("all-mpnet-base-v2")
        vectors = encoder.encode(data)


        serialized_data = {
            'documents': splits,
            'embeddings': vectors
        }
        pickle.dump(serialized_data, f)
        print("Vector store saved to pickle file")

dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)

if 'chat_history' in st.session_state:
    chat_history = st.session_state.chat_history
else:
    chat_history = []


template = """
Answer the question correctly based on the context.
Question: {question}
Context: {context}
"""
# prompt = PromptTemplate(
#     input_variables=["question", "context"],
#     template=template
# )

llm = OpenAI(temperature = 0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))

search_query = input("Search: ")
if st.button("Submit"):
    chat_history.append(search_query)
    vec = encoder.encode(search_query)
    D, I = index.search(vec, 4)
    context = [splits[i] for i in I[0]]
    print(I[i])
    print(D[i])
    response = llm.invoke(template, context=context, question=search_query)
    st.write(response)
    chat_history.append(response)