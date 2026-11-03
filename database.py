import os
import sqlite3
import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

@st.cache_data
def load_data(file_path="Womens Clothing E-Commerce Reviews.csv/Womens Clothing E-Commerce Reviews.csv"):
    try:
        if not os.path.exists(file_path):
            file_path = "Womens Clothing E-Commerce Reviews.csv"
        if not os.path.exists(file_path):
            st.error("Error: Dataset file nahi mili.")
            return None
            
        df = pd.read_csv(file_path)
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        if 'review_text' in df.columns:
            df['review_text'] = df['review_text'].fillna("")
        else:
            df['review_text'] = ""
        return df
    except Exception as e:
        st.error(f"Dataset load karne mein error aaya: {e}")
        return None

@st.cache_resource
def init_sqlite_db(df):
    try:
        conn = sqlite3.connect(':memory:', check_same_thread=False)
        df.to_sql('reviews', conn, index=False, if_exists='replace')
        return conn
    except Exception as e:
        return None

@st.cache_resource
def get_vector_store(data):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        sample_df = data.head(5000)
        docs = []
        for i, row in sample_df.iterrows():
            class_name = row.get('class_name', 'N/A')
            rating = row.get('rating', 'N/A')
            review = row.get('review_text', '')
            text = f"Product Class: {class_name} | Rating: {rating} Stars | Review: {review}"
            docs.append(Document(page_content=text, metadata={"row": i}))
        
        if not docs:
            return None
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Vector store error: {e}")
        return None