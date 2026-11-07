import os
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from duckduckgo_search import DDGS

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Product Intelligence Copilot", layout="wide")

def init_chat_db():
    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            user_msg TEXT,
            bot_msg TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    """)
    conn.commit()
    return conn

chat_conn = init_chat_db()

def get_or_create_sessions():
    cursor = chat_conn.cursor()
    cursor.execute("SELECT id, name FROM sessions")
    rows = cursor.fetchall()
    if not rows:
        cursor.execute("INSERT INTO sessions (name) VALUES ('Chat 1')")
        chat_conn.commit()
        cursor.execute("SELECT id, name FROM sessions")
        rows = cursor.fetchall()
    return [{"id": r[0], "name": r[1]} for r in rows]

if "current_session_id" not in st.session_state:
    sessions = get_or_create_sessions()
    st.session_state.current_session_id = sessions[0]["id"]

def get_session_messages(session_id):
    cursor = chat_conn.cursor()
    cursor.execute("SELECT user_msg, bot_msg FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    return [{"user": row[0], "bot": row[1]} for row in cursor.fetchall()]

if "vector_store_cache" not in st.session_state:
    st.session_state.vector_store_cache = None
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None

with st.sidebar:
    st.markdown("### Navigation & Settings")
    
    with st.expander("Data Source Configuration", expanded=True):
        data_source_option = st.selectbox(
            "Select Source", 
            ["Default Dataset", "Upload Custom Files"]
        )
        uploaded_files = None
        if data_source_option == "Upload Custom Files":
            uploaded_files = st.file_uploader(
                "Upload CSV or Excel files", 
                type=["csv", "xlsx", "xls"], 
                accept_multiple_files=True
            )

    st.markdown("---")
    
    with st.expander("Chat Sessions Manager", expanded=True):
        if st.button("Start New Chat", use_container_width=True):
            cursor = chat_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sessions")
            new_id = cursor.fetchone()[0] + 1
            new_session_name = f"Chat {new_id}"
            cursor.execute("INSERT INTO sessions (name) VALUES (?)", (new_session_name,))
            chat_conn.commit()
            st.session_state.current_session_id = cursor.lastrowid
            st.rerun()

        sessions = get_or_create_sessions()
        session_names = [s["name"] for s in sessions]
        
        current_idx = 0
        for i, s in enumerate(sessions):
            if s["id"] == st.session_state.current_session_id:
                current_idx = i
                break
                
        selected_chat = st.selectbox("Switch Active Chat", session_names, index=current_idx)
        
        for s in sessions:
            if s["name"] == selected_chat and st.session_state.current_session_id != s["id"]:
                st.session_state.current_session_id = s["id"]
                st.rerun()

    st.markdown("---")
    
    with st.expander("Chat History Overview", expanded=False):
        active_session = next((s for s in sessions if s["id"] == st.session_state.current_session_id), None)
        if active_session:
            st.markdown(f"**Current: {active_session['name']}**")
            msgs = get_session_messages(active_session["id"])
            if msgs:
                for msg in msgs:
                    st.caption(f"You: {msg['user']}")
                    st.caption(f"Bot: {msg['bot'][:50]}...")
                    st.markdown("---")
            else:
                st.caption("No messages in this chat yet.")

def load_data(data_option, file_objs=None, default_path="Womens Clothing E-Commerce Reviews.csv/Womens Clothing E-Commerce Reviews.csv"):
    try:
        file_dfs = {}
        combined_df = None
        
        if data_option == "Upload Custom Files" and file_objs:
            for file_obj in file_objs:
                ext = file_obj.name.split('.')[-1].lower()
                if ext == 'csv':
                    temp_df = pd.read_csv(file_obj)
                elif ext in ['xlsx', 'xls']:
                    temp_df = pd.read_excel(file_obj)
                else:
                    continue
                
                temp_df.columns = [str(col).strip().replace(' ', '_').lower() for col in temp_df.columns]
                if 'review_text' not in temp_df.columns:
                    text_cols = [c for c in temp_df.columns if 'text' in c or 'comment' in c or 'review' in c or 'content' in c]
                    if text_cols:
                        temp_df['review_text'] = temp_df[text_cols[0]].fillna("")
                    else:
                        temp_df['review_text'] = temp_df.iloc[:, 0].astype(str).fillna("")
                else:
                    temp_df['review_text'] = temp_df['review_text'].fillna("")
                
                file_dfs[file_obj.name] = temp_df
            
            if file_dfs:
                combined_df = pd.concat(list(file_dfs.values()), ignore_index=True, sort=False)
        else:
            file_path = default_path
            if not os.path.exists(file_path):
                file_path = "Womens Clothing E-Commerce Reviews.csv"
            if not os.path.exists(file_path):
                return {}, None
            temp_df = pd.read_csv(file_path)
            temp_df.columns = [str(col).strip().replace(' ', '_').lower() for col in temp_df.columns]
            if 'review_text' not in temp_df.columns:
                temp_df['review_text'] = temp_df.iloc[:, 0].astype(str).fillna("")
            else:
                temp_df['review_text'] = temp_df['review_text'].fillna("")
            
            file_dfs["Default Dataset"] = temp_df
            combined_df = temp_df
            
        return file_dfs, combined_df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return {}, None

if data_source_option == "Default Dataset":
    file_dfs, df = load_data("Default Dataset", None)
    st.session_state.df = df
    st.session_state.file_dfs = file_dfs
    d_hash = "default"
else:
    if uploaded_files:
        file_dfs, df = load_data("Upload Custom Files", uploaded_files)
        st.session_state.df = df
        st.session_state.file_dfs = file_dfs
        d_hash = str(df.shape) if df is not None else "empty"
    else:
        file_dfs, df = {}, None
        st.session_state.df = None
        st.session_state.file_dfs = {}
        d_hash = "none"

active_df = st.session_state.get("df", None)
active_file_dfs = st.session_state.get("file_dfs", {})

if active_df is not None:
    files_overview_text = "Uploaded Datasets Summary:\n"
    if active_file_dfs:
        for fname, fdf in active_file_dfs.items():
            files_overview_text += f"- File Name: {fname} | Shape (Rows, Cols): {fdf.shape} | Columns: {list(fdf.columns)}\n"

    # --- Advanced RAG Vector Store & Metadata Indexing ---
    if st.session_state.vector_store_cache is None or st.session_state.last_hash != d_hash:
        with st.spinner("Building Advanced RAG Vector Index with Structured Metadata..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            index_path = "vector_index"
            
            if os.path.exists(index_path) and d_hash == "default":
                vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            else:
                sample_df = active_df.head(2000)
                docs = []
                for i, row in sample_df.iterrows():
                    rev_text = str(row.get('review_text', '')).strip()
                    if rev_text and rev_text.lower() != 'nan':
                        row_content = f"Review Index {i} | Rating: {row.get('rating', 'N/A')} | Department: {row.get('department_name', 'N/A')} | Class: {row.get('class_name', 'N/A')} | Review Text: {rev_text}"
                        docs.append(Document(page_content=row_content, metadata={"row": i}))
                
                if docs:
                    vector_store = FAISS.from_documents(docs, embeddings)
                    if d_hash == "default":
                        vector_store.save_local(index_path)
                else:
                    vector_store = None
                    
            st.session_state.vector_store_cache = vector_store
            st.session_state.last_hash = d_hash

    vector_store = st.session_state.vector_store_cache
    
    # --- Advanced RAG Property: MMR Retriever ---
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    ) if vector_store else None

    llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY, temperature=0.0)

    left_margin, center_col, right_margin = st.columns([1, 8, 1])

    with center_col:
        st.title("AI Product Intelligence Copilot")
        
        with st.expander("View Dataset Preview", expanded=False):
            if active_file_dfs:
                selected_preview_file = st.selectbox("Select file to preview:", list(active_file_dfs.keys()))
                st.dataframe(active_file_dfs[selected_preview_file].head(10), use_container_width=True)
            else:
                st.dataframe(active_df.head(10), use_container_width=True)

        st.markdown("### Ask Anything About Product Feedback")
        
        user_query = st.text_input("Enter your query:", key="main_query_input")
        
        col1, col2 = st.columns([1, 4])
        generate_clicked = col1.button("Generate Answer", use_container_width=True)
        get_rec = col2.button("Generate Recommendations", use_container_width=True)

        current_id = st.session_state.current_session_id

        if generate_clicked and user_query:
            if not GROQ_API_KEY:
                st.error("Error: GROQ_API_KEY is not set in environment variables.")
            else:
                try:
                    # Optional RAG retrieval if query is explicitly asking about reviews/feedback
                    rag_context = ""
                    if any(kw in user_query.lower() for kw in ["review", "feedback", "complaint", "rating", "customer", "product", "issue"]):
                        if retriever:
                            docs = retriever.invoke(user_query)
                            rag_context = "\n\n".join([d.page_content for d in docs])

                    system_msg = SystemMessage(
                        content=(
                            "You are an expert AI Product Analyst and Conversational Assistant. "
                            "Strictly follow these rules:\n"
                            "1. Match the language of the user's query: if English, reply in English. If Hinglish/Hindi, reply in Hinglish.\n"
                            "2. YOU HAVE COMPLETE CONVERSATIONAL MEMORY. Look at the entire chat history provided below. Remember user's name, previous numbers, logic, comparisons, and answers given across previous turns.\n"
                            "3. If the user asks general questions, math, logic, or about themselves (like 'what is my name'), answer directly and accurately using the history.\n"
                            "4. If relevant customer review context is provided, use it to give professional analytical insights.\n"
                            "5. Keep answers short, crisp, and precise."
                        )
                    )
                    
                    # --- STRICT SESSION-SPECIFIC HISTORY LIST LOOP ---
                    past_messages = get_session_messages(current_id)
                    history_messages = []
                    for msg in past_messages:
                        history_messages.append(HumanMessage(content=msg["user"]))
                        history_messages.append(AIMessage(content=msg["bot"]))

                    # Current turn message with optional context
                    current_turn_content = f"User Query:\n{user_query}"
                    if rag_context:
                        current_turn_content += f"\n\nCustomer Reviews Context:\n{rag_context}"

                    human_msg = HumanMessage(content=current_turn_content)
                    
                    # Complete strict list loop passed to LLM
                    prompt_messages = [system_msg] + history_messages + [human_msg]
                    prompt = ChatPromptTemplate.from_messages(prompt_messages)
                    chain = prompt | llm | StrOutputParser()
                    
                    st.markdown("### AI Analysis")
                    response = st.write_stream(chain.stream({}))
                    
                    # Save current exchange to active session SQLite database
                    cursor = chat_conn.cursor()
                    cursor.execute("INSERT INTO messages (session_id, user_msg, bot_msg) VALUES (?, ?, ?)", (current_id, user_query, response))
                    chat_conn.commit()
                        
                except Exception as e:
                    st.error(f"Error occurred: {e}")

        elif get_rec and user_query:
            if not GROQ_API_KEY:
                st.error("Error: GROQ_API_KEY is not set in environment variables.")
            else:
                try:
                    docs = retriever.invoke(user_query) if retriever else []
                    rag_context = "\n\n".join([d.page_content for d in docs])
                    
                    web_context = ""
                    try:
                        with DDGS(timeout=10) as ddgs:
                            results = [r.get('body', '') for r in ddgs.text(f"market trends and best practices for {user_query}", max_results=3)]
                        web_context = "\n".join(results) if results else "Web search unavailable."
                    except Exception:
                        web_context = "Web search network/DNS issue ki wajah से unavailable hai (fallback active)."
                    
                    combined_context = f"Internal Dataset Context:\n{rag_context}\n\nExternal Market Context:\n{web_context}"
                    
                    sys_rec = SystemMessage(
                        content="You are an expert AI Product Analyst. Provide 2 to 3 short, punchy recommendations based on the provided context and user query."
                    )
                    
                    past_messages = get_session_messages(current_id)
                    history_messages = []
                    for msg in past_messages:
                        history_messages.append(HumanMessage(content=msg["user"]))
                        history_messages.append(AIMessage(content=msg["bot"]))

                    hum_rec = HumanMessage(
                        content=f"Context Data:\n{combined_context}\n\nUser Query:\n{user_query}"
                    )
                    
                    rec_prompt = ChatPromptTemplate.from_messages([sys_rec] + history_messages + [hum_rec])
                    chain = rec_prompt | llm | StrOutputParser()
                    
                    st.markdown("### AI Recommendations")
                    rec_response = st.write_stream(chain.stream({}))
                    
                    cursor = chat_conn.cursor()
                    cursor.execute("INSERT INTO messages (session_id, user_msg, bot_msg) VALUES (?, ?, ?)", (current_id, f"Recommendations for: {user_query}", rec_response))
                    chat_conn.commit()
                        
                except Exception as e:
                    st.error(f"Error occurred: {e}")
else:
    st.warning("Please select a dataset or upload custom files to begin.")