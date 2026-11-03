import os
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
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
    cursor.execute("SELECT user_msg, bot_msg FROM messages WHERE session_id = ?", (session_id,))
    return [{"user": row[0], "bot": row[1]} for row in cursor.fetchall()]

# --- Session State Caches for Performance ---
if "vector_store_cache" not in st.session_state:
    st.session_state.vector_store_cache = None
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None

# --- SIDEBAR: Clean Dropdown & Management ---
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

# --- DATA LOADING & MULTI-FILE HANDLING LOGIC ---
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
    d_hash = "default"
else:
    if uploaded_files:
        file_dfs, df = load_data("Upload Custom Files", uploaded_files)
        d_hash = str(df.shape) if df is not None else "empty"
    else:
        file_dfs, df = {}, None
        d_hash = "none"

@st.cache_resource
def init_sqlite_db(df_in):
    try:
        conn = sqlite3.connect(':memory:', check_same_thread=False)
        df_in.to_sql('reviews', conn, index=False, if_exists='replace')
        return conn
    except Exception as e:
        return None

if df is not None:
    db_conn = init_sqlite_db(df)
    
    files_overview_text = "Uploaded Datasets Summary:\n"
    if file_dfs:
        for fname, fdf in file_dfs.items():
            files_overview_text += f"- File Name: {fname} | Shape (Rows, Cols): {fdf.shape} | Columns: {list(fdf.columns)}\n"

    if st.session_state.vector_store_cache is None or st.session_state.last_hash != d_hash:
        with st.spinner("Processing data and preparing search index..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            index_path = "vector_index"
            
            if os.path.exists(index_path) and d_hash == "default":
                vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            else:
                sample_df = df.head(1000)
                docs = []
                for i, row in sample_df.iterrows():
                    row_str = " | ".join([f"{col}: {row[col]}" for col in df.columns[:5]])
                    docs.append(Document(page_content=row_str, metadata={"row": i}))
                
                if docs:
                    vector_store = FAISS.from_documents(docs, embeddings)
                    if d_hash == "default":
                        vector_store.save_local(index_path)
                else:
                    vector_store = None
                    
            st.session_state.vector_store_cache = vector_store
            st.session_state.last_hash = d_hash

    vector_store = st.session_state.vector_store_cache
    retriever = vector_store.as_retriever(search_kwargs={"k": 6}) if vector_store else None

    # --- CENTERED MAIN LAYOUT ---
    left_margin, center_col, right_margin = st.columns([1, 8, 1])

    with center_col:
        st.title("AI Product Intelligence Copilot")
        
        with st.expander("View Dataset Preview", expanded=False):
            if file_dfs:
                selected_preview_file = st.selectbox("Select file to preview:", list(file_dfs.keys()))
                st.dataframe(file_dfs[selected_preview_file].head(10), use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)

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
                    llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY, temperature=0.0)
                    
                    router_prompt = f"""
                    You are a smart router for an analytics copilot. 
                    Classify the user query into one of four categories:
                    1. 'SQL' if the query asks for counting, aggregates, statistics, metrics, sorting, or grouping from the database.
                    2. 'RAG' if the query asks for qualitative reasons, feedback details, specific complaints, or opinions from the reviews dataset.
                    3. 'WEB' if the query asks for current affairs, current political leaders or presidents, real-time facts, market trends, external industry knowledge, or general web facts that require up-to-date web search.
                    4. 'GENERAL' if it's a casual greeting (like hi, hello) or general conversation not requiring data lookup or search.

                    User Query: {user_query}
                    
                    Output ONLY ONE WORD: SQL, RAG, WEB, or GENERAL.
                    """
                    route_type = llm.invoke(router_prompt).content.strip().upper()
                    
                    context_data = ""
                    if "SQL" in route_type and db_conn:
                        sql_prompt = f"""
                        You are a SQLite expert. The table name is `reviews`. 
                        Columns available: {list(df.columns)}.
                        Write a valid SQL query to answer the user's question. 
                        Return ONLY the raw SQL query inside a markdown code block. No explanation.
                        User Question: {user_query}
                        """
                        sql_query_raw = llm.invoke(sql_prompt).content
                        sql_query = sql_query_raw.replace("```sql", "").replace("```", "").strip()
                        try:
                            sql_result_df = pd.read_sql_query(sql_query, db_conn)
                            context_data = f"SQL Query Executed:\n{sql_query}\n\nSQL Results Data:\n{sql_result_df.to_string()}"
                        except Exception:
                            docs = retriever.invoke(user_query) if retriever else []
                            context_data = "\n\n".join([d.page_content for d in docs])
                    elif "RAG" in route_type and retriever:
                        docs = retriever.invoke(user_query)
                        context_data = "\n\n".join([d.page_content for d in docs])
                    elif "WEB" in route_type:
                        try:
                            search_results = []
                            with DDGS() as ddgs:
                                for r in ddgs.text(user_query, max_results=3):
                                    search_results.append(r.get('body', ''))
                            search_result = "\n".join(search_results) if search_results else "No web results found."
                            context_data = f"Web Search Results:\n{search_result}"
                        except Exception as search_err:
                            context_data = f"Web search failed: {search_err}"
                    else:
                        if any(keyword in user_query.lower() for keyword in ["data", "file", "overview", "what", "tell me"]):
                            context_data = files_overview_text
                        else:
                            context_data = "General conversation or greeting."

                    system_msg = SystemMessage(
                        content=(
                            "You are an expert AI Product Analyst and Customer Insights Copilot. "
                            "Strictly follow these rules:\n"
                            "1. Match the language of the user's query: if English, reply in English. If Hinglish/Hindi, reply in Hinglish.\n"
                            "2. If the user query is a casual greeting (like hi, hello), reply back politely and concisely without mentioning any datasets.\n"
                            "3. If Route Type is WEB, you MUST base your answer STRICTLY and EXCLUSIVELY on the provided Web Search Results. Do NOT rely on old pre-trained memory.\n"
                            "4. If the user asks about their data or files, list the available uploaded datasets summary and ask which one they want to explore.\n"
                            "5. Keep answers short, crisp, and data-driven."
                        )
                    )
                    
                    human_msg = HumanMessage(content=f"Route Type: {route_type}\n\nContext:\n{context_data}\n\nQuery:\n{user_query}")
                    prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
                    chain = prompt | llm | StrOutputParser()
                    
                    st.markdown("### AI Analysis")
                    response = st.write_stream(chain.stream({}))
                    
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
                    llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY, temperature=0.0)
                    
                    docs = retriever.invoke(user_query) if retriever else []
                    rag_context = "\n\n".join([d.page_content for d in docs])
                    
                    web_context = ""
                    try:
                        with DDGS() as ddgs:
                            results = [r.get('body', '') for r in ddgs.text(f"market trends and best practices for {user_query}", max_results=3)]
                        web_context = "\n".join(results) if results else "Web search unavailable."
                    except Exception:
                        web_context = "Web search unavailable."
                    
                    combined_context = f"Internal Dataset Context:\n{rag_context}\n\nExternal Market Context:\n{web_context}"
                    
                    sys_rec = SystemMessage(
                        content="You are an expert AI Product Analyst. Provide 2 to 3 short, punchy recommendations based on the provided context and user query."
                    )
                    hum_rec = HumanMessage(
                        content=f"Context Data:\n{combined_context}\n\nUser Query:\n{user_query}"
                    )
                    
                    rec_prompt = ChatPromptTemplate.from_messages([sys_rec, hum_rec])
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