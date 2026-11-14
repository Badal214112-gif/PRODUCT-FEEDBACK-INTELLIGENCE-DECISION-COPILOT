import os
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from duckduckgo_search import DDGS

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Any

from database import load_data, init_sqlite_db, get_vector_store
from memory import init_memory

load_dotenv()

st.set_page_config(page_title="Product Intelligence Copilot", layout="wide")
init_memory()

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

# Load Data based on selection
if data_source_option == "Default Dataset":
    df = load_data()
    file_dfs = {"Default Dataset": df} if df is not None else {}
else:
    if uploaded_files:
        file_dfs = {}
        for f in uploaded_files:
            if f.name.endswith('.csv'):
                temp_df = pd.read_csv(f)
            else:
                temp_df = pd.read_excel(f)
            temp_df.columns = [col.strip().replace(' ', '_').lower() for col in temp_df.columns]
            # Ensure review_text mapping for custom files if exists
            if 'review_text' not in temp_df.columns:
                text_cols = [c for c in temp_df.columns if 'text' in c or 'comment' in c or 'review' in c or 'content' in c]
                if text_cols:
                    temp_df['review_text'] = temp_df[text_cols[0]].fillna("")
                else:
                    temp_df['review_text'] = temp_df.iloc[:, 0].astype(str).fillna("")
            else:
                temp_df['review_text'] = temp_df['review_text'].fillna("")
                
            file_dfs[f.name] = temp_df
        df = pd.concat(list(file_dfs.values()), ignore_index=True, sort=False) if file_dfs else None
    else:
        df = None
        file_dfs = {}

st.session_state.df = df
st.session_state.file_dfs = file_dfs

if df is not None:
    vector_store = get_vector_store(df)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}) if vector_store else None
    sqlite_conn = init_sqlite_db(df)

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)

    # --- LangGraph State & Checkpointer Setup ---
    class CopilotState(TypedDict):
        user_query: str
        active_source: str
        tabular_context: str
        rag_context: str
        response: str

    def analysis_node(state: CopilotState):
        query = state["user_query"]
        source_type = state["active_source"]
        tabular_context = ""
        
        # Explicit check for general dataset overview queries based on active data source
        is_overview_query = any(kw in query.lower() for kw in ["tell me about data", "about the data", "dataset", "overview", "what is the data"])
        if is_overview_query and df is not None:
            if source_type == "Upload Custom Files" and file_dfs:
                details = "".join([f"\n- File Name: {fname} | Rows: {fdf.shape[0]} | Columns: {list(fdf.columns)}" for fname, fdf in file_dfs.items()])
                tabular_context = f"Custom Uploaded Datasets Overview:{details}\n- Total Combined Rows: {df.shape[0]}, Total Columns: {df.shape[1]}"
            else:
                tabular_context = f"Default Dataset Overview:\n- Total Rows: {df.shape[0]}\n- Total Columns: {df.shape[1]}\n- Column Names: {list(df.columns)}"
        else:
            is_tabular_query = any(kw in query.lower() for kw in ["row", "column", "shape", "count", "stat", "average", "rating", "distribution", "how many", "list columns", "summary"])
            if is_tabular_query and sqlite_conn:
                try:
                    code_prompt = f"""
                    You are a SQLite and Pandas data expert. The dataset is loaded in SQLite table named `reviews`.
                    Columns available: {list(df.columns)}.
                    Write a valid SQL query to answer this user request: "{query}"
                    Return ONLY the raw SQL query inside ```sql ... ``` block. No extra text.
                    """
                    sql_raw = llm.invoke(code_prompt).content
                    sql_clean = sql_raw.replace("```sql", "").replace("```", "").strip()
                    if sql_clean.lower().startswith("select"):
                        res_df = pd.read_sql_query(sql_clean, sqlite_conn)
                        tabular_context = f"SQL Execution Query:\n{sql_clean}\n\nResult:\n{res_df.to_string(index=False)}"
                except Exception:
                    tabular_context = f"Dataset shape: {df.shape}, Columns: {list(df.columns)}"
                    
        return {"tabular_context": tabular_context}

    def rag_node(state: CopilotState):
        query = state["user_query"]
        rag_context = ""
        if any(kw in query.lower() for kw in ["review", "feedback", "complaint", "unhappy", "why", "issue", "sentiment", "reason"]):
            if retriever:
                docs = retriever.invoke(query)
                rag_context = "\n\n".join([d.page_content for d in docs])
        return {"rag_context": rag_context}

    workflow = StateGraph(CopilotState)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("rag", rag_node)
    workflow.set_entry_point("analysis")
    workflow.add_edge("analysis", "rag")
    workflow.add_edge("rag", END)

    memory_saver = MemorySaver()
    copilot_graph = workflow.compile(checkpointer=memory_saver)

    left_margin, center_col, right_margin = st.columns([1, 8, 1])

    with center_col:
        st.title("AI Product Intelligence Copilot")
        
        with st.expander("View Dataset Preview", expanded=False):
            if file_dfs and data_source_option == "Upload Custom Files":
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
            try:
                config = {"configurable": {"thread_id": str(current_id)}}
                initial_state = {
                    "user_query": user_query, 
                    "active_source": data_source_option,
                    "tabular_context": "", 
                    "rag_context": "", 
                    "response": ""
                }
                final_state = copilot_graph.invoke(initial_state, config=config)

                tabular_context = final_state.get("tabular_context", "")
                rag_context = final_state.get("rag_context", "")

                system_msg = SystemMessage(
                    content=(
                        "You are an expert AI Product Analyst and Conversational Analytics Copilot. "
                        "Strictly follow these rules:\n"
                        "1. Match the language of the user's query: if English, reply in English. If Hinglish/Hindi, reply in Hinglish.\n"
                        "2. YOU HAVE COMPLETE STATEFUL MEMORY. Look at the entire chat history provided below. Remember user's name, previous numbers, logic, and answers given across previous turns.\n"
                        "3. If the user asks about the dataset overview, rows, columns, or shape, use the provided Dataset Overview data directly (whether it's the Default Dataset or Uploaded Custom Files) and do not give generic definitions.\n"
                        "4. If the user asks about customer feedback, complaints, or sentiment, use the provided Customer Reviews Context.\n"
                        "5. Keep answers professional, precise, and structured."
                    )
                )
                
                past_messages = get_session_messages(current_id)
                history_messages = []
                for msg in past_messages:
                    history_messages.append(HumanMessage(content=msg["user"]))
                    history_messages.append(AIMessage(content=msg["bot"]))

                current_turn_content = f"User Query:\n{user_query}"
                if tabular_context:
                    current_turn_content += f"\n\nTabular/SQL Analysis Data:\n{tabular_context}"
                if rag_context:
                    current_turn_content += f"\n\nCustomer Reviews Context:\n{rag_context}"

                human_msg = HumanMessage(content=current_turn_content)
                
                prompt_messages = [system_msg] + history_messages + [human_msg]
                prompt = ChatPromptTemplate.from_messages(prompt_messages)
                chain = prompt | llm | StrOutputParser()
                
                st.markdown("### AI Analysis")
                response = st.write_stream(chain.stream({}))
                
                cursor = chat_conn.cursor()
                cursor.execute("INSERT INTO messages (session_id, user_msg, bot_msg) VALUES (?, ?, ?)", (current_id, user_query, response))
                chat_conn.commit()
                    
            except Exception as e:
                st.error(f"Error occurred: {e}")

        elif get_rec and user_query:
            try:
                docs = retriever.invoke(user_query) if retriever else []
                rag_context = "\n\n".join([d.page_content for d in docs])
                
                web_context = ""
                try:
                    with DDGS(timeout=10) as ddgs:
                        results = [r.get('body', '') for r in ddgs.text(f"market trends and best practices for {user_query}", max_results=3)]
                    web_context = "\n".join(results) if results else "Web search unavailable."
                except Exception:
                    web_context = "Web search network unavailable."
                
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