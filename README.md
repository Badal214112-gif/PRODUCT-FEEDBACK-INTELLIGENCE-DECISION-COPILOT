# Product Feedback Intelligence & Decision Copilot

An AI-powered analytics and decision-support copilot built to analyze e-commerce product feedback, execute schema-aware data queries, and provide instant, context-driven insights using Retrieval-Augmented Generation (RAG).

---

## Project Overview
Managing and extracting actionable insights from thousands of customer reviews is a major challenge for product teams. **Product Feedback Intelligence & Decision Copilot** automates this workflow by combining vector search, natural language processing, and interactive data querying. It allows stakeholders to query product reviews naturally, uncover pain points, and make data-driven decisions instantly.

---

## Tech Stack & Architecture

* **Frontend & UI:** Streamlit (Interactive web application framework)
* **Orchestration & Framework:** LangChain (For chaining prompts, retrievers, and LLM logic)
* **LLM & Inference:** Groq Cloud APIs (Lightning-fast open-source LLM inference like Llama 3)
* **Vector Embeddings & Search:** Sentence Transformers & FAISS (`faiss-cpu`) for high-performance similarity search over feedback datasets
* **Data Processing & Analytics:** Pandas (For structured CSV data manipulation and dataset handling)
* **State & Configuration:** Python-dotenv & SQLite (For chat history and local environment management)

---

## Key Features

1. **Natural Language Feedback Querying:** Ask questions about product reviews in plain English and get precise analytical summaries.
2. **Hybrid RAG Pipeline:** Combines dense vector retrieval (FAISS) with structured context to eliminate hallucinations.
3. **Optimized Performance:** Fast embedding generation and lightning-fast inference via Groq Cloud.
4. **Interactive Dashboard:** Clean, user-friendly Streamlit interface with session state management and chat history.

---

## Local Installation & Setup

To run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Badal214112-gif/PRODUCT-FEEDBACK-INTELLIGENCE-DECISION-COPILOT.git
   cd PRODUCT-FEEDBACK-INTELLIGENCE-DECISION-COPILOT
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
