# 🤖 Simple GenAI App Using LangChain

A beginner-friendly **Retrieval-Augmented Generation (RAG)** project built with **LangChain, Hugging Face Embeddings, and FAISS**.

The project demonstrates how to collect information from a website, split the content into smaller chunks, convert the chunks into vector embeddings, store them in a vector database, and retrieve the most relevant information using semantic similarity search.

> **Current implementation:** This project focuses on the **data ingestion, embedding, vector storage, and retrieval** stages of a RAG pipeline.

---

## 🚀 Project Overview

This project uses the **Tata Consultancy Services (TCS) Wikipedia page** as the knowledge source.

The pipeline follows these steps:

```text
Website
   ↓
Web Scraping
   ↓
Document Loading
   ↓
Text Chunking
   ↓
Hugging Face Embeddings
   ↓
FAISS Vector Database
   ↓
Similarity Search
   ↓
Relevant Documents
```

The project is designed to demonstrate the core concepts behind modern **RAG-based GenAI applications**.

---

## ✨ Features

* 🌐 Scrapes content from a web page
* 📄 Loads website content using LangChain
* ✂️ Splits large documents into smaller chunks
* 🧠 Generates semantic embeddings using Hugging Face
* 🔎 Performs similarity search
* ⚡ Uses FAISS for efficient vector storage and retrieval
* 🔐 Supports environment variables using `.env`
* 📊 Includes optional LangSmith tracing configuration

---

## 🛠️ Tech Stack

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Programming language            |
| LangChain             | LLM application framework       |
| WebBaseLoader         | Website data ingestion          |
| Hugging Face          | Text embeddings                 |
| Sentence Transformers | Embedding model                 |
| FAISS                 | Vector database                 |
| LangSmith             | Experiment tracking/tracing     |
| python-dotenv         | Environment variable management |

---

## 🧠 Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through LangChain's `HuggingFaceEmbeddings`.

This converts text chunks into numerical vectors that can be compared based on semantic similarity.

---

## 📚 Data Source

The current implementation uses the Tata Consultancy Services Wikipedia page:

```text
https://en.wikipedia.org/wiki/Tata_Consultancy_Services
```

The webpage is loaded using LangChain's `WebBaseLoader`.

The retrieved content is then converted into LangChain `Document` objects for further processing.

---

## 🔄 How the RAG Pipeline Works

### 1. Load Environment Variables

The project uses `python-dotenv` to load environment variables from a `.env` file.

It also configures LangSmith tracing:

```python
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
```

---

### 2. Data Ingestion

The website is loaded using LangChain's `WebBaseLoader`:

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://en.wikipedia.org/wiki/Tata_Consultancy_Services"
)

docs = loader.load()
```

This retrieves the webpage and converts it into documents that can be processed by LangChain.

---

### 3. Text Chunking

Large documents are divided into smaller pieces using `RecursiveCharacterTextSplitter`.

The current configuration is:

```python
chunk_size = 1000
chunk_overlap = 200
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

text = text_splitter.split_documents(docs)
```

### Why chunking?

LLMs and embedding models work better when large documents are broken into manageable sections.

The overlap helps preserve context between consecutive chunks.

---

### 4. Generate Embeddings

The project uses Hugging Face's `all-MiniLM-L6-v2` model:

```python
from langchain_huggingface import HuggingFaceEmbeddings

Embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Each text chunk is transformed into a vector representation.

---

### 5. Create FAISS Vector Database

The embeddings are stored using FAISS:

```python
from langchain_community.vectorstores import FAISS

DB = FAISS.from_documents(text, Embedding)
```

FAISS allows the application to efficiently search for documents that are semantically similar to a user's query.

---

### 6. Similarity Search

The vector database can then be queried:

```python
query = "ALL about TCS ? ?"

result = DB.similarity_search(query)

result[0].page_content
```

The system searches the vector database and returns the documents that are most relevant to the query.

---

## 💡 Example

A query such as:

```text
ALL about TCS?
```

is converted into an embedding and compared against the embeddings stored in FAISS.

The most semantically relevant document chunks are returned as the search results.

---

## 📁 Project Structure

A recommended GitHub structure for this project is:

```text
Simple-GenAI-App/
│
├── simpleApp.ipynb
├── README.md
├── requirements.txt
├── .env
└── .gitignore
```

> Never commit your `.env` file or API keys to GitHub.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -U langchain
pip install -U langchain-community
pip install -U langchain-text-splitters
pip install -U langchain-huggingface
pip install -U sentence-transformers
pip install -U faiss-cpu
pip install -U python-dotenv
pip install -U beautifulsoup4
```

Or install everything using:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project directory:

```env
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_project_name
```

If you later enable Google Gemini generation in the project, you can also add:

```env
GOOGLE_API_KEY=your_google_api_key
```

**Do not upload `.env` to GitHub.**

Add this to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
.ipynb_checkpoints/
```

---

## ▶️ Running the Project

Open the notebook:

```bash
jupyter notebook simpleApp.ipynb
```

or:

```bash
jupyter lab
```

Then execute the cells sequentially.

---

## 🧩 Current Architecture

```text
              ┌──────────────────────┐
              │   TCS Wikipedia Page │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   WebBaseLoader      │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Document Chunks    │
              │  1000 chars / 200    │
              │      overlap         │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ HuggingFace Embedding │
              │ all-MiniLM-L6-v2     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │    FAISS Vector DB   │
              └──────────┬───────────┘
                         │
                    User Query
                         │
                         ▼
              ┌──────────────────────┐
              │ Similarity Search    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Relevant Documents   │
              └──────────────────────┘
```

---

## 🔮 Future Improvements

The current project establishes the retrieval layer of a RAG application. It can be extended into a complete GenAI application by adding:

* [ ] Connect retrieved documents to an LLM
* [ ] Generate natural-language answers from retrieved context
* [ ] Add a proper RAG prompt template
* [ ] Add Google Gemini or another LLM for response generation
* [ ] Build a Streamlit/Gradio chat interface
* [ ] Add conversation history
* [ ] Persist the FAISS index
* [ ] Add source citations to generated answers
* [ ] Allow users to provide their own URLs
* [ ] Improve chunking and retrieval strategies
* [ ] Add retrieval evaluation
* [ ] Deploy the application

---

## 🎯 Learning Outcomes

Through this project, you can understand:

* What RAG is
* How website data can be ingested
* Why documents are chunked
* What embeddings are
* How semantic search works
* How vector databases are used
* How FAISS can be integrated with LangChain
* How LangSmith can be used for tracing

---

## ⚠️ Limitations

This is currently a **retrieval-focused prototype**, rather than a complete conversational RAG application.

The notebook retrieves relevant document chunks using FAISS, but the current code does not yet pass those retrieved chunks to an LLM to generate a final natural-language response.

The notebook also contains a `ChatGoogleGenerativeAI` import, but the reviewed implementation does not actually instantiate or invoke the model.

---

## 📌 Project Status

```text
Data Ingestion       ✅
Document Loading     ✅
Text Chunking        ✅
Embeddings           ✅
Vector Database      ✅
Similarity Search    ✅
LLM Generation       ⏳
RAG Chat Interface   ⏳
Deployment            ⏳
```

---

## 👨‍💻 Author

**Anand Yadav**

B.Tech Graduate | AI/ML & Generative AI

Interested in:

* Generative AI
* LLM Applications
* RAG Systems
* AI Agents
* Machine Learning
* AI Engineering

---

## ⭐ If You Found This Useful

If this project helped you understand the basics of RAG and vector search, consider giving the repository a ⭐.

---

## 📄 License

This project is intended for educational and learning purposes.
