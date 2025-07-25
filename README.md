# AI-Engineer-Level-1-Assessment

This is a **FastAPI-based Retrieval-Augmented Generation (RAG)** system that supports **Bangla** and **English** queries. It retrieves semantically relevant answers from a Bangla HSC book PDF and generates grounded responses using a local **Mistral LLM** via **Ollama**.

---

## 🚀 Features

- ✅ Supports Bangla & English queries
- 🔍 Semantic document search using FAISS + multilingual embeddings
- 🤖 Local LLM inference using Ollama + Mistral
- 📄 Preprocess and chunk Bangla HSC book (Unicode PDF)
- 📡 `/ask` API endpoint returns answers, similarity scores, and context
- 🧠 Built-in evaluation for relevance and groundedness

---

## 🗂️ Folder Structure

```
AI-Engineer-Level-1-Assessment/
├── app.py                     # FastAPI application with RAG implementation
├── document_loader.py         # PDF loading and text chunking logic
├── vector_store.py           # FAISS vector store management
├── requirements.txt          # Project dependencies
├── HSC26-Bangla1st-Paper.pdf # Source PDF document
├── .gitignore               # Git ignore file
└── faiss_index/             # Directory for storing FAISS indices
```

---

## 📊 Evaluation Matrix

### 1. Text Extraction & Formatting

**Implementation:**

- Used `PyPDF` library for text extraction due to its robust Unicode support, essential for Bangla text
- Implemented custom preprocessing to handle:
  - Unicode normalization for Bangla characters
  - Proper paragraph separation
  - Removal of headers/footers and page numbers

**Challenges:**

- Handling embedded fonts and ensuring proper Bangla character rendering
- Maintaining proper paragraph breaks and text flow
- Dealing with multi-column layouts and text boxes

### 2. Chunking Strategy

**Implementation:**

- Chunk size: 1000 characters with 100-character overlap
- Paragraph-aware chunking to maintain context
- Overlap ensures semantic continuity between chunks

**Rationale:**

- Balances context window size with retrieval granularity
- Preserves natural text boundaries for better semantic understanding
- Overlap prevents information loss at chunk boundaries

### 3. Embedding Model

**Choice: `intfloat/multilingual-e5-base`**

**Rationale:**

- Strong multilingual support, especially for Bangla
- Good balance between model size and performance
- State-of-the-art performance on cross-lingual retrieval tasks
- Efficient encoding of both semantic and contextual information

### 4. Similarity & Storage

**Implementation:**

- FAISS for efficient similarity search
- Cosine similarity for comparing query and document embeddings
- In-memory vector store with disk persistence

**Benefits:**

- Fast nearest neighbor search with FAISS
- Cosine similarity handles different text lengths well
- Efficient scaling with large document collections

### 5. Query-Document Matching

**Strategy:**

- Top-K retrieval (K=3) for relevant chunks
- Similarity threshold filtering
- Context aggregation for answer generation

**Handling Edge Cases:**

- For vague queries: Return lower confidence scores
- For missing context: Prompt user for clarification
- Multiple retrieval rounds if needed

### 6. Results Analysis

**Current Performance:**

- Strong relevance for specific queries
- Good handling of bilingual queries
- Accurate context retrieval

**Potential Improvements:**

- Fine-tune embedding model on domain-specific data
- Implement dynamic chunking based on content
- Add query expansion for better matching
- Increase context window for complex queries

---

## 🔧 Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

<details>

  <summary>📦 Click here to see the packages</summary>

- `fastapi`
- `uvicorn`
- `langchain`
- `langchain-community`
- `sentence-transformers`
- `faiss-cpu`
- `pypdf`
- `ollama`

</details>

---

## 🛠️ Setup & Run Instructions

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/your-username/rag-bangla-fastapi.git
cd rag-bangla-fastapi
python -m venv .venv
source env/bin/activate  # Windows: .\env\Scripts\activate

```

### 2️⃣ Start Ollama & Pull Model

Install and run Ollama:

```bash
ollama serve
ollama pull mistral
```

```bash
ollama run mistral
```

### 3️⃣ Launch the FastAPI Server

```bash
uvicorn main:app --reload --port 8000
```

---

## 📡 API Documentation

### POST `/ask`

Send questions in Bangla or English to get answers from the HSC book content.

#### Request Payload

```json
{
    "question": "string",          // Your question in Bangla or English
    "include_context": boolean     // Optional: Set true to include matched context (default: false)
}
```

#### Response Format

```json
{
    "answer": "string",           // Generated answer from the LLM
    "similarities": [             // Cosine similarity scores for each retrieved chunk
        float,                    // Range: 0.0 to 1.0
        ...
    ],
    "groundedness": float,        // How well the answer is grounded in the context (0.0 to 1.0)
    "contexts": [                 // Only included if include_context=true
        "string",                // Retrieved text chunks used for answer generation
        ...
    ]
}
```

#### Example

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "question": "বাংলা সাহিত্যের প্রথম যুগে কোন কোন ধরনের রচনা ছিল?",
           "include_context": true
         }'
```

#### Response Example

```json
{
  "answer": "বাংলা সাহিত্যের প্রথম যুগে মূলত চর্যাপদ, মঙ্গলকাব্য এবং অনুবাদ সাহিত্য রচিত হয়েছিল...",
  "similarities": [0.89, 0.78, 0.72],
  "groundedness": 0.83,
  "contexts": [
    "চর্যাপদ বাংলা সাহিত্যের প্রাচীনতম নিদর্শন...",
    "মঙ্গলকাব্যের যুগে বাংলা সাহিত্য...",
    "রামায়ণ মহাভারতের অনুবাদের মাধ্যমে..."
  ]
}
```

#### Notes

- The API supports both Bangla and English queries
- Similarity scores indicate how well each retrieved chunk matches the query
- Groundedness score shows how well the answer is supported by the source text
- Include `include_context: true` to see the text chunks used for answer generation

---
