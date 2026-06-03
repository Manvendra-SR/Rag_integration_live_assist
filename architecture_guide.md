# Live Assist MVP Architecture & Onboarding Guide

## 1. Project Overview

The Live Assist MVP is built to ingest PDF documents into a local RAG (Retrieval-Augmented Generation) index and answer user queries against those documents.

* **Frontend**: Provides the UI for uploading PDFs, managing document filters, and a chat interface for asking questions.
* **API Layer**: Exposes FastAPI HTTP endpoints to receive requests from the frontend (uploads, queries, status checks).
* **Workflow Layer**: Uses LangGraph to manage the state machine for queries, ensuring steps like query rewriting, retrieval, and generation happen sequentially.
* **Retrieval Layer**: Handles semantic (ChromaDB), keyword (BM25), and hybrid searches, applying document filters and reranking results.
* **Storage Layer**: Orchestrates chunking, embedding, and saving vectors/metadata to disk.
* **Runtime Data**: All persistent state (databases, chunks, parsed files, logs, document metadata) lives locally in the `runtime/` folder.

---

## 2. Upload / Ingestion Flow

The complete execution path from when a user uploads a PDF until the document becomes searchable.

```text
User Uploads PDF
       ↓
rag_documents.py
       ↓
  ingestion.py
       ↓
docling_parser.py
       ↓
 semantic_refine.py
       ↓
  local_embed.py
       ↓
 chroma_store.py / bm25_index.py
       ↓
Document Becomes Searchable
```

### Route / API Entry
**File:** `api/routes/rag_documents.py`  
**Purpose:** Receives the PDF file via HTTP POST, saves it to the local `uploads/` folder, and launches the background ingestion task.  
**Calls Next:** `rag_pipeline/ingestion.py`  
**Output:** A fast HTTP 200 response with a `job_id` so the frontend can poll for status.

### Stage 1: Parse
**File:** `rag_pipeline/ingestion.py` (calls `parsers/docling_parser.py`)  
**Purpose:** Reads the raw PDF and extracts text, headings, and structure into a standardized JSON format.  
**Calls Next:** Stage 2 (Chunking) inside `ingestion.py`  
**Output:** Parsed JSON file saved in `runtime/parsed_documents_fast/`.

### Stage 2: Chunk & Enrich
**File:** `rag_pipeline/ingestion.py` (calls `chunkers/semantic_refine.py`)  
**Purpose:** Splits the parsed document into smaller blocks, attaching contextual metadata (like the parent heading hierarchy) to each chunk.  
**Calls Next:** Stage 3 (Embedding) inside `ingestion.py`  
**Output:** Enriched chunks JSON file saved in `runtime/chunks_anthropic/`.

### Stage 3: Vector Embedding
**File:** `rag_pipeline/ingestion.py` (calls `embedders/local_embed.py`)  
**Purpose:** Passes each chunk through an embedding model (e.g., `BAAI/bge-m3`) to generate dense semantic vectors.  
**Calls Next:** Stage 4 (Indexing) inside `ingestion.py`  
**Output:** `.npz` file containing numerical vectors.

### Stage 4: Indexing & Storage
**File:** `retrieval/chroma_store.py` & `retrieval/bm25_index.py`  
**Purpose:** Saves the chunks, vectors, and document metadata into the persistent ChromaDB collection and BM25 index.  
**Calls Next:** Control returns to `rag_documents.py`  
**Output:** Updated vector databases in `runtime/`.

### Stage 5: Finalization
**File:** `api/routes/rag_documents.py` (calls `rag_pipeline/documents.py`)  
**Purpose:** Updates the document's metadata JSON file status to `"ready"` and dumps the execution logs.  
**Calls Next:** End of flow.  
**Output:** Document is fully searchable in the frontend.

---

## 3. Query Flow

The complete execution path from when a user asks a question to when the final answer is returned.

```text
User Asks Question
       ↓
 live_feedback.py
       ↓
    service.py
       ↓
     graph.py
       ↓
    advanced.py
       ↓
     graph.py
       ↓
Final Answer Returned
```

### Route / API Entry
**File:** `api/routes/live_feedback.py`  
**Purpose:** Receives the raw query string and any active document filter from the frontend UI.  
**Calls Next:** `live_cycle/service.py`  
**Output:** A Python dict of the HTTP payload.

### Stage 1: Workflow Bridge
**File:** `live_cycle/service.py` (`handle_manual_question`)  
**Purpose:** Unpacks the request, sets up the LangGraph workflow state (including the document filter), and invokes the graph.  
**Calls Next:** `live_cycle/graph.py`  
**Output:** Initializes the `LiveAssistState` object.

### Stage 2: Query Enrichment
**File:** `live_cycle/graph.py` (`enrich_query` node)  
**Purpose:** Uses a fast LLM call to rewrite the user's potentially vague question into a standalone, retrieval-friendly query.  
**Calls Next:** Stage 3 (`retrieve_knowledge` node) in `graph.py`  
**Output:** `rewriten_question` string added to state.

### Stage 3: Knowledge Retrieval
**File:** `providers/rag/advanced.py` (called by `graph.py`)  
**Purpose:** Based on the strategy (semantic/hybrid/bm25), queries ChromaDB and BM25, applying the `doc_filter` to the metadata, and reranks the results.  
**Calls Next:** Stage 4 (`generate_assist_response` node) in `graph.py`  
**Output:** A list of the most relevant chunks (`rag_top_chunks`) and an assembled context block.

### Stage 4: Response Generation
**File:** `live_cycle/graph.py` (`generate_assist_response` node)  
**Purpose:** Sends the rewritten question and the assembled context block to the primary LLM to generate a factual answer.  
**Calls Next:** Control returns to `service.py`  
**Output:** `answer` string added to state.

### Stage 5: Finalization & Logging
**File:** `live_cycle/service.py`  
**Purpose:** Formats the final state into an `AssistResult`, writes the query execution log to disk, and returns the HTTP 200 response.  
**Calls Next:** End of flow.  
**Output:** Final answer displayed in the frontend.

---

## 4. End-to-End Call Chains

> [!TIP]
> Keep these chains in mind when tracing the flow of data through the backend.

### Upload Flow
`api/routes/rag_documents.py`  
↓  
`rag_pipeline/ingestion.py`  
↓  
`retrieval/chroma_store.py` (and `bm25_index.py`)  
↓  
`rag_pipeline/documents.py`

### Query Flow
`api/routes/live_feedback.py`  
↓  
`live_cycle/service.py`  
↓  
`live_cycle/graph.py`  
↓  
`providers/rag/advanced.py`  
↓  
`live_cycle/service.py`

---

## 5. Main Folders and Responsibilities

* **`frontend/`**  
  * **Purpose:** The user interface (Electron/Web).  
  * **Main files:** `index.html`, `renderer.js`.  
  * **Interaction:** Sends HTTP requests to the `api/` layer to upload PDFs and ask questions.

* **`backend/live_assist/api/`**  
  * **Purpose:** The HTTP entry points for the backend.  
  * **Main files:** `routes/rag_documents.py`, `routes/live_feedback.py`.  
  * **Interaction:** Receives web traffic and delegates work to `rag_pipeline/` or `live_cycle/`.

* **`backend/live_assist/live_cycle/`**  
  * **Purpose:** Orchestrates the conversational state machine.  
  * **Main files:** `service.py` (API bridge), `graph.py` (LangGraph state machine), `state.py` (data schema).  
  * **Interaction:** Calls `providers/` to get data and uses LLMs to process conversations.

* **`backend/live_assist/rag_pipeline/`**  
  * **Purpose:** Handles the offline processing of documents.  
  * **Main files:** `ingestion.py` (orchestrator), `documents.py` (metadata tracker).  
  * **Interaction:** Called by the `api/` upon upload, uses `chunkers/` and `embedders/` to prepare data, and saves to `runtime/`.

* **`backend/live_assist/providers/`**  
  * **Purpose:** Interfaces to external or complex data systems.  
  * **Main files:** `rag/advanced.py` (retrieval engine), `llm/groq.py` (LLM client).  
  * **Interaction:** Called by `graph.py` during a query to fetch context or generate text.

* **`backend/runtime/`**  
  * **Purpose:** The local database and file storage directory.  
  * **Main files:** Contains directories like `chroma_db/`, `bm25/`, `uploads/`, `documents_metadata/`, and `logs/`.  
  * **Interaction:** Read/written to by `rag_pipeline/` and `providers/` constantly.

---

## 6. Debugging Map

> [!IMPORTANT]
> If a specific part of the system breaks, start your investigation in these primary files.

* **Upload or Status Issues:**  
  Start in `api/routes/rag_documents.py` and check `rag_pipeline/documents.py`.

* **PDF Parsing / Chunking Issues:**  
  Start in `rag_pipeline/ingestion.py`.

* **No Results Found (Retrieval Logic):**  
  Start in `providers/rag/advanced.py`.

* **Document Filter Not Working:**  
  Start in `providers/rag/advanced.py` (check how `doc_filter` is mapped to metadata keys) and `retrieval/chroma_store.py`.

* **Query Returning "NO_MATCH" / Bad Answers:**  
  Start in `live_cycle/graph.py` (specifically `enrich_query` and `generate_assist_response` nodes) and check the prompts in `core/config.py`.

* **ChromaDB / BM25 Issues:**  
  Start in `retrieval/chroma_store.py` or `retrieval/bm25_index.py`.
