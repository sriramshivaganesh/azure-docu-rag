import io
import json
import os
import uuid

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AzureOpenAI
from pydantic import BaseModel

load_dotenv()

# ---------- config ----------
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
EMBED_DIMS = 1536  # text-embedding-3-small / ada-002

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "rag-docs")

STORAGE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "documents")

aoai = AzureOpenAI(api_key=OPENAI_KEY, azure_endpoint=OPENAI_ENDPOINT, api_version=OPENAI_VERSION)
cred = AzureKeyCredential(SEARCH_KEY)
index_client = SearchIndexClient(SEARCH_ENDPOINT, cred)
search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, cred)


def ensure_index():
    """Create the search index (text + vector fields) if it doesn't exist."""
    if INDEX_NAME in [n for n in index_client.list_index_names()]:
        return
    index = SearchIndex(
        name=INDEX_NAME,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBED_DIMS,
                vector_search_profile_name="vec-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="vec-profile", algorithm_configuration_name="hnsw")],
        ),
    )
    index_client.create_index(index)


def blob_container():
    if not STORAGE_CONN:
        return None
    svc = BlobServiceClient.from_connection_string(STORAGE_CONN)
    c = svc.get_container_client(STORAGE_CONTAINER)
    if not c.exists():
        c.create_container()
    return c


# ---------- helpers ----------
def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        from docx import Document

        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    if name.endswith((".txt", ".md", ".csv")):
        return data.decode("utf-8", errors="ignore")
    raise HTTPException(400, "Supported types: .pdf .docx .txt .md .csv")


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 16):
        r = aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=texts[i : i + 16])
        out += [d.embedding for d in r.data]
    return out


# ---------- app ----------
app = FastAPI(title="Azure Doc RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    ensure_index()


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    data = file.file.read()
    text = extract_text(file.filename, data)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "No text could be extracted from the file.")

    # 1) keep the original file in Azure Blob Storage (optional)
    c = blob_container()
    if c:
        c.upload_blob(file.filename, data, overwrite=True)

    # 2) replace any previous version of this file in the index
    old = list(search_client.search("*", filter=f"filename eq '{file.filename}'", select=["id"], top=1000))
    if old:
        search_client.delete_documents([{"id": d["id"]} for d in old])

    # 3) embed + index
    vectors = embed(chunks)
    docs = [
        {
            "id": uuid.uuid4().hex,
            "content": ch,
            "filename": file.filename,
            "chunk_index": i,
            "embedding": v,
        }
        for i, (ch, v) in enumerate(zip(chunks, vectors))
    ]
    search_client.upload_documents(docs)
    return {"filename": file.filename, "chunks": len(docs)}


@app.get("/documents")
def documents():
    r = search_client.search("*", facets=["filename,count:100"], top=0)
    return [{"filename": f["value"], "chunks": f["count"]} for f in r.get_facets().get("filename", [])]


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


SYSTEM = (
    "You are a helpful assistant. Answer ONLY from the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Cite sources like [filename]."
)


@app.post("/chat")
def chat(req: ChatRequest):
    qv = embed([req.question])[0]
    results = search_client.search(
        search_text=req.question,  # keyword (BM25)
        vector_queries=[VectorizedQuery(vector=qv, k_nearest_neighbors=5, fields="embedding")],  # vector
        select=["content", "filename"],
        top=5,
    )
    hits = [{"content": r["content"], "filename": r["filename"]} for r in results]
    context = "\n\n".join(f"[{h['filename']}]\n{h['content']}" for h in hits)

    messages = [{"role": "system", "content": SYSTEM}]
    messages += req.history[-6:]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"})

    def stream():
        sources = sorted({h["filename"] for h in hits})
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        try:
            for ev in aoai.chat.completions.create(
                model=CHAT_DEPLOYMENT, messages=messages, stream=True  # gpt-5 models only allow default temperature
            ):
                if ev.choices and ev.choices[0].delta.content:
                    yield f"data: {json.dumps({'token': ev.choices[0].delta.content})}\n\n"
        except Exception as e:  # surface errors to the UI
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
