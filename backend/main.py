import json
import os
import tempfile
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
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
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
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict

load_dotenv()

# ---------- chat log file (one JSON line per question, no database) ----------
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
chat_logger = logging.getLogger("chat_log")
chat_logger.setLevel(logging.INFO)
chat_logger.propagate = False
if not chat_logger.handlers:
    _h = RotatingFileHandler(LOG_DIR / "chat.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(message)s"))
    chat_logger.addHandler(_h)


def log_chat(**fields):
    fields = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    chat_logger.info(json.dumps(fields, ensure_ascii=False))


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

# ---------- LangChain models (no temperature set: gpt-5 models only allow the default) ----------
llm = AzureChatOpenAI(
    azure_deployment=CHAT_DEPLOYMENT,
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version=OPENAI_VERSION,
)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBED_DEPLOYMENT,
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version=OPENAI_VERSION,
)

# ---------- Azure AI Search (index management stays on the Azure SDK) ----------
cred = AzureKeyCredential(SEARCH_KEY)
index_client = SearchIndexClient(SEARCH_ENDPOINT, cred)
search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, cred)

SEMANTIC_CONFIG = "default-semantic"
semantic_search = SemanticSearch(
    configurations=[
        SemanticConfiguration(
            name=SEMANTIC_CONFIG,
            prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="content")]),
        )
    ]
)


def ensure_index():
    """Create the search index (text + vector + semantic config) or add the semantic config to an old index."""
    if INDEX_NAME in [n for n in index_client.list_index_names()]:
        existing = index_client.get_index(INDEX_NAME)
        if not existing.semantic_search or not existing.semantic_search.configurations:
            existing.semantic_search = semantic_search  # needed for reranking; keeps your data
            index_client.create_or_update_index(existing)
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
        semantic_search=semantic_search,
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


# ---------- LangChain: loading + splitting ----------
LOADERS = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader, ".md": TextLoader, ".csv": TextLoader}
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def load_and_split(filename: str, data: bytes) -> list[Document]:
    """Document loader (by file type) -> RecursiveCharacterTextSplitter -> chunks."""
    ext = os.path.splitext(filename.lower())[1]
    if ext not in LOADERS:
        raise HTTPException(400, "Supported types: .pdf .docx .txt .md .csv")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:  # loaders read from a path
        tmp.write(data)
        path = tmp.name
    try:
        kwargs = {"encoding": "utf-8", "autodetect_encoding": True} if LOADERS[ext] is TextLoader else {}
        docs = LOADERS[ext](path, **kwargs).load()
    finally:
        os.remove(path)
    return [d for d in splitter.split_documents(docs) if d.page_content.strip()]


# ---------- LangChain retriever: hybrid search + Azure semantic reranker ----------
class AzureHybridRetriever(BaseRetriever):
    """Keyword + vector search -> Azure semantic reranker -> top k. Falls back to plain hybrid."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    k: int = 5
    candidates: int = 30

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        qv = embeddings.embed_query(query)
        common = dict(
            search_text=query,  # keyword (BM25)
            vector_queries=[VectorizedQuery(vector=qv, k_nearest_neighbors=self.candidates, fields="embedding")],
            select=["content", "filename"],
        )
        try:
            results = search_client.search(
                **common, query_type="semantic", semantic_configuration_name=SEMANTIC_CONFIG, top=self.k
            )
            return [
                Document(
                    page_content=r["content"],
                    metadata={"filename": r["filename"], "score": r.get("@search.reranker_score")},
                )
                for r in results
            ]
        except Exception:
            results = search_client.search(**common, top=self.k)
            return [
                Document(page_content=r["content"], metadata={"filename": r["filename"], "score": None})
                for r in results
            ]


retriever = AzureHybridRetriever()

# ---------- LangChain chains (prompt | llm | parser) ----------
rewrite_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user's latest question as a single standalone search query. "
                "Use the chat history to resolve pronouns and references (e.g. 'it', 'that', 'the second one'), "
                "and add useful keywords. Do not answer. Output ONLY the rewritten query.",
            ),
            MessagesPlaceholder("history"),
            ("human", "Latest question: {question}"),
        ]
    )
    | llm
    | StrOutputParser()
)

answer_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer ONLY from the provided context. "
                "If the answer is not in the context, say you don't know. "
                "Cite sources like [filename].",
            ),
            MessagesPlaceholder("history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    | llm
    | StrOutputParser()
)


def rewrite_query(question: str, history: list[dict]) -> str:
    """QUERY REWRITING: same chat LLM turns a follow-up into a standalone query."""
    try:
        return rewrite_chain.invoke({"history": history[-6:], "question": question}).strip() or question
    except Exception:
        return question  # never block the chat because rewriting failed


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['filename']}]\n{d.page_content}" for d in docs)


# ---------- app ----------
app = FastAPI(title="Azure Doc RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    ensure_index()


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    data = file.file.read()
    chunks = load_and_split(file.filename, data)
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

    # 3) embed (LangChain batches the calls) + index
    vectors = embeddings.embed_documents([ch.page_content for ch in chunks])
    docs = [
        {
            "id": uuid.uuid4().hex,
            "content": ch.page_content,
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


@app.post("/chat")
def chat(req: ChatRequest):
    query = rewrite_query(req.question, req.history)
    docs = retriever.invoke(query)

    def stream():
        sources = sorted({d.metadata["filename"] for d in docs})
        yield f"data: {json.dumps({'sources': sources, 'query': query})}\n\n"
        answer, error = [], None
        try:
            for token in answer_chain.stream(
                {"history": req.history[-6:], "context": format_context(docs), "question": req.question}
            ):
                answer.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:  # surface errors to the UI
            error = str(e)
            yield f"data: {json.dumps({'error': error})}\n\n"
        finally:  # also runs if the client disconnects mid-stream
            log_chat(
                question=req.question,
                rewritten_query=query,
                answer="".join(answer),
                sources=sources,
                error=error,
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
