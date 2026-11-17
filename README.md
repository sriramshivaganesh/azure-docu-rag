# Azure Doc RAG – simple chatbot with file upload

React (streaming chat UI) + FastAPI backend.

Flow: **Upload** → text extracted & chunked → original saved in **Azure Blob Storage** → chunks embedded (**Azure OpenAI** embeddings) → indexed in **Azure AI Search** (keyword + vector).
**Chat** → hybrid search in AI Search → top chunks sent to **Azure OpenAI** chat model → answer streamed token-by-token (SSE).

```
azure docu rag/
├─ backend/   main.py, requirements.txt, .env.example
└─ frontend/  Vite + React
```

---

## 1. Create everything in Azure (from scratch)

Use the portal: https://portal.azure.com. Use **one region for everything** (e.g. `East US` or `Sweden Central`; pick one where Azure OpenAI models are available to you). Names below are examples – storage/search/openai names must be globally unique.

### 1.1 Resource group
1. Portal → search **Resource groups** → **Create**.
2. Subscription: your free one. Name: `rg-doc-rag`. Region: `East US`. → **Review + create** → **Create**.

### 1.2 Storage account (saves the uploaded files)
1. Search **Storage accounts** → **Create**.
2. Resource group `rg-doc-rag`, name `docragstore<random digits>` (lowercase, no dashes), same region.
3. Performance: Standard, Redundancy: **LRS** (cheapest). → **Review + create** → **Create**.
4. After deployment → **Go to resource** → left menu **Security + networking → Access keys** → click **Show** on key1 → copy **Connection string**.
   → this is `AZURE_STORAGE_CONNECTION_STRING`.
5. The container `documents` is created automatically by the app (optional: create it yourself under **Data storage → Containers**).

### 1.3 Azure AI Search (the index)
1. Search **AI Search** (Azure AI Search) → **Create**.
2. Resource group `rg-doc-rag`, Service name `doc-rag-search-<unique>`, same region.
3. Pricing tier → **Change Pricing Tier → Free** (50 MB, 3 indexes – enough for this; only one Free service per subscription). → **Review + create** → **Create**.
4. **Go to resource**:
   - **Overview → Url** (e.g. `https://doc-rag-search-xyz.search.windows.net`) → `AZURE_SEARCH_ENDPOINT`
   - **Settings → Keys** → copy the **Primary admin key** → `AZURE_SEARCH_API_KEY`
   - Leave API access control as **API key** (or "Both") – the app uses the key.
5. You do **not** create the index manually; the backend creates the index `rag-docs` (text + vector fields) on first start. You can view it afterwards under **Search management → Indexes** and query it in **Search explorer**.

### 1.4 Azure OpenAI (the LLM + embeddings, via Azure AI Foundry)
1. Search **Azure OpenAI** → **Create** (or "Azure AI Foundry" → create an Azure OpenAI resource).
2. Resource group `rg-doc-rag`, same region, name `doc-rag-openai-<unique>`, pricing **Standard S0**. Network: *All networks*. → **Create**.
   (If you're told access is restricted, submit the form at https://aka.ms/oai/access; on newer subscriptions it works straight away. Free-trial accounts sometimes need to be upgraded to **pay-as-you-go** – your free credit still applies and you are not charged until it is used up.)
3. **Go to resource**:
   - **Resource Management → Keys and Endpoint**: copy **KEY 1** → `AZURE_OPENAI_API_KEY` and **Endpoint** → `AZURE_OPENAI_ENDPOINT`.
4. Click **Go to Azure AI Foundry portal** (or **Explore Azure AI Foundry portal**) → **Deployments** (Shared resources) → **Deploy model → Deploy base model**:
   - Deploy **`gpt-4o-mini`** (chat). Deployment name: `gpt-4o-mini`. Deployment type: Standard (or Global Standard).
     → `AZURE_OPENAI_CHAT_DEPLOYMENT`
   - Deploy **`text-embedding-3-small`** (embeddings). Deployment name: `text-embedding-3-small`.
     → `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
   - If a model isn't offered in your region/quota, pick another chat model (e.g. `gpt-4.1-mini`, `gpt-4o`) and just put *its deployment name* in the env file. The embedding model must produce **1536 dims** (`text-embedding-3-small` or `text-embedding-ada-002`); for `-3-large` change `EMBED_DIMS` in `backend/main.py` to 3072.

### 1.5 What goes in the `.env` file

Create `backend/.env` (copy `.env.example`):

| Variable | Where to copy it from |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | OpenAI resource → Keys and Endpoint → Endpoint (`https://xxx.openai.azure.com/`) |
| `AZURE_OPENAI_API_KEY` | OpenAI resource → Keys and Endpoint → KEY 1 |
| `AZURE_OPENAI_API_VERSION` | leave `2024-10-21` |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | the **deployment name** of your chat model (e.g. `gpt-4o-mini`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | the **deployment name** of your embedding model |
| `AZURE_SEARCH_ENDPOINT` | AI Search → Overview → Url |
| `AZURE_SEARCH_API_KEY` | AI Search → Settings → Keys → Primary admin key |
| `AZURE_SEARCH_INDEX` | leave `rag-docs` (auto-created) |
| `AZURE_STORAGE_CONNECTION_STRING` | Storage account → Access keys → Connection string |
| `AZURE_STORAGE_CONTAINER` | leave `documents` (auto-created) |

> Never commit `.env`. Leave `AZURE_STORAGE_CONNECTION_STRING` empty to skip blob saving.

---

## 2. Run locally (no Docker)

Prereqs: Python 3.10+ and Node 18+.

**Backend** (PowerShell):
```powershell
cd "azure docu rag\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then edit .env with your values
uvicorn main:app --reload --port 8000
```

**Frontend** (second terminal):
```powershell
cd "azure docu rag\frontend"
npm install
npm run dev
```
Open http://localhost:5173 → **Upload file** (pdf / docx / txt / md / csv) → wait for "Indexed …" → ask questions; the answer streams in with the source file names.

## 3. Verify it worked in Azure
- Storage account → Containers → `documents`: your uploaded file is there.
- AI Search → Indexes → `rag-docs` → **Search explorer** → Search: you'll see the chunks (document count grows).
- Chat answers cite `[filename]`.

## 4. Troubleshooting
- **`KeyError: 'AZURE_...'` on start** – `.env` missing or not in `backend/`.
- **404 DeploymentNotFound** – `AZURE_OPENAI_*_DEPLOYMENT` must be the *deployment name*, not the model name; wait a minute after creating a deployment.
- **401/403 from Search** – used the query key or a wrong key; use the **admin** key.
- **Vector dimension error** – embedding model dims ≠ `EMBED_DIMS` (1536). Delete the index (portal → Indexes) after changing it.
- **Scanned PDFs return "No text"** – they're images; this simple app does no OCR.
- Free Search tier is 50 MB total – delete the index if you hit the limit.
- To avoid charges after experimenting: delete the resource group `rg-doc-rag`.

## 5. LangChain usage (backend/main.py)
| Step | LangChain piece |
|---|---|
| Read files | `PyPDFLoader`, `Docx2txtLoader`, `TextLoader` |
| Chunking | `RecursiveCharacterTextSplitter` (1000 / 150 overlap) |
| Embeddings | `AzureOpenAIEmbeddings` |
| Chat + rewrite LLM | `AzureChatOpenAI` |
| Retrieval | custom `AzureHybridRetriever(BaseRetriever)` (hybrid + semantic reranker) |
| Query rewriting / answering | `prompt \| llm \| StrOutputParser` chains (`rewrite_chain`, `answer_chain`), streamed with `.stream()` |

Index creation, Blob upload and delete stay on the Azure SDKs.

## 6. Notes / next steps
Keys are used for simplicity; for production use Managed Identity + Key Vault, add auth to the API, and consider Azure AI Search semantic ranking.
