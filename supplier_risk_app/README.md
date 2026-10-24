# AI-Driven Supplier Risk Assessment

Flask application for supplier risk assessment using structured CSV data and uploaded documents. It supports an optional local Ollama model for conversational assessments without an API key; OpenAI remains an optional hosted provider.

## Integrated flow

`Supplier CSV -> supplier profile -> document retrieval -> risk score/level -> procurement action`

Uploaded PDF/TXT documents are chunked into a local JSON vector store. With an OpenAI API key, `/analyze` can use OpenAI embeddings and a chat model. Without a key, it uses a local Ollama model when the Ollama service and configured model are available. Otherwise it performs keyword retrieval and returns an offline, rule-based assessment.

## Run

### Run without a hosted API key

```powershell
# Download the default model on first use and start Ollama.
ollama run llama3.2:3b

# In another PowerShell window, start the Flask app.
cd "C:\path\to\supplier_risk_app"
python app.py
```

Ollama serves the model locally at `http://localhost:11434`. The app defaults to `llama3.2:3b`; override these settings if needed:

```powershell
$env:OLLAMA_MODEL="llama3.2:3b"
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

To use OpenAI instead, set an API key and an available model:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
$env:OPENAI_MODEL="gpt-6-luna"
```

## Endpoints
- `/dashboard` — CSV-backed supplier counts
- `/suppliers` — CSV-backed supplier directory
- `/assistant` — integrated CSV + RAG + LLM assessment
- `/upload` — upload and ingest supplier documents
- `/vector-store` — inspect RAG chunks
- `/api/suppliers` — supplier list JSON
- `/api/supplier/<Supplier_ID>` — supplier record JSON
- `/analyze` — integrated risk analysis API

The included 1,000-row CSV remains the project's structured dataset. It is not automatically labelled as live data.

## Deploy to Render

1. Push this workspace to a GitHub repository. Keep the root `render.yaml` and the `supplier_risk_app` directory together.
2. Confirm that `.gitignore` excludes `.env`, `users.json`, `vector_store.json`, and uploaded documents. Do not commit credentials or private supplier documents.
3. In Render, create a new **Blueprint** and connect the GitHub repository. Render reads `render.yaml`, builds the app, and provisions a persistent disk for user accounts, the document index, and uploads.
4. Set the requested `INITIAL_ADMIN_PASSWORD` in Render. After the first login, the account is stored on the persistent disk; later changes to that initial password setting do not reset it.
5. Open the deployed `onrender.com` URL and sign in as `admin` with the password you supplied.

The Blueprint uses a paid Render web service because persistent disks are required for uploaded files and app-managed data. OpenAI-based assessments are optional and require an `OPENAI_API_KEY` environment variable in Render; without it, the app uses its local rule-based assessment. A local Ollama instance on your computer is not available to the hosted app.
