# FinDecision

Financial text and article sentiment classification using local transformer models, FastAPI, and a clean web UI.

---

## Features

- **Direct Text & URL Extraction**: Analyze raw financial statements or provide an article URL.
- **SSRF-Protected Scraping**: Safe article retrieval via Trafilatura with DNS validation, private IP blocking, streaming size caps, and redirect controls.
- **Three-Class Sentiment**: Predicts positive, negative, and neutral categories with calibrated probability distributions.
- **Sentence-Aware Chunking**: Handles lengthy financial reports beyond the 512-token transformer limit using boundary-aware text chunking and length-weighted aggregation.
- **100% Local CPU Inference**: Optimized for laptops and standard CPUs using PyTorch CPU wheels (`torch+cpu`) with no discrete GPU or paid external APIs required.
- **Embedded Web Frontend**: Fast single-page application served at `http://localhost:8075/`.
- **Interactive OpenAPI Docs**: Complete Swagger UI documentation at `http://localhost:8075/docs`.
- **Testing & Benchmarking Suite**: Complete automated test coverage, performance profiling script, and held-out evaluation pipeline.

---

## Project Structure

```text
financial-article-prediction/
├── app/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── article_service.py        # Safe HTTP fetching, SSRF mitigations, Trafilatura extraction
│   │   ├── chunking.py               # Sentence-aware window chunking and text normalization
│   │   ├── classification_service.py # Sentiment classification, softmax, and weighted probability aggregation
│   │   ├── classification.py         # Backward-compatibility alias for classification_service
│   │   ├── model_service.py          # Hugging Face AutoModel/AutoTokenizer lifecycle & CPU inference
│   │   └── model_services.py         # Backward-compatibility alias for model_service
│   ├── static/
│   │   ├── app.css                   # Responsive dark-theme styling
│   │   ├── app.js                    # Frontend UI state handling, API calls, and chart rendering
│   │   └── index.html                # Single-page application entry point
│   ├── __init__.py
│   ├── config.py                     # Pydantic Settings and environment configuration
│   ├── main.py                       # FastAPI application, routing, lifespan, and concurrency controls
│   └── schemas.py                    # Pydantic request/response validation schemas
├── reports/
│   ├── benchmark.json                # Latency, cold-start, and RSS memory benchmark outputs
│   └── evaluation.json               # Accuracy, F1, and confusion matrix evaluation outputs
├── scripts/
│   ├── benchmark.py                  # Profiles cold-start load, inference latency, and memory usage
│   └── evaluate.py                   # Evaluates model performance on labeled CSV datasets
├── tests/
│   ├── conftest.py                   # Pytest fixtures and mock model configurations
│   ├── test_api.py                   # Integration tests for FastAPI endpoints
│   ├── test_classification.py        # Unit tests for softmax and chunk aggregation logic
│   └── test_security.py              # SSRF defense and URL validation regression tests
├── .env.example                      # Default configuration template
├── .gitignore                        # Git exclusion rules
├── pytest.ini                        # Pytest configuration and pythonpath resolution
├── README.md                         # Project documentation and architectural reference
└── requirements.txt                  # Python dependencies pinned with CPU PyTorch index
```

---

## Architectural & API Flows

### 1. Direct Text Classification Flow (`POST /predict/text`)

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / Browser UI
    participant API as FastAPI Router (/predict/text)
    participant Sem as Asyncio Semaphore (inference_slots)
    participant CS as ClassificationService
    participant Chunk as Text Chunker (split_text)
    participant MS as ModelService (PyTorch CPU)

    User->>API: POST {"text": "Revenue increased by 14%..."}
    API->>API: Validate schema & character limit (max 50k chars)
    API->>Sem: Acquire inference slot (concurrency guard)
    Sem->>CS: run_classification(text)
    CS->>Chunk: split_text(text, chunk_chars=2000, max_chunks=12)
    Chunk-->>CS: List of bounded sentence chunks
    loop For each chunk
        CS->>MS: predict_logits(chunk)
        MS->>MS: Tokenize & forward pass (torch.inference_mode)
        MS-->>CS: Raw logits [-1.2, 0.4, 2.1]
        CS->>CS: Numerically stable softmax & length weighting
    end
    CS->>CS: Aggregate weighted probabilities & determine top class
    CS-->>API: PredictionResponse (class, probabilities, warnings)
    API->>Sem: Release inference slot
    API-->>User: 200 OK JSON response
```

---

### 2. Web Article Retrieval & Classification Flow (`POST /predict/url`)

```mermaid
flowchart TD
    Start["Client sends URL to /predict/url"] --> ValidateScheme["Validate Scheme (http/https only)"]
    ValidateScheme --> CheckCredentials["Reject embedded credentials (user:pass@host)"]
    CheckCredentials --> CheckHostname["Blocklist check (localhost, *.local, cloud metadata)"]
    CheckHostname --> ResolveDNS["DNS resolution via getaddrinfo()"]
    ResolveDNS --> VerifyGlobal["Check IP: Reject private, loopback & link-local ranges"]

    VerifyGlobal --> StreamHTTP["Stream HTTP GET (httpx) with timeout"]
    StreamHTTP --> CheckRedirects{"Is Redirect?"}
    CheckRedirects -- Yes --> IncRedirect["Increment redirect count (max 3)"]
    IncRedirect --> ValidateScheme
    CheckRedirects -- No --> CheckContentType{"Is text/html?"}

    CheckContentType -- No --> RejectMime["422 Error: Non-HTML content rejected"]
    CheckContentType -- Yes --> StreamChunks["Read chunks up to max 5 MB ceiling"]
    StreamChunks --> ExtractBody["Extract main text & title (Trafilatura)"]
    ExtractBody --> MinimumLength{"Text >= 80 chars?"}
    MinimumLength -- No --> RejectEmpty["422 Error: Inaccessible / paywalled content"]
    MinimumLength -- Yes --> AcquireLock["Acquire Asyncio Semaphore"]
    AcquireLock --> RunInference["Run Sentence-Aware Chunking & Inference"]
    RunInference --> ReturnResult["Return 200 OK PredictionResponse with Article Metadata"]
```

---

### 3. Sentence-Aware Chunking & Probability Aggregation

When an article or financial statement exceeds the transformer's token context window:
1. The text is split into chunks of up to `CHUNK_CHARS` (default: 2,000 characters).
2. The chunker searches backward for sentence boundaries (`. `, `; `, or whitespace) in the latter half of the window to prevent mid-sentence fragmentation.
3. Each chunk is scored individually by the model.
4. Chunk probabilities are combined using a character-length weighting heuristic:

$$
\bar{P}(\text{label}) =
\frac{\sum_{i=1}^{N} P_i(\text{label}) \cdot |\text{chunk}_i|}
{\sum_{i=1}^{N} |\text{chunk}_i|}
$$

5. An advisory warning is returned in the API payload informing callers that chunk-averaged scores represent an aggregation heuristic rather than a joint article-level probability.

---

## System Requirements

- **Operating System**: Linux (Arch, Ubuntu, Debian, Fedora), macOS, or Windows WSL2.
- **Python**: Version 3.11 through 3.14.
- **Hardware**:
  - **CPU**: Any modern 64-bit multi-core CPU (tested on 11th Gen Intel Core i3-1125G4).
  - **RAM**: 4 GB minimum (8 GB recommended).
  - **Storage**: ~500 MB for virtual environment and model checkpoint.
  - **GPU**: **None required**. PyTorch is installed via the official CPU index, preventing multi-gigabyte CUDA/Triton bloat.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/financial-article-prediction.git
   cd financial-article-prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > [!NOTE]
   > The `requirements.txt` file specifies `--extra-index-url https://download.pytorch.org/whl/cpu`, ensuring lightweight, CPU-optimized PyTorch wheels are installed directly without large GPU binaries.

4. **Initialize configuration**:
   ```bash
   cp .env.example .env
   ```

5. **Start the application**:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8075
   ```

6. **Access the application**:
   - Web Interface: [http://localhost:8075/](http://localhost:8075/)
   - Swagger API Docs: [http://localhost:8075/docs](http://localhost:8075/docs)
   - OpenAPI Specification: [http://localhost:8075/openapi.json](http://localhost:8075/openapi.json)

---

## API Endpoints Reference

### 1. `GET /health`
Returns server operational status and model readiness.

**Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "9mark9/finbert-minilm-sentiment"
}
```

---

### 2. `GET /model-info`
Provides details on the currently loaded model checkpoint, classification labels, and limitations.

**Response (200 OK):**
```json
{
  "model": "9mark9/finbert-minilm-sentiment",
  "supported_labels": ["negative", "neutral", "positive"],
  "max_input_length": 512,
  "model_loaded": true,
  "inference_local": true,
  "limitations": [
    "Softmax scores are not necessarily calibrated.",
    "Sentiment classification does not predict investment returns.",
    "Long-text probabilities use length-weighted chunk aggregation.",
    "URL extraction may fail on paywalled or dynamic pages."
  ]
}
```

---

### 3. `POST /predict/text`
Analyzes sentiment of raw financial statements or paragraphs.

**Request Body:**
```json
{
  "text": "Quarterly operating margins expanded by 180 basis points, driven by operational efficiencies."
}
```

**Response (200 OK):**
```json
{
  "predicted_class": "positive",
  "probabilities": {
    "positive": 0.9612,
    "neutral": 0.0276,
    "negative": 0.0112
  },
  "input_type": "text",
  "warnings": [],
  "model": "9mark9/finbert-minilm-sentiment",
  "article": null
}
```

---

### 4. `POST /predict/url`
Fetches a financial news article from a public web address and classifies its overall sentiment.

**Request Body:**
```json
{
  "url": "https://www.reuters.com/markets/example-financial-story"
}
```

**Response (200 OK):**
```json
{
  "predicted_class": "neutral",
  "probabilities": {
    "neutral": 0.6841,
    "positive": 0.1834,
    "negative": 0.1325
  },
  "input_type": "url",
  "warnings": [
    "Long text was divided into 3 chunks. The length-weighted score is not a calibrated article-level probability."
  ],
  "model": "9mark9/finbert-minilm-sentiment",
  "article": {
    "title": "Central Banks Hold Rates Steady Amid Inflation Data",
    "source_url": "https://www.reuters.com/markets/example-financial-story"
  }
}
```

---

## Security & SSRF Protection

The article fetcher in `app/services/article_service.py` applies strict security boundaries before initiating HTTP requests:

- **Protocol Whitelist**: Only `http://` and `https://` schemes are accepted; protocols such as `file://`, `ftp://`, and `gopher://` are rejected.
- **No Embedded Credentials**: Rejects URLs containing basic authentication credentials (`http://user:pass@host`).
- **Internal Host Blocklist**: Blocks `localhost`, `*.local`, `metadata.google.internal`, and other internal routing endpoints.
- **DNS IP Validation**: Resolves DNS records prior to connection and discards loopback (`127.0.0.0/8`, `::1`), private RFC 1918 ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), and link-local cloud metadata addresses (`169.254.169.254`).
- **Redirect Re-validation**: Every HTTP 3xx redirect location is re-validated against the full security suite before being followed.
- **Resource Constraints**: Caps downloads at 5 MB and enforces connection (5s) and read (10s) timeouts.

---

## Testing & Benchmarks

### Running Automated Tests
The project includes a comprehensive suite of unit, integration, and security regression tests:

```bash
pytest -q
```

All tests mock the model forward passes so they run completely offline in milliseconds.

### Running CPU Performance Benchmark
To measure cold-start model load time, median inference latency on your CPU, and peak RAM consumption:

```bash
python scripts/benchmark.py
```

The output report is automatically saved to `reports/benchmark.json`.

### Evaluating on Labeled Datasets
To evaluate classification accuracy, macro F1, and per-class precision/recall on a held-out dataset:

```bash
python scripts/evaluate.py --csv path/to/dataset.csv
```
*(The input CSV must contain `text` and `label` columns).*

---

## Limitations

- **Not Financial Advice**: Sentiment classification measures textual sentiment and does not predict future stock performance or investment returns.
- **Softmax Heuristic**: Raw softmax outputs represent relative class distribution within the candidate space, not calibrated empirical likelihoods.
- **Aggregation Heuristic**: Multi-chunk scoring employs length-weighted averaging across individual chunks rather than full document-level contextual attention.
- **Dynamic Content**: Pages relying entirely on client-side JavaScript rendering (SPA) or strict paywalls may fail during Trafilatura article extraction.
