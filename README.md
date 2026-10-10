# LLMKeeper

Local topic-drift guard for LLM conversations. Validates whether messages in a conversation stay within the topic scope defined by a system prompt, using local embeddings with 100% offline processing.

## Features

- **Local embeddings**: Uses `sentence-transformers` with `paraphrase-multilingual-MiniLM-L12-v2` model (384-dim, supports Spanish and other languages, CPU-only).
- **Topic field creation**: Automatically chunks a system prompt (by bullets or sentences) and builds a semantic field (centroid + anchors) representing valid topics.
- **Message validation**: Scores new messages against the topic field using top-k anchor similarity + centroid similarity (configurable `alpha`).
- **Conversational sessions**: Tracks conversation state with rolling averages, consecutive off-topic counter, and patient threshold (tolerance for temporary deviations).
- **Adjustable strictness**: Presets (`muy_estricto`, `estricto`, `moderado`, `permisivo`, `muy_permisivo`) or custom thresholds in [0,1].
- **FastAPI**: Full REST API with 7 endpoints for field and session management.
- **Hexagonal architecture**: Domain, ports, application, adapters, and infrastructure layers strictly separated.

## Installation

```bash
# Clone and install
cd llmkeeper
pip install -e ".[dev]"

# Install torch CPU-only (optional, avoids CUDA dependencies)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### Run the API server

```bash
python main.py
```

Server runs on `http://127.0.0.1:8000`. Visit `/docs` for Swagger UI.

### Create a topic field

```bash
curl -X POST http://127.0.0.1:8000/api/v1/fields \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "You are a cooking expert. Help users with recipes, ingredients, and cooking techniques.",
    "strictness_preset": "moderado"
  }'
```

Response:
```json
{
  "field_id": "e2d8f5b6-...",
  "num_anchors": 3,
  "threshold": 0.55,
  "preset": "moderado"
}
```

### Validate a message

```bash
curl -X POST http://127.0.0.1:8000/api/v1/fields/e2d8f5b6-.../validate \
  -H "Content-Type: application/json" \
  -d '{"text": "What are the best ingredients for paella?"}'
```

Response:
```json
{
  "is_valid": true,
  "score": 0.78,
  "threshold": 0.55,
  "nearest_anchor_text": "You are a cooking expert."
}
```

### Create a session and add messages

```bash
# Create session
curl -X POST http://127.0.0.1:8000/api/v1/fields/e2d8f5b6-.../sessions \
  -H "Content-Type: application/json" \
  -d '{"patience": 3, "rolling_window": 5}'

# Add messages
curl -X POST http://127.0.0.1:8000/api/v1/sessions/session-id/messages \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the capital of France?"}'
```

Session tracks: `consecutive_off` (counter), `rolling_average` (trend), `status` (`on_topic` / `warning` / `off_topic`).

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/fields` | Create topic field from system prompt |
| `GET` | `/api/v1/fields/{field_id}` | Get field metadata |
| `PATCH` | `/api/v1/fields/{field_id}/strictness` | Update strictness level |
| `POST` | `/api/v1/fields/{field_id}/validate` | Validate single message (stateless) |
| `POST` | `/api/v1/fields/{field_id}/sessions` | Create conversation session |
| `POST` | `/api/v1/sessions/{session_id}/messages` | Add message to session |
| `GET` | `/api/v1/sessions/{session_id}` | Get session state |

## Testing

### Run fast tests (no real embedding model)

```bash
pytest tests/ -v -m "not integration"
```

Tests use `ScriptedEmbeddingProvider` (deterministic vectors from a dict) and in-memory repositories for speed.

### Run integration tests (with real SentenceTransformer)

```bash
pytest tests/ -v -m integration
```

Slower, but validates real semantic matching on Spanish examples.

## Configuration

Create a `.env` file in the project root to override settings:

```env
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
DB_PATH=llmkeeper.db
DEFAULT_PATIENCE=3
DEFAULT_ROLLING_WINDOW=5
DEFAULT_ALPHA=0.7
DEFAULT_TOP_K=3
```

## Architecture

```
llmkeeper/
├── domain/
│   ├── models.py              # Embedding, Anchor, TopicField, Strictness, Session, ValidationResult
│   └── services/
│       ├── field_builder.py   # split_into_chunks, build_centroid, build_topic_field
│       └── drift_evaluator.py # score_against_field, append_message_to_session
├── ports/
│   └── outbound.py            # EmbeddingProviderPort, FieldRepositoryPort, SessionRepositoryPort (Protocols)
├── application/
│   └── use_cases.py           # CreateField, ValidateMessage, AdjustStrictness, CreateSession, AppendMessageToSession, GetSession
├── adapters/
│   ├── inbound/
│   │   ├── api.py             # FastAPI router
│   │   └── schemas.py         # Pydantic request/response models
│   └── outbound/
│       ├── embedding_sentence_transformers.py
│       ├── repository_memory.py  # In-memory repositories (suitable for dev/testing)
│       └── repository_sqlite.py  # [Not yet implemented] SQLite repositories
├── infrastructure/
│   ├── config.py              # Settings (pydantic-settings)
│   ├── container.py           # Dependency injection wiring
│   └── app.py                 # FastAPI app factory + lifespan
└── main.py                    # Entry point
```

## Scoring Model

For a message, the score is computed as:

```
score = alpha * mean(top_k_similarities_to_anchors) + (1 - alpha) * centroid_similarity
```

Where:
- `alpha` = 0.7 (default, configurable)
- `top_k` = 3 (default, number of most-similar anchors to average)
- `centroid` = mean of all anchor embeddings

A message is valid if `score >= threshold` (derived from strictness level).

## Session State Machine

```
on_topic (0 off-topic msgs)
    ↓
warning (1 to patience-1 off-topic msgs)
    ↓
off_topic (patience or more off-topic msgs)
```

Once on-topic again, counter resets to 0 and status returns to `on_topic`.

## Future Improvements

- SQLite repository implementation (`repository_sqlite.py`) for persistence
- Batch processing optimizations for large conversations
- Custom clustering of anchors for heterogeneous prompts
- Webhook callbacks for `off_topic` alerts
- Configurable chunking strategies (sliding windows, recursive splitting)
- Support for multimodal embeddings (text + images)

## License

MIT
