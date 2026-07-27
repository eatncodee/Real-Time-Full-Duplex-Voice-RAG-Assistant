# Future Things — Make This Project Extremely Hirable
Target: SDE / Applied AI, backend-focused, 2027 grad

Current state (from repo scan): FastAPI + WebSockets, Sarvam STT, ChromaDB+fastembed, Gemini LLM, Cartesia TTS.
No auth on `/chat` or `/voice`, only a shared API key on `/documents`, CORS wide open (`allow_origins=["*"]`),
no Dockerfile, no automated tests (only manual latency scripts). These are the gaps below address.

## P0 — Security (do these first, they're the "obviously missing" red flags)
- [ ] **Auth**: add JWT-based auth (e.g. `fastapi-users` or hand-rolled with `python-jose`). Protect `/chat`, `/voice`,
      and `/documents` routes with a real user model, not just a shared API key.
- [ ] **Route security**: per-user document namespacing in ChromaDB (metadata filter by `user_id`) so users can't
      see/query each other's uploaded docs.
- [ ] **Lock down CORS**: replace `allow_origins=["*"]` with an explicit allow-list (your deployed frontend domain).
- [ ] **Rate limiting**: `slowapi` or a Redis token-bucket on `/chat` and `/voice` to prevent abuse of paid LLM/TTS APIs.
- [ ] **Secrets hygiene**: move all keys to a secrets manager (or at least confirm `.env` is never committed); rotate
      the `key` env var name in config.py to something less generic like `GEMINI_API_KEY`.
- [ ] **Input validation**: Pydantic request/response models everywhere (some routes may be using raw dicts).

## P1 — Deployability (turns "runs on my machine" into a real deliverable)
- [ ] **Dockerize**: multi-stage Dockerfile (build deps in one stage, slim runtime in the next) + `docker-compose.yml`
      wiring FastAPI + ChromaDB (+ Redis if you add rate limiting/session state).
- [ ] **Deploy it live**: Railway/Render/Fly.io for the API, or a small GPU/CPU VM if latency needs it. A live demo
      link in your README is worth more to recruiters than any bullet point.
- [ ] **CI/CD**: GitHub Actions — lint (ruff), test (pytest), build Docker image, deploy on merge to main.
- [ ] **Environment configs**: proper dev/staging/prod settings via Pydantic `BaseSettings` instead of a plain class.
- [ ] **Structured logging + observability**: replace print statements with `structlog`/`loguru`; add basic metrics
      (request latency, STT/LLM/TTS stage timings you already benchmark) exported via Prometheus or just logged.

## P2 — Engineering rigor (what separates "project" from "production system")
- [ ] **Automated tests**: pytest suite — unit tests for `search_documents()`, `needs_search()` heuristic, VAD
      logic; integration tests for the WebSocket voice flow using `pytest-asyncio` + `httpx`.
- [ ] **Database**: you have `app/database.py` — if it's not already Postgres-backed for user/session data, move off
      SQLite for anything deployed; add Alembic migrations.
- [ ] **Graceful degradation**: fallback path if Gemini/Cartesia/Sarvam APIs are down or rate-limited (queue, retry
      with backoff, or a cheaper fallback model).
- [ ] **Load testing**: `locust` or `k6` script simulating concurrent WebSocket voice sessions — publish the numbers
      (this pairs perfectly with your existing latency benchmarking work).
- [ ] **API documentation**: polish the auto-generated OpenAPI/Swagger docs with proper descriptions and examples.

## P3 — Differentiators (what makes YOU stand out, not just "another RAG demo")
- [ ] **Write up the architecture decisions**: a blog post / README section on why cascading (STT→RAG→LLM→TTS)
      over end-to-end speech models, and the latency tradeoffs you measured at each stage — this is genuinely
      interesting systems content for interviews.
- [ ] **Multi-tenancy**: turn it from a personal demo into a "product" — multiple users, multiple knowledge bases,
      usage quotas.
- [ ] **Streaming everywhere**: confirm true token-streaming from LLM → sentence-chunked TTS (you've already done
      work here) and document the pipeline with a sequence diagram.
- [ ] **Cost tracking**: log token/API cost per conversation — shows product sense, not just engineering.
- [ ] **Evaluation harness**: a small eval set (query → expected doc retrieved) to quantify RAG retrieval quality,
      not just latency. Recruiters doing applied-AI interviews love seeing eval rigor.

## Portfolio polish
- [ ] Record a clean demo video (you already have GIF_demo.gif / Video.mp4 — make sure they're current).
- [ ] README: architecture diagram, live demo link, "what I'd do with more time" section (this file, basically).
- [ ] Pin this repo, add topics/tags on GitHub (fastapi, rag, voice-ai, websockets) for discoverability.
