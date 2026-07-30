# AI Content Generation Service

A single-responsibility FastAPI service: takes a user prompt, retrieves
context from an Amazon Bedrock Knowledge Base, generates structured AI
content, generates an optimized image prompt, and renders exactly three
image variations. Deployable to AWS Lambda via Mangum, or run locally
with uvicorn. Ships with a bundled static frontend served from the same
app.

## Architecture

```
User (browser)
 -> GET /                                        [frontend/index.html]
 -> Amazon API Gateway (HTTP API)
 -> AWS Lambda
 -> Mangum
 -> FastAPI (app/main.py)
 -> GET  /api/content-types (app/routes/content.py)
 -> POST /api/generate      (app/routes/content.py)
    -> KnowledgeBaseService.retrieve(prompt)        [Bedrock Knowledge Base, via boto3]
    -> ContentGenerationEngine.generate()
        -> Content Generation Agent (LLM)           [Bedrock or OpenRouter]
        -> Image Prompt Generation Agent (LLM)      [Bedrock or OpenRouter]
    -> generate_variations(image_prompt, count=3)   [AWS / HF / Pollinations / Freepik]
    -> Response Builder -> JSON
```

Both the LLM and the image renderer are provider-independent: swapping
either is a one-line environment variable change (`LLM_PROVIDER`,
`IMAGE_PROVIDER`) — no code changes.

The frontend and the API are served from the **same Lambda / same
origin**, so `frontend/index.html`'s `fetch("/api/...")` calls work with
zero CORS configuration. See "Frontend" below.

## Project Structure

```
app/
  main.py                          FastAPI app + Lambda handler (Mangum) + serves frontend/index.html at "/"
  config.py                        Single source of truth for env vars
  routes/
    content.py                     GET /api/content-types, POST /api/generate
  services/
    knowledge_base_service.py      Bedrock Knowledge Base client (bedrock-agent-runtime, via boto3)
    content_generation_engine.py   Core business logic
    prompt_builder.py               Loads prompts/*.txt, builds LLM user turns
    llm_service.py                  Provider-agnostic LLM abstraction
    image_generation_service.py    Provider-agnostic image abstraction
    response_builder.py            Assembles the final API response
  schemas/
    content.py                     Pydantic request/response models + CONTENT_TYPES list
frontend/
  index.html                       Static single-page frontend, served at GET /
prompts/
  content_generation_system.txt    Content Generation Agent system prompt
  image_prompt_system.txt          Image Prompt Generation Agent system prompt
requirements.txt
template.yaml                      AWS SAM deployment template
.env.example
```

## API

### `GET /api/content-types`

Returns the content types the frontend's dropdown populates itself
from (sourced from `app/schemas/content.py`'s `CONTENT_TYPES`, which
mirrors the table in `prompts/image_prompt_system.txt`):
```json
{
  "content_types": [
    "Recall Card", "AI Image", "Infographic", "Flashcard", "Scenario",
    "Spot the Mistake Challenge", "Daily Quiz", "Fun Fact",
    "Reflection Question", "Safety / Best Practice Tip", "Daily Tip"
  ]
}
```

### `POST /api/generate`

Request:
```json
{
  "prompt": "Forklift pre-operation checklist",
  "content_type": "Safety / Best Practice Tip"
}
```

Response:
```json
{
  "content": {
    "title": "",
    "summary": "",
    "content": "...",
    "hashtags": ["..."],
    "cta": "",
    "image_prompt": "..."
  },
  "images": ["<base64>", "<base64>", "<base64>"]
}
```

### `GET /api/health`

Returns `{"status": "ok"}` — used for Lambda warm-up checks / uptime monitors.

### `GET /`

Serves `frontend/index.html` directly (same origin as the API above, so
its relative `fetch("/api/...")` calls resolve correctly with no CORS
setup required).

## Design note: Daily Tip focus, avoid_repeating, negative-prompt fallback

Three behaviors from the original codebase are carried over into the
engine, since `prompts/content_generation_system.txt` and
`prompts/image_prompt_system.txt` are kept unchanged and reference them:

- **Daily Tip Focus rotation**: when `content_type == "Daily Tip"`, the
  engine randomly picks one State or Error from the framework
  (`prompt_builder.pick_daily_tip_focus()`) and includes the required
  `Daily Tip Focus (...)` line in the prompt, then retries generation up
  to 3 times if the model misses the 100-180 word range the system
  prompt specifies.
- **`avoid_repeating`**: `GenerateContentRequest.avoid_repeating` is an
  optional list of previously-generated text for the same
  `(content_type, prompt)` pair. If your caller keeps its own generation
  history, pass it here to steer the model away from repeating itself —
  the service itself is stateless and keeps no history.
- **Negative-prompt fallback**: if the Image Prompt Generation Agent's
  JSON response comes back with an empty `negative_prompt`, the engine
  falls back to a fixed, mode-specific negative prompt
  (`prompt_builder.negative_prompt_for_mode()`) rather than sending the
  image renderer nothing.

## Design note: how "exactly three images" is implemented

The spec asks for one optimized image prompt and three image variations.
This service generates **one** prompt via the Image Prompt Generation
Agent, then calls the configured image provider **three times** with
that same prompt. Every supported provider is stochastic, so three calls
produce three distinct images rather than three copies.

## Frontend

`frontend/index.html` is a plain HTML/CSS/JS single-page app — no build
step, no npm, nothing to compile. It's served directly by FastAPI at
`GET /` (see `app/main.py`) and bundled into the Lambda package
automatically, since `template.yaml`'s `CodeUri: .` packages the whole
repo root.

The page's JS uses `const API = "";` — every `fetch` call is a relative
path (e.g. `fetch("/api/content-types")`). That only resolves correctly
when the page and the API share an origin, which is exactly what serving
it from the same FastAPI app guarantees. If you ever move the frontend
to a separate static host (S3/CloudFront), you'd need to either set
`API` to the full API Gateway URL or route both through one CloudFront
distribution.

---

## Local Development Guide

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in KNOWLEDGE_BASE_ID and your chosen providers
uvicorn app.main:app --reload --port 8000
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # fill in KNOWLEDGE_BASE_ID and your chosen providers
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/` in a browser to see the frontend, or test
the API directly:
```bash
curl http://127.0.0.1:8000/api/content-types

curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ladder safety", "content_type": "Daily Tip"}'
```

---

## AWS Deployment Guide (SAM CLI, no Docker, no git required)

`sam build`/`sam deploy` only need Docker if you explicitly pass
`--use-container` (for compiling packages that need a Linux build
environment). None of this project's dependencies require that, so
plain `sam build` works directly against your local Python install.
Git is not involved anywhere in this flow — SAM zips your local folder
directly.

### Phase 1 — Install prerequisites (Windows)

```powershell
winget install Amazon.AWSCLI
winget install Amazon.SAM-CLI
```
If `sam --version` isn't recognized after installing, open a **brand
new** terminal window (PATH changes don't apply to already-open ones).
If it's still not found, download the MSI directly from
https://github.com/aws/aws-sam-cli/releases/latest and run it — this is
the most reliable install path on Windows.

Then configure AWS credentials:
```powershell
aws configure
```

### Phase 2 — AWS Preparation

1. **IAM Role**: SAM creates this automatically from `template.yaml`'s
   `Policies` block — grants `bedrock:InvokeModel` / `bedrock:Converse`
   plus `bedrock:Retrieve` scoped to your specific Knowledge Base ARN.
2. **Enable Bedrock model access** (if `LLM_PROVIDER=bedrock` or
   `IMAGE_PROVIDER=aws`): in the Bedrock console, request access to the
   Nova model family in your target region. This is a one-time,
   manual, account-level step — SAM can't do it for you.
3. **CloudWatch**: log group is created automatically per Lambda
   function; no manual setup needed. Adjust `LOG_LEVEL` via the
   function's environment variables.
4. **API Gateway**: defined declaratively in `template.yaml` as an
   `AWS::Serverless::HttpApi` — created on deploy.

### Phase 3 — Build

```powershell
sam build
```
This installs `requirements.txt` locally into `.aws-sam/build/` and
copies in `app/`, `frontend/`, and `prompts/`. No Docker needed for this
project's dependencies.

### Phase 4 — Deploy (first time — guided)

```powershell
sam deploy --guided
```

You'll be prompted for:
- **Stack Name** — e.g. `ai-content-generation-service`
- **AWS Region** — e.g. `us-east-1`
- `KnowledgeBaseId` — your Bedrock Knowledge Base's **ID**, found in the
  Bedrock console under Knowledge Bases -> your KB -> Knowledge base
  overview. **Not a URL.**
- `KnowledgeBaseRegion` — e.g. `us-east-1`
- `LlmProvider` — `bedrock` (default) or `openrouter`
- `OpenRouterApiKey` — leave blank if using bedrock
- `ImageProvider` — `aws` (default), `huggingface`, `pollinations`, or `freepik`
- `HfToken` / `FreepikApiKey` — only if using those providers
- **Confirm changes before deploy** → `Y`
- **Allow SAM CLI IAM role creation** → `Y`
- **Save arguments to samconfig.toml** → `Y`

SAM writes your answers to `samconfig.toml`, so every deploy after this
is just:
```powershell
sam build
sam deploy
```

Runtime: Python 3.11 (set in `template.yaml`'s `Globals`).
Timeout: 60s (image generation is the long pole — increase if your
provider is slow). Memory: 1024MB (raise if Pillow/huggingface_hub
image decoding needs more headroom).

### Phase 5 — API Gateway + Frontend

Already wired via the `HttpApi` events in `template.yaml` — one for
`/{proxy+}` (all API routes) and one for the bare `/` (the frontend
page, which `{proxy+}` alone would not match). `sam deploy` prints the
invoke URL under `Outputs.ApiUrl`. CORS is open (`*`) by default;
restrict `AllowOrigins` in `template.yaml` before production use.

Test the deployed endpoint:
```powershell
$ApiUrl = aws cloudformation describe-stacks --stack-name ai-content-generation-service `
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text

curl "$ApiUrl/api/health"
curl "$ApiUrl/api/content-types"
```
Open `$ApiUrl/` directly in a browser to see the frontend live.

### Phase 6 — Knowledge Base Integration

Set `KNOWLEDGE_BASE_ID` (and `KNOWLEDGE_BASE_REGION`) to your existing
Amazon Bedrock Knowledge Base. `knowledge_base_service.py` calls
`bedrock-agent-runtime`'s `retrieve` API via boto3 — no HTTP endpoint
or URL involved. If you ever swap to a different, externally hosted
Knowledge Base with its own REST contract instead, only
`KnowledgeBaseService.retrieve` / `_parse_response` need to change.

### Phase 7 — LLM Integration

Set `LLM_PROVIDER=bedrock` (default) or `LLM_PROVIDER=openrouter`.
For Bedrock, set `CONTENT_MODEL` / `IMAGE_PROMPT_MODEL` (defaults to
Nova Micro / Nova Lite) and ensure the Lambda's IAM role can invoke
them. For OpenRouter, set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.

### Phase 8 — Image Generation

Set `IMAGE_PROVIDER` to `aws` (Bedrock Nova Canvas, default),
`huggingface` (requires `HF_TOKEN`), `pollinations` (no key needed), or
`freepik` (requires `FREEPIK_API_KEY`). `IMAGE_VARIATIONS_COUNT`
controls how many variations are rendered (default 3, per spec).

### Phase 9 — Testing

- **Local FastAPI**: `uvicorn app.main:app --reload` + browser at `/`, or `curl`/Postman against `/api/*`.
- **Local Lambda**: `sam local invoke ContentGenerationFunction -e event.json`
  or `sam local start-api` to hit it over HTTP (this does use Docker —
  skip it if you're avoiding Docker entirely and test against the
  deployed `ApiUrl` instead).
- **API Gateway**: `curl` the deployed `ApiUrl` output.
- **Knowledge Base**: call `KnowledgeBaseService().retrieve("test query")`
  directly in a REPL to confirm Bedrock access and IAM permissions.
- **LLM**: call `get_content_llm()._call_llm("You are a test.", "Say hi.")`
  directly to confirm credentials/model access.
- **Image Generation**: call `get_image_gen_service().generate_image("a red circle")`
  directly to confirm the provider works before wiring the full pipeline.
- **End-to-End**: open `/` locally, generate content through the UI, then repeat against the deployed `ApiUrl`.

### Phase 10 — Production Checklist

- [ ] `KNOWLEDGE_BASE_ID`/`KNOWLEDGE_BASE_REGION` set to production KB
- [ ] Bedrock model access granted in the deployment region (if used)
- [ ] `bedrock:Retrieve` IAM permission scoped to the correct KB ARN
- [ ] `IMAGE_PROVIDER` credentials set and verified with a direct test call
- [ ] CORS `AllowOrigins` restricted to your actual frontend origin(s) if serving the frontend separately
- [ ] Lambda timeout/memory sized for your slowest image provider
- [ ] CloudWatch log retention configured (default is "never expire" — set one)
- [ ] Secrets (`OPENROUTER_API_KEY`, `HF_TOKEN`, `FREEPIK_API_KEY`) moved
      from plain Lambda env vars to AWS Secrets Manager or SSM Parameter
      Store for production
- [ ] Concurrency: set a reserved/provisioned concurrency limit if you
      need to protect downstream provider rate limits
- [ ] Cost: Lambda cost is dominated by image-provider latency (up to
      the 60s timeout) — Nova Canvas/Bedrock calls are billed per image;
      Pollinations is free but rate-limited; size `IMAGE_VARIATIONS_COUNT`
      and timeout accordingly

## Cost Optimization

- Nova Micro (`CONTENT_MODEL`) and Nova Lite (`IMAGE_PROMPT_MODEL`) are
  the cheapest Bedrock text models suited to this workload — avoid
  upsizing to Nova Pro/Claude unless content quality genuinely requires it.
- `IMAGE_VARIATIONS_COUNT=3` means 3x the image-provider cost per
  request; if cost matters more than choice, drop to 1-2.
- Pollinations is free but has no SLA — use it for dev/staging, a paid
  provider for production.
- Lambda memory above ~1024MB rarely helps this workload (I/O-bound, not
  CPU-bound) — don't over-provision.

## Future Scalability Recommendations

- Add response caching (e.g. keyed on `prompt + content_type`) in front
  of the Knowledge Base call if the same topics are requested repeatedly.
- If image generation latency becomes a bottleneck, move image rendering
  to an async pattern (API returns a job ID immediately, images generate
  in a background Lambda invoked via SQS/EventBridge, client polls or is
  notified) rather than blocking one Lambda invocation on 3 sequential
  renders.
- Add per-provider circuit breaking if you support multiple image
  providers and want automatic failover rather than a hard failure.
- The current frontend calls a few endpoints (generation history,
  version selection, image editing) that aren't implemented in
  `app/routes/content.py` yet — only `/api/generate`, `/api/content-types`,
  and `/api/health` exist today. Build those out if the full frontend
  experience (pick-an-image, edit, version history) is needed, or trim
  the frontend down to match the current single-shot API.