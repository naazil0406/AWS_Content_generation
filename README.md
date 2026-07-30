# AI Content Generation Service

A single-responsibility FastAPI service: takes a user prompt, retrieves
context from an existing Knowledge Base, generates structured AI content,
generates an optimized image prompt, and renders exactly three image
variations. Deployable to AWS Lambda via Mangum, or run locally with
uvicorn.

## Architecture

```
User
 -> Amazon API Gateway (HTTP API)
 -> AWS Lambda
 -> Mangum
 -> FastAPI (app/main.py)
 -> POST /api/generate (app/routes/content.py)
    -> KnowledgeBaseService.retrieve(prompt)        [external KB, HTTP]
    -> ContentGenerationEngine.generate()
        -> Content Generation Agent (LLM)           [Bedrock or OpenRouter]
        -> Image Prompt Generation Agent (LLM)      [Bedrock or OpenRouter]
    -> generate_variations(image_prompt, count=3)   [AWS / HF / Pollinations / Freepik]
    -> Response Builder -> JSON
```

Both the LLM and the image renderer are provider-independent: swapping
either is a one-line environment variable change (`LLM_PROVIDER`,
`IMAGE_PROVIDER`) — no code changes.

## Project Structure

```
app/
  main.py                          FastAPI app + Lambda handler (Mangum)
  config.py                        Single source of truth for env vars
  routes/
    content.py                     POST /api/generate
  services/
    knowledge_base_service.py      Thin client to the external Knowledge Base
    content_generation_engine.py   Core business logic
    prompt_builder.py               Loads prompts/*.txt, builds LLM user turns
    llm_service.py                  Provider-agnostic LLM abstraction
    image_generation_service.py    Provider-agnostic image abstraction
    response_builder.py            Assembles the final API response
  schemas/
    content.py                     Pydantic request/response models
prompts/
  content_generation_system.txt    Content Generation Agent system prompt
  image_prompt_system.txt          Image Prompt Generation Agent system prompt
requirements.txt
template.yaml                      AWS SAM deployment template
.env.example
```

## API

### `POST /api/generate`

Request:
```json
{
  "prompt": "Forklift pre-operation checklist",
  "content_type": "Safety / Best Practice Tip"
}
```

`content_type` is one of: `Recall Card`, `AI Image`, `Infographic`,
`Flashcard`, `Scenario`, `Spot the Mistake Challenge`, `Daily Quiz`,
`Fun Fact`, `Reflection Question`, `Safety / Best Practice Tip`,
`Daily Tip`.

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
produce three distinct images rather than three copies. If you'd rather
have three *prompt* variations (one call each) instead of three renders
of one prompt, that's a small change confined to
`content_generation_engine.py` + `image_generation_service.py` — let me
know if you want that instead.

---

## Local Development Guide

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in KNOWLEDGE_BASE_URL and your chosen providers
uvicorn app.main:app --reload --port 8000
```

Test:
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ladder safety", "content_type": "Daily Tip"}'
```

---

## AWS Deployment Guide

### Phase 1 — AWS Preparation

1. **IAM Role**: SAM creates this automatically from `template.yaml`'s
   `Policies` block (grants `bedrock:InvokeModel` / `bedrock:Converse`).
   If you use a different image/LLM provider that doesn't need AWS, you
   can remove that policy block.
2. **Enable Bedrock model access** (if `LLM_PROVIDER=bedrock` or
   `IMAGE_PROVIDER=aws`): in the Bedrock console, request access to the
   Nova model family in your target region.
3. **CloudWatch**: log group is created automatically per Lambda
   function; no manual setup needed. Adjust `LOG_LEVEL` via the
   function's environment variables.
4. **API Gateway**: defined declaratively in `template.yaml` as an
   `AWS::Serverless::HttpApi` — created on deploy.

### Phase 2 — Application Preparation

```bash
pip install --target .aws-sam/build/ContentGenerationFunction -r requirements.txt
```
(SAM does this for you during `sam build` — see Phase 3.)

Set provider credentials as SAM parameters or in a `samconfig.toml` (see
below) rather than committing them to source control.

### Phase 3 — Lambda Deployment

```bash
sam build
sam deploy --guided
```

You'll be prompted for the parameters defined in `template.yaml`:
`KnowledgeBaseUrl`, `KnowledgeBaseApiKey`, `LlmProvider`,
`OpenRouterApiKey`, `ImageProvider`, `HfToken`, `FreepikApiKey`. SAM
writes your answers to `samconfig.toml` for repeat deploys.

Runtime: Python 3.11 (set in `template.yaml`'s `Globals`).
Timeout: 60s (image generation is the long pole — increase if your
provider is slow). Memory: 1024MB (raise if Pillow/huggingface_hub
image decoding needs more headroom).

### Phase 4 — API Gateway

Already wired via the `HttpApi` event in `template.yaml` — `sam deploy`
prints the invoke URL under `Outputs.ApiUrl`. CORS is open (`*`) by
default; restrict `AllowOrigins` in `template.yaml` before production
use.

Test the deployed endpoint:
```bash
curl -X POST "$(aws cloudformation describe-stacks --stack-name <your-stack> \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ladder safety", "content_type": "Daily Tip"}'
```

### Phase 5 — Knowledge Base Integration

Set `KNOWLEDGE_BASE_URL` (and `KNOWLEDGE_BASE_API_KEY` if required) to
your existing Knowledge Base's retrieval endpoint. `knowledge_base_service.py`
expects `POST {url}/retrieve` with `{"query": str, "top_k": int}`,
returning `{"chunks": [{"text": str, "source": str}, ...]}`. If your KB's
contract differs, only `KnowledgeBaseService._build_request` /
`_parse_response` need to change.

### Phase 6 — LLM Integration

Set `LLM_PROVIDER=bedrock` (default) or `LLM_PROVIDER=openrouter`.
For Bedrock, set `CONTENT_MODEL` / `IMAGE_PROMPT_MODEL` (defaults to
Nova Micro / Nova Lite) and ensure the Lambda's IAM role can invoke
them. For OpenRouter, set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.

### Phase 7 — Image Generation

Set `IMAGE_PROVIDER` to `aws` (Bedrock Nova Canvas, default),
`huggingface` (requires `HF_TOKEN`), `pollinations` (no key needed), or
`freepik` (requires `FREEPIK_API_KEY`). `IMAGE_VARIATIONS_COUNT`
controls how many variations are rendered (default 3, per spec).

### Phase 8 — Testing

- **Local FastAPI**: `uvicorn app.main:app --reload` + `curl`/Postman.
- **Local Lambda**: `sam local invoke ContentGenerationFunction -e event.json`
  or `sam local start-api` to hit it over HTTP.
- **API Gateway**: `curl` the deployed `ApiUrl` output.
- **Knowledge Base**: call `KnowledgeBaseService().retrieve("test query")`
  directly in a REPL to confirm the contract matches.
- **LLM**: call `get_content_llm()._call_llm("You are a test.", "Say hi.")`
  directly to confirm credentials/model access.
- **Image Generation**: call `get_image_gen_service().generate_image("a red circle")`
  directly to confirm the provider works before wiring the full pipeline.
- **End-to-End**: `POST /api/generate` locally, then again after deploy.

### Phase 9 — Production Checklist

- [ ] `KNOWLEDGE_BASE_URL`/`KNOWLEDGE_BASE_API_KEY` set to production KB
- [ ] Bedrock model access granted in the deployment region (if used)
- [ ] `IMAGE_PROVIDER` credentials set and verified with a direct test call
- [ ] CORS `AllowOrigins` restricted to your actual frontend origin(s)
- [ ] Lambda timeout/memory sized for your slowest image provider
- [ ] CloudWatch log retention configured (default is "never expire" — set one)
- [ ] Secrets (`OPENROUTER_API_KEY`, `HF_TOKEN`, `FREEPIK_API_KEY`,
      `KNOWLEDGE_BASE_API_KEY`) moved from plain Lambda env vars to AWS
      Secrets Manager or SSM Parameter Store for production
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
