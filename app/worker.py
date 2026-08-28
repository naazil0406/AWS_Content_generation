"""
Image Worker — SQS-triggered image generation logic, invoked from
WITHIN ContentGenerationFunction (app/main.py's `handler`) rather than
as a separate Lambda function. Renders ONE image per invocation via the
EXISTING Freepik/AWS/Pollinations integration
(app/services/image_generation_service.py), unchanged.

This file does not reimplement or duplicate any image-generation logic:
it calls the exact same get_image_gen_service()/FreepikImageService that
the synchronous /regenerate-image endpoint (app/routes/content.py) and
the pre-SQS code path both used. The only thing new here is the trigger
(SQS instead of an HTTP request) and where the result goes (a
deterministic S3 key computed by the caller — see
app/services/image_storage_service.py's image_object_key() — rather than
a URL returned in an HTTP response body).

Wiring (see template.yaml): ContentGenerationFunction has TWO event
sources now — the existing HttpApi events, and a new SQS event
(ImageGenerationQueue, BatchSize: 1). app/main.py's `handler` inspects
each invocation's event shape and routes SQS-shaped events here
(handler() below) instead of through Mangum/FastAPI. There is no
separate "ImageWorkerFunction" resource — merged into
ContentGenerationFunction by explicit choice; see app/main.py's handler
docstring for the concurrency trade-off this implies.

    ImageGenerationQueue (SQS, BatchSize: 1)
        -> ContentGenerationFunction (app.main.handler, routed here)
        -> get_image_gen_service() [existing, unchanged]
        -> image_storage_service.put_generated_image() [existing S3 key
           convention, reused — never a second/parallel upload path]

Message body (produced by app/routes/content.py's _enqueue_image_jobs(),
one message per requested image — see that function's docstring for the
exact schema): JSON object with request_id, image_index, image_prompt,
negative_prompt, aspect_ratio, resolution, model (informational only —
see NOTE below).

Failure handling: this handler does NOT catch and swallow exceptions.
Any failure (Freepik error, S3 error, malformed message) is allowed to
propagate, which is what tells Lambda/SQS the message was NOT
successfully processed — SQS then makes it visible again for another
delivery attempt (up to the queue's redrive policy maxReceiveCount),
and finally routes it to ImageGenerationDLQ if every attempt fails. This
is the queue's own built-in retry/dead-letter mechanism (configured
entirely in template.yaml) — no application-level job-tracking,
retry-counting, or status database is implemented here, matching the
explicit "no DynamoDB, no job tracking" requirement this worker was
built under. A failed image simply never appears at its S3 key; the
client's presigned URL (already handed back synchronously by the main
Lambda before this job was even queued) will 404 indefinitely for that
one image, same as if the whole request had failed outright.
"""

import json
import logging

from app.services.image_generation_service import get_image_gen_service
from app.services.image_storage_service import put_generated_image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _process_one(body: dict) -> None:
    request_id = body["request_id"]
    index = body["image_index"]
    prompt = body["image_prompt"]
    negative_prompt = body.get("negative_prompt", "")
    aspect_ratio = body.get("aspect_ratio")
    resolution = body.get("resolution")
    # NOTE on "model": Freepik model selection (FREEPIK_MODEL) is a
    # per-DEPLOYMENT setting today, not a per-request one — there is no
    # frontend control for it (see app/schemas/content.py's
    # GenerateContentRequest, which has no image_model field). The value
    # in the message is what the main Lambda's settings.FREEPIK_MODEL
    # was AT ENQUEUE TIME, included purely for traceability/debugging
    # (e.g. if you change FREEPIK_MODEL between when a job is queued and
    # when this worker picks it up, you can see the mismatch in logs).
    # The model actually used to render is always THIS function's own
    # settings.FREEPIK_MODEL (read inside get_image_gen_service() below)
    # — that's the existing, unchanged mechanism; this worker does not
    # accept a per-message model override, since doing so would mean
    # reimplementing model selection outside the existing config system.
    enqueued_model = body.get("model")

    logger.info(
        "Rendering image %s (request_id=%s, aspect_ratio=%s, resolution=%s, "
        "model at enqueue time=%s)",
        index, request_id, aspect_ratio, resolution, enqueued_model,
    )

    service = get_image_gen_service(aspect_ratio=aspect_ratio, resolution=resolution)
    image_bytes = service.generate_image(prompt, negative_prompt=negative_prompt)
    key = put_generated_image(image_bytes, request_id, index)

    logger.info("Uploaded image %s for request_id=%s to s3://%s", index, request_id, key)


def handler(event, context):
    """SQS-record processor, called from app/main.py's `handler` when it
    detects an SQS-shaped event (not a direct Lambda entry point itself
    anymore — see main.py's docstring for the merged-Lambda routing).
    With BatchSize: 1 (see template.yaml), `event["Records"]` will
    always contain exactly one message in practice, but this loops over
    whatever SQS actually delivers rather than assuming — if BatchSize
    is ever changed, this still processes every record in the batch, one
    at a time, letting the first exception abort the batch (SQS's
    default at-least-once behavior: the whole batch is retried since no
    partial-batch-failure reporting is configured — fine at BatchSize 1,
    where "the batch" IS "the one message").
    """
    records = event.get("Records", [])
    logger.info("Image worker invoked with %d record(s)", len(records))
    for record in records:
        body = json.loads(record["body"])
        _process_one(body)