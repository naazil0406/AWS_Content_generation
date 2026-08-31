"""
Image job processor for SQS-triggered invocations of ContentGenerationFunction
— the SAME Lambda that serves API Gateway requests, not a separate function.
See app/main.py's `handler`, which inspects each invocation's event shape and
routes SQS-shaped events here instead of through Mangum/FastAPI.

Only reached when image_count is 2 or 3 (see app/routes/content.py's
_enqueue_image_jobs()) — a 1-image request always renders synchronously
in the same request/response cycle and never touches this file or SQS at
all. This keeps the common case (1 image) as fast as it always was, and
only pays the "queue + separate invocation" cost when there's genuinely
more than one image to parallelize.

This file does not reimplement or duplicate any image-generation logic:
it calls the exact same get_image_gen_service()/FreepikImageService that
the synchronous 1-image path and /regenerate-image both use. The only
thing new here is the trigger (SQS instead of an HTTP request) and where
the result goes (a deterministic S3 key computed by the caller — see
app/services/image_storage_service.py's image_object_key() — rather than
a URL returned directly in an HTTP response body).

Message body (produced by app/routes/content.py's _enqueue_image_jobs(),
one message per requested image beyond the first): JSON object with
request_id, image_index, image_prompt, negative_prompt, aspect_ratio,
resolution. The image PROMPT is already generated and included — this
file never regenerates content or image prompts, it only renders.

Failure handling: exceptions are NOT caught and swallowed here. Any
failure (Freepik error, S3 error, malformed message) propagates, which
is what tells Lambda/SQS the message was NOT successfully processed —
SQS then makes it visible again for another delivery attempt (up to the
queue's redrive policy maxReceiveCount), and finally routes it to
ImageGenerationDLQ if every attempt fails. This is the queue's own
built-in retry/dead-letter mechanism (configured entirely in
template.yaml) — no application-level job-tracking, retry-counting, or
status database. A failed image simply never appears at its S3 key; the
client's presigned URL (already handed back synchronously before this
job was even queued) will keep 404ing for that one image.
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

    logger.info(
        "Rendering queued image %s (request_id=%s, aspect_ratio=%s, resolution=%s)",
        index, request_id, aspect_ratio, resolution,
    )

    service = get_image_gen_service(aspect_ratio=aspect_ratio, resolution=resolution)
    image_bytes = service.generate_image(prompt, negative_prompt=negative_prompt)
    key = put_generated_image(image_bytes, request_id, index)

    logger.info("Uploaded queued image %s for request_id=%s to s3://%s", index, request_id, key)


def handler(event, context):
    """SQS-record processor, called from app/main.py's `handler` when it
    detects an SQS-shaped event. With BatchSize: 1 (see template.yaml),
    `event["Records"]` normally contains exactly one message, but this
    loops over whatever SQS actually delivers rather than assuming.
    """
    records = event.get("Records", [])
    logger.info("Image job processor invoked with %d record(s)", len(records))
    for record in records:
        body = json.loads(record["body"])
        _process_one(body)