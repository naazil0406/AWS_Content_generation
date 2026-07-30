"""
Response Builder — assembles the Content Generation Engine's output and
the three generated images into the API's final response shape.
"""

import base64
from typing import List

from app.schemas.content import GeneratedContent, GenerateContentResponse


def build_response(engine_result: dict, images: List[bytes]) -> GenerateContentResponse:
    """Build the final GenerateContentResponse from the engine's
    structured result and the raw image bytes for each variation.
    """
    content = GeneratedContent(
        title=engine_result.get("summary", "")[:80],
        summary=engine_result.get("summary", ""),
        content=engine_result["content_text"],
        hashtags=engine_result.get("tags", []),
        cta="",
        image_prompt=engine_result["image_prompt"],
    )
    encoded_images = [base64.b64encode(img).decode("utf-8") for img in images]
    return GenerateContentResponse(content=content, images=encoded_images)
