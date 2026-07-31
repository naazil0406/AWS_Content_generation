"""
Response Builder — assembles the Content Generation Engine's output and
the generated images' S3 URLs into the API's final response shape.
"""

from typing import List

from app.schemas.content import GeneratedContent, GenerateContentResponse


def build_response(engine_result: dict, image_urls: List[str]) -> GenerateContentResponse:
    """Build the final GenerateContentResponse from the engine's
    structured result and the presigned S3 URLs for each image variation.
    """
    content = GeneratedContent(
        title=engine_result.get("summary", "")[:80],
        summary=engine_result.get("summary", ""),
        content=engine_result["content_text"],
        hashtags=engine_result.get("tags", []),
        cta="",
        image_prompt=engine_result["image_prompt"],
    )
    return GenerateContentResponse(content=content, images=image_urls)