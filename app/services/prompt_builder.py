"""
Prompt Builder — loads the system prompts from prompts/*.txt and
assembles the dynamic (user-turn) half of each LLM call.

System prompts are never hardcoded in Python: each lives as its own .txt
file so it can be edited by a prompt engineer without touching code. This
module owns loading those files and formatting retrieved Knowledge Base
context into the shape both system prompts expect.
"""

import os
import random
from functools import lru_cache
from typing import List, Optional, Tuple

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompts")

_CONTENT_GENERATION_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "content_generation_system.txt")
_IMAGE_PROMPT_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "image_prompt_system.txt")

# Content types whose most effective visualization is a real photographic
# moment ("scene") rather than a flat explainer graphic ("infographic").
# See prompts/image_prompt_system.txt's CONTENT TYPE -> OUTPUT MODE MAPPING.
_SCENE_MODE_CONTENT_TYPES = {"Scenario", "Spot the Mistake Challenge", "AI Image"}

# Must stay in sync with the "STATES AND ERRORS REFERENCE FRAMEWORK"
# section of prompts/content_generation_system.txt — same four States,
# same four Errors, same wording. Every Daily Tip is anchored to exactly
# one of these eight, chosen at random per request, so the model is
# always given a concrete human-performance lens to build the tip
# around rather than picking one freely.
_STATES = ["Rushing", "Frustration", "Fatigue", "Complacency"]
_ERRORS = ["Eyes not on task", "Mind not on task", "Line of fire", "Balance, traction, grip"]
_DAILY_TIP_FOCUS_ROTATION: List[Tuple[str, str]] = [("State", s) for s in _STATES] + [
    ("Error", e) for e in _ERRORS
]

# Negative-prompt fallbacks, used only when the Image Prompt Generation
# Agent's JSON response comes back with an empty negative_prompt field.
# A "scene" image needs photorealism/cinematic lighting, so the
# infographic negative prompt (which bans exactly those things) would
# actively fight against Scenario/Spot-the-Mistake images ever looking
# realistic — hence two separate fallbacks, not one shared default.
_INFOGRAPHIC_NEGATIVE_PROMPT = (
    "photorealistic, realistic photo, photograph, cinematic lighting, "
    "realistic human faces, realistic skin texture, depth of field, "
    "camera bokeh, film grain, 3D render of real people, blurry, low detail, "
    "watermarks, logos"
)
_SCENE_NEGATIVE_PROMPT = (
    "infographic, icon, iconographic illustration, flat vector illustration, "
    "diagram, chart, flowchart, numbered steps, labeled boxes, comparison "
    "columns, collage, grid of panels, cartoon, clip art, text, captions, "
    "titles, labels, watermarks, logos, low quality, blurry, distorted "
    "anatomy, extra limbs"
)


@lru_cache(maxsize=1)
def load_content_generation_system_prompt() -> str:
    with open(_CONTENT_GENERATION_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=1)
def load_image_prompt_system_prompt() -> str:
    with open(_IMAGE_PROMPT_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def format_context(chunks: List[dict]) -> str:
    """Format Knowledge Base chunks for inclusion in a prompt."""
    if not chunks:
        return "No relevant context found."

    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "")
        lines = [f"[{i}]"]
        if source:
            lines.append(f"Source: {source}")
        lines.append(chunk.get("text", "").strip())
        formatted.append("\n".join(lines))
    return "\n\n---\n\n".join(formatted)


def image_mode_for_content_type(content_type: str) -> str:
    return "scene" if content_type in _SCENE_MODE_CONTENT_TYPES else "infographic"


def pick_daily_tip_focus() -> Tuple[str, str]:
    """Pick one random (kind, name) pair — e.g. ("State", "Rushing") or
    ("Error", "Eyes not on task") — from the 8-item States/Errors
    rotation. Called once per Daily Tip generation."""
    return random.choice(_DAILY_TIP_FOCUS_ROTATION)


def negative_prompt_for_mode(mode: str) -> str:
    """Fallback negative prompt used only when the LLM's own
    negative_prompt came back empty."""
    return _SCENE_NEGATIVE_PROMPT if mode == "scene" else _INFOGRAPHIC_NEGATIVE_PROMPT


def build_content_generation_prompt(
    content_type: str,
    topic: str,
    context_chunks: List[dict],
    monthly_topic_content: Optional[str] = None,
    common_data: Optional[str] = None,
    web_results: Optional[str] = None,
    daily_tip_focus: Optional[Tuple[str, str]] = None,
    avoid_repeating: Optional[List[str]] = None,
) -> str:
    """Build the user-turn prompt for the Content Generation Agent.

    daily_tip_focus: only meaningful when content_type == "Daily Tip" — a
      ("State", "Rushing") or ("Error", "Eyes not on task") pair (see
      pick_daily_tip_focus()). Required by prompts/content_generation_system.txt's
      "DAILY TIP" section, which states every Daily Tip request includes a
      "Daily Tip Focus" line.
    avoid_repeating: previously-generated content for this same
      (content_type, topic) pair, so the model picks a different
      angle/fact/phrasing instead of converging on the same output.
    """
    user_prompt = (
        f"Content Type: {content_type}\n"
        f"Topic: {topic}\n\n"
        f"Retrieved Context:\n{format_context(context_chunks)}"
    )
    if daily_tip_focus:
        kind, name = daily_tip_focus
        user_prompt += (
            f"\n\nDaily Tip Focus ({kind}): {name}\n"
            f"Build this tip around a moment where this {kind.lower()} "
            f"leads (or nearly leads) to an incident related to the "
            f"Topic/Retrieved Context. Per the framework rules above, "
            f"show it only through the learner's actions/thoughts — "
            f"do not write \"{name}\" or any other State/Error name in "
            f"the tip text itself."
        )
    if avoid_repeating:
        already_generated = "\n".join(f"- {t}" for t in avoid_repeating)
        user_prompt += (
            f"\n\nAlready generated in this batch — pick a different fact, "
            f"angle, or example; do not repeat or lightly reword any of "
            f"these:\n{already_generated}"
        )
    if monthly_topic_content and monthly_topic_content.strip():
        user_prompt += f"\n\nMonthly Topic Content:\n{monthly_topic_content.strip()}"
    if common_data and common_data.strip():
        user_prompt += f"\n\nCommon Knowledge:\n{common_data.strip()}"
    if web_results and web_results.strip():
        user_prompt += f"\n\nInternet Research:\n{web_results.strip()}"
    return user_prompt


def build_image_prompt_request(
    content_type: str,
    topic: str,
    generated_content: str,
    context_chunks: List[dict],
    mode: str,
) -> str:
    """Build the user-turn prompt for the Image Prompt Generation Agent."""
    if mode not in ("infographic", "scene"):
        mode = "infographic"
    return (
        f"Output Mode: {mode}\n\n"
        f"Content Type: {content_type}\n"
        f"Topic: {topic}\n\n"
        f"Generated Content:\n{generated_content}\n\n"
        f"Retrieved Context:\n{format_context(context_chunks)}"
    )
