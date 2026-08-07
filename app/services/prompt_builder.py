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

# Content Type -> Output Mode. Must stay in sync with the "CONTENT TYPE
# -> OUTPUT MODE MAPPING" table in prompts/image_prompt_system.txt —
# same eleven Content Types, same three modes (infographic/concept/scene).
_CONTENT_TYPE_TO_MODE = {
    "Recall Card": "concept",
    "AI Image": "scene",
    "Infographic": "infographic",
    "Flashcard": "concept",
    "Scenario": "scene",
    "Spot the Mistake Challenge": "scene",
    "Daily Quiz": "concept",
    "Fun Fact": "concept",
    "Reflection Question": "scene",
    "Safety / Best Practice Tip": "scene",
    "Daily Tip": "scene",
}
_VALID_MODES = ("infographic", "concept", "scene")

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

# Must stay in sync with the "Setting selection" bullet under SHARED RULES
# in prompts/image_prompt_system.txt — same fifteen industries, same
# wording/order. The Image Prompt Generation Agent is told to match the
# industry to whatever the Generated Content actually names; this
# rotation only supplies a *suggested* industry for when the content is
# generic and names no specific industry itself, so that generic content
# doesn't keep converging on the same default (e.g. Food & Beverage/
# commercial-kitchen) across requests. Chosen at random per request,
# excluding whichever industry was used last time if the caller supplies
# it — the same "give the model a concrete pick instead of leaving it to
# chance" approach already used for _DAILY_TIP_FOCUS_ROTATION above.
_INDUSTRIES: List[str] = [
    "Warehouse & Logistics",
    "Manufacturing",
    "Construction",
    "Oil & Gas",
    "Mining",
    "Utilities & Electrical",
    "Agriculture",
    "Healthcare",
    "Transportation & Fleet",
    "Food & Beverage Manufacturing",
    "Chemical & Petrochemical",
    "Aviation",
    "Maritime & Ports",
    "Waste Management",
    "Corporate Office",
]

# Negative-prompt fallbacks, used only when the Image Prompt Generation
# Agent's JSON response comes back with an empty negative_prompt field.
# One fallback per Output Mode (infographic/concept/scene) — a "scene"
# image needs photorealism/cinematic lighting, so the infographic
# negative prompt (which bans exactly those things) would actively
# fight against Scenario/Spot-the-Mistake images ever looking realistic;
# concept images are neither, so they get their own set of bans too.
_INFOGRAPHIC_NEGATIVE_PROMPT = (
    "photorealistic, realistic photo, photograph, cinematic lighting, "
    "realistic human faces, realistic skin texture, depth of field, "
    "camera bokeh, film grain, 3D render of real people, blurry, low detail, "
    "watermarks, logos, misspelled text, garbled text, illegible text, typos"
)
_SCENE_NEGATIVE_PROMPT = (
    "infographic, icon, iconographic illustration, flat vector illustration, "
    "diagram, chart, flowchart, numbered steps, labeled boxes, comparison "
    "columns, collage, grid of panels, cartoon, clip art, text, captions, "
    "titles, labels, watermarks, logos, low quality, blurry, distorted "
    "anatomy, extra limbs, misspelled text, garbled text, illegible text, typos"
)
# concept mode now defaults to a single real photograph of one specific
# object/hazard (not a full scene like scene mode, and not a multi-step
# structured layout like infographic) — so its fallback no longer bans
# photorealism. It instead guards against the two failure modes actually
# seen in practice: multiple unrelated hazards/objects mashed into one
# illogical scene, and busy/cluttered compositions that defeat the
# "recognizable in under a second" goal of this mode.
_CONCEPT_NEGATIVE_PROMPT = (
    "multiple unrelated objects, combined hazards, mashed-together scene, "
    "physically implausible combination, multi-panel layout, numbered "
    "steps, comparison columns, timeline, flowchart, cluttered background, "
    "busy composition, long paragraphs of text, multiple text labels, "
    "narrative scene with multiple people, watermarks, logos, low quality, "
    "blurry, distorted anatomy, extra limbs, misspelled text, garbled "
    "text, illegible text, typos"
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
    return _CONTENT_TYPE_TO_MODE.get(content_type, "infographic")


def pick_daily_tip_focus() -> Tuple[str, str]:
    """Pick one random (kind, name) pair — e.g. ("State", "Rushing") or
    ("Error", "Eyes not on task") — from the 8-item States/Errors
    rotation. Called once per Daily Tip generation."""
    return random.choice(_DAILY_TIP_FOCUS_ROTATION)


def pick_industry(avoid: Optional[str] = None) -> str:
    """Pick one random industry from the 15-item rotation, excluding
    `avoid` (typically the industry used for this same caller's previous
    piece of content) so consecutive generations don't converge on the
    same one. If `avoid` isn't in the list or leaves no other choices,
    falls back to picking from the full list."""
    choices = [i for i in _INDUSTRIES if i != avoid] or _INDUSTRIES
    return random.choice(choices)


def negative_prompt_for_mode(mode: str) -> str:
    """Fallback negative prompt used only when the LLM's own
    negative_prompt came back empty."""
    if mode == "scene":
        return _SCENE_NEGATIVE_PROMPT
    if mode == "concept":
        return _CONCEPT_NEGATIVE_PROMPT
    return _INFOGRAPHIC_NEGATIVE_PROMPT


_COMBINED_OUTPUT_FORMAT_OVERRIDE = """
======================================================================
COMBINED-CALL MODE — read this section last, it overrides both
"OUTPUT FORMAT" sections above.
======================================================================
You are doing BOTH roles above in a single response, to save a network
round trip: first the Content Generation Agent's job (exactly per its
instructions above), then — using that content as your single source of
truth, exactly as the Image Prompt Generation Agent's instructions
above require — its job.

Do not show your work for either step. Return ONLY one JSON object and
nothing else: no markdown fences, no commentary before or after it.

{
  "content_text": "<the learner-facing text the Content Generation Agent
      would have produced, following ALL of its rules exactly>",
  "image_prompt": "<the Image Prompt Generation Agent's image_prompt,
      derived only from content_text above>",
  "negative_prompt": "...",
  "alt_text": "...",
  "tags": ["..."],
  "summary": "..."
}
"""


@lru_cache(maxsize=1)
def build_combined_system_prompt() -> str:
    """System prompt for the merged single-call path: both agents'
    original instructions, unmodified, plus a wrapper telling the model
    to do both jobs in sequence within one response and emit one JSON
    object. The two source .txt files are loaded as-is — nothing in
    them is rewritten — so their individual contracts (including "content
    is the image's single source of truth") stay intact; only the
    transport (one call instead of two) changes.
    """
    return (
        load_content_generation_system_prompt()
        + "\n\n"
        + load_image_prompt_system_prompt()
        + "\n\n"
        + _COMBINED_OUTPUT_FORMAT_OVERRIDE
    )


def build_combined_user_prompt(
    content_type: str,
    topic: str,
    context_chunks: List[dict],
    mode: str,
    monthly_topic_content: Optional[str] = None,
    common_data: Optional[str] = None,
    web_results: Optional[str] = None,
    daily_tip_focus: Optional[Tuple[str, str]] = None,
    avoid_repeating: Optional[List[str]] = None,
) -> str:
    """User-turn prompt for the merged call — the union of what
    build_content_generation_prompt() and build_image_prompt_request()
    would each have sent, minus Generated Content/Output Mode-as-afterthought
    (the model produces the content itself this call, then uses it)."""
    user_prompt = build_content_generation_prompt(
        content_type=content_type,
        topic=topic,
        context_chunks=context_chunks,
        monthly_topic_content=monthly_topic_content,
        common_data=common_data,
        web_results=web_results,
        daily_tip_focus=daily_tip_focus,
        avoid_repeating=avoid_repeating,
    )
    if mode not in _VALID_MODES:
        mode = "infographic"
    user_prompt += f"\n\nOutput Mode (for the image-prompt half of your response): {mode}"
    return user_prompt


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
    suggested_industry: Optional[str] = None,
) -> str:
    """Build the user-turn prompt for the Image Prompt Generation Agent.

    suggested_industry: one of the 15 industries from _INDUSTRIES,
      picked by pick_industry(). Passed through only as a fallback the
      system prompt tells the model to use when the Generated Content
      itself names no specific industry — it must NOT override an
      industry the content already implies.
    """
    if mode not in _VALID_MODES:
        mode = "infographic"
    user_prompt = (
        f"Output Mode: {mode}\n\n"
        f"Content Type: {content_type}\n"
        f"Topic: {topic}\n\n"
        f"Generated Content:\n{generated_content}\n\n"
        f"Retrieved Context:\n{format_context(context_chunks)}"
    )
    if suggested_industry:
        user_prompt += (
            f"\n\nSuggested Industry: {suggested_industry}\n"
            f"(Use this industry for the setting ONLY if the Generated "
            f"Content above names no specific industry or hazard type of "
            f"its own. If the content already implies a specific industry, "
            f"use that instead and ignore this suggestion.)"
        )
    return user_prompt