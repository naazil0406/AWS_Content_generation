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
import re
from functools import lru_cache
from typing import List, Optional, Tuple

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompts")

_CONTENT_GENERATION_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "content_generation_system.txt")
_IMAGE_PROMPT_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "image_prompt_system.txt")
_IMAGE_PROMPT_VALIDATOR_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "image_prompt_validator_system.txt")

# Content Type -> Output Mode. Must stay in sync with the "CONTENT TYPE
# -> OUTPUT MODE MAPPING" table in prompts/image_prompt_system.txt —
# same seven Content Types, same three modes (infographic/concept/scene).
_CONTENT_TYPE_TO_MODE = {
    "Recall Card": "concept",
    "Infographic": "infographic",
    "Scenario": "scene",
    "Spot the Mistake": "scene",
    "Question": "scene",
    "Safety Tip": "scene",
    "Fun Fact": "concept",
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

# The available industries are NOT hardcoded here. They live in exactly
# one place — the "AVAILABLE INDUSTRIES" section (between the
# "--- INDUSTRY LIST START ---" / "--- INDUSTRY LIST END ---" markers)
# near the top of prompts/image_prompt_system.txt — so there is a single
# source of truth both the Image Prompt Agent (which reads that section
# as part of its system prompt) and this application (which parses the
# same section below) agree on. To add/remove/rename an industry, edit
# ONLY that section of the .txt file; nothing here needs to change.
_INDUSTRY_LIST_START_MARKER = "--- INDUSTRY LIST START ---"
_INDUSTRY_LIST_END_MARKER = "--- INDUSTRY LIST END ---"
# Defensive fallback ONLY — used if the markers above are ever missing or
# reordered in the .txt file (e.g. mid-edit) so a random request doesn't
# hard-crash. This is not a second source of truth to keep in sync; if
# you see this list being used, the .txt file's markers need fixing.
_FALLBACK_INDUSTRIES: List[str] = ["Manufacturing", "Warehouse & Logistics", "Corporate Office"]

# Used by detect_explicit_industry() below: if the user's topic text
# names one of these — the canonical industry name itself, or one of
# these short aliases — that industry is used EXACTLY as given, never
# randomized. Keys must match load_available_industries()'s canonical
# names exactly; an industry added to the .txt list without an entry
# here still works, it just can't be explicitly detected by alias (the
# canonical name itself is always checked too) and won't be favored by
# the relevance-biased random fallback in pick_industry() below — both
# degrade gracefully rather than erroring.
_INDUSTRY_NAME_ALIASES: dict = {
    "Warehouse & Logistics": ["warehouse", "logistics", "distribution center", "fulfillment center"],
    "Manufacturing": ["manufacturing", "factory", "assembly line", "production plant"],
    "Construction": ["construction", "construction site", "job site", "building site"],
    "Oil & Gas": ["oil and gas", "oil & gas", "oil rig", "drilling rig", "refinery"],
    "Mining": ["mining", "mine site", "underground mine", "quarry"],
    "Utilities & Electrical": ["utilities", "utility", "electrical utility", "power grid", "substation"],
    "Agriculture": ["agriculture", "farming", "farm workers", "agricultural"],
    "Healthcare": ["healthcare", "hospital", "clinic", "medical facility"],
    "Transportation & Fleet": ["trucking", "fleet", "delivery drivers", "transportation company"],
    "Food & Beverage Manufacturing": ["food and beverage", "food processing", "food plant", "beverage plant"],
    "Chemical & Petrochemical": ["chemical plant", "petrochemical", "chemical manufacturing"],
    "Aviation": ["aviation", "airport", "airline", "aircraft maintenance"],
    "Maritime & Ports": ["maritime", "port operations", "shipping port", "dockyard"],
    "Waste Management": ["waste management", "landfill", "recycling facility", "garbage collection"],
    "Corporate Office": ["corporate office", "office workers", "office environment"],
}

# Used by pick_industry() below ONLY when detect_explicit_industry()
# found nothing — i.e. the topic never names an industry directly, but
# often still contains a concrete clue (an object, role, or hazard) that
# makes some industries a far more plausible random pick than others.
# Broader/looser than the aliases above on purpose: these are meant to
# catch incidental mentions ("watch for forklifts") without those
# mentions counting as the user explicitly declaring an industry.
_INDUSTRY_CONTEXT_HINTS: dict = {
    "Warehouse & Logistics": ["forklift", "pallet", "loading dock", "conveyor", "inventory", "shipping crate"],
    "Manufacturing": ["assembly", "machine operator", "production line", "cnc", "press machine"],
    "Construction": ["scaffolding", "ladder", "excavator", "rebar", "hard hat", "crane", "trench"],
    "Oil & Gas": ["pipeline", "drilling", "wellhead", "offshore platform", "flammable gas"],
    "Mining": ["underground mine", "excavator", "blasting", "quarry", "ore"],
    "Utilities & Electrical": ["voltage", "wiring", "power line", "circuit breaker", "transformer", "substation"],
    "Agriculture": ["tractor", "livestock", "pesticide", "harvest", "irrigation", "crop"],
    "Healthcare": ["patient", "needle", "surgical", "clinical", "nurse", "ppe gown"],
    "Transportation & Fleet": ["truck driver", "delivery route", "cargo van", "commercial vehicle", "dashcam"],
    "Food & Beverage Manufacturing": ["conveyor belt", "packaging line", "food safety", "sanitation"],
    "Chemical & Petrochemical": ["solvent", "corrosive", "reactor vessel", "hazmat", "fume hood"],
    "Aviation": ["runway", "aircraft", "ground crew", "jet fuel", "hangar"],
    "Maritime & Ports": ["cargo ship", "shipping port", "crane operator", "shipping container", "vessel", "dockyard", "cargo dock"],
    "Waste Management": ["garbage truck", "landfill", "recycling", "hazardous waste disposal"],
    "Corporate Office": ["cubicle", "desk", "meeting room", "office chair", "computer monitor"],
}

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


@lru_cache(maxsize=1)
def load_image_prompt_validator_system_prompt() -> str:
    """The second-pass QA agent (see content_generation_engine.py) that
    checks each of the Image Prompt Generation Agent's three draft
    prompts against the Generated Content — content grounding, semantic
    match, industry consistency, specificity, the Generic Image Test, the
    Content Type Test, and distinctness — and rewrites any that fail
    BEFORE they ever reach Freepik. See
    prompts/image_prompt_validator_system.txt for the full rules.
    """
    with open(_IMAGE_PROMPT_VALIDATOR_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=1)
def load_available_industries() -> Tuple[str, ...]:
    """Parse the industry list straight out of image_prompt_system.txt's
    "AVAILABLE INDUSTRIES" section (between the START/END markers) — the
    single source of truth both this application and the Image Prompt
    Agent read. Returns a tuple (cached — the .txt file doesn't change at
    runtime) of non-empty, stripped industry names in file order.

    Falls back to a tiny defensive list (_FALLBACK_INDUSTRIES) only if
    the markers can't be found, so a malformed prompt file degrades
    randomization rather than crashing every request — this fallback is
    not meant to be a maintained list; fix the .txt file's markers
    instead of relying on it.
    """
    text = load_image_prompt_system_prompt()
    start = text.find(_INDUSTRY_LIST_START_MARKER)
    end = text.find(_INDUSTRY_LIST_END_MARKER)
    if start == -1 or end == -1 or end <= start:
        return tuple(_FALLBACK_INDUSTRIES)
    block = text[start + len(_INDUSTRY_LIST_START_MARKER):end]
    industries = [line.strip() for line in block.splitlines() if line.strip()]
    return tuple(industries) or tuple(_FALLBACK_INDUSTRIES)


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


def detect_explicit_industry(topic: str) -> Optional[str]:
    """If `topic` names a specific industry — either the canonical name
    itself (e.g. "Healthcare") or one of _INDUSTRY_NAME_ALIASES' short
    aliases (e.g. "hospital") — returns that industry EXACTLY, to be used
    as-is and never randomized or replaced. Returns None if no industry
    is named, in which case the caller should fall back to
    pick_industry() below.

    Checked in load_available_industries() order (file order in
    image_prompt_system.txt) so results are deterministic if a topic
    happens to match more than one industry's alias.
    """
    if not topic:
        return None
    topic_lower = topic.lower()
    for industry in load_available_industries():
        candidates = [industry.lower()] + [a.lower() for a in _INDUSTRY_NAME_ALIASES.get(industry, [])]
        for alias in candidates:
            if re.search(r"\b" + re.escape(alias) + r"\b", topic_lower):
                return industry
    return None


def pick_industry(topic: str = "", avoid: Optional[str] = None) -> str:
    """Randomly select ONE industry from the single-source-of-truth list
    (load_available_industries()), excluding `avoid` — called ONLY when
    detect_explicit_industry() found nothing, i.e. the topic never names
    an industry directly.

    Still genuinely random, but no longer uniformly so: if `topic`
    contains any of _INDUSTRY_CONTEXT_HINTS' looser contextual clues
    (an object, role, or hazard associated with a specific industry —
    without naming that industry outright), the random choice is
    restricted to the matching industries instead of the full 15-item
    list, so the result is topic-relevant rather than arbitrary. Only
    when the topic gives zero such clues does this fall back to a fully
    unrestricted random choice across every industry — still randomized
    (never the same default every time), just with no basis to narrow it.

    Called once per request by ContentGenerationEngine.generate() when
    the caller supplies no explicit `industry` AND detect_explicit_industry()
    found none in the topic. Whatever this returns becomes THE industry
    for that entire request — passed identically to both the Content
    Generation Agent and the Image Prompt Agent, never re-picked
    independently by either.
    """
    industries = load_available_industries()
    relevant = industries
    if topic:
        topic_lower = topic.lower()
        hinted = [
            industry
            for industry in industries
            if any(re.search(r"\b" + re.escape(hint) + r"\b", topic_lower) for hint in _INDUSTRY_CONTEXT_HINTS.get(industry, []))
        ]
        if hinted:
            relevant = tuple(hinted)
    choices = [i for i in relevant if i != avoid] or [i for i in industries if i != avoid] or list(industries)
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
above require — its job, INCLUDING that agent's requirement to extract
distinct meaningful visual concepts and produce THREE separately
content-anchored prompts, not one. The reminder in that agent's
instructions about avoiding industry-generic objects (a glove, a boot, a
wrench) that would make just as much sense with the content deleted
applies exactly as written — do not relax it just because this is a
single combined call.

IMPORTANT: this combined path saves a network round trip but skips the
separate second-pass validator step the two-call pipeline normally runs
before Freepik. That validator exists specifically to catch prompts that
only look grounded. Since it won't run here, hold yourself to that same
bar on the first attempt — re-read your own content_text before writing
each image entry and confirm you can point to the exact clause it
represents, exactly as that agent's instructions describe.

Do not show your work for either step. Return ONLY one JSON object and
nothing else: no markdown fences, no commentary before or after it.

{
  "content_text": "<the learner-facing text the Content Generation Agent
      would have produced, following ALL of its rules exactly>",
  "image_prompts": [
    {"content_anchor": "<exact phrase/clause from content_text>", "visual_concept": "<one-sentence bridge from that anchor to the image>", "prompt": "<the image-generation prompt text>"},
    {"content_anchor": "...", "visual_concept": "...", "prompt": "..."},
    {"content_anchor": "...", "visual_concept": "...", "prompt": "..."}
  ],
  "negative_prompt": "...",
  "alt_text": "...",
  "tags": ["..."],
  "summary": "..."
}

image_prompts must be EXACTLY THREE entries, each built around a
DIFFERENT extracted concept (see the Image Prompt Generation Agent's
instructions above for how to extract them and what to do if that list
comes up thin) — never three reworded variants of the same idea.
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
    industry: Optional[str] = None,
    monthly_topic_content: Optional[str] = None,
    common_data: Optional[str] = None,
    web_results: Optional[str] = None,
    daily_tip_focus: Optional[Tuple[str, str]] = None,
    avoid_repeating: Optional[List[str]] = None,
) -> str:
    """User-turn prompt for the merged call — the union of what
    build_content_generation_prompt() and build_image_prompt_request()
    would each have sent, minus Generated Content/Output Mode-as-afterthought
    (the model produces the content itself this call, then uses it).

    industry: same resolved-once value ContentGenerationEngine.generate()
      passes everywhere else — previously this function silently dropped
      it (a real gap: combined-mode content generation never saw the
      authoritative industry at all, unlike the separate pipeline's
      build_content_generation_prompt() call). Threaded through here so
      combined mode's content_text and image_prompts share the same
      industry grounding the separate pipeline already guaranteed.
    """
    user_prompt = build_content_generation_prompt(
        content_type=content_type,
        topic=topic,
        context_chunks=context_chunks,
        industry=industry,
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
    industry: Optional[str] = None,
    monthly_topic_content: Optional[str] = None,
    common_data: Optional[str] = None,
    web_results: Optional[str] = None,
    daily_tip_focus: Optional[Tuple[str, str]] = None,
    avoid_repeating: Optional[List[str]] = None,
) -> str:
    """Build the user-turn prompt for the Content Generation Agent.

    industry: the professional/workplace context resolved for this
      request — either explicitly given by the caller, or, if not,
      randomly selected once by ContentGenerationEngine.generate() (see
      prompt_builder.pick_industry()) before this prompt was built. This
      is always populated (never invented or swapped mid-request) and is
      the SAME value also sent to the Image Prompt Agent for this
      request's three images. The content must actually take place in
      and reflect this industry's realistic day-to-day work, environment,
      or hazards — not merely mention the industry's name once and
      otherwise stay generic. Never invent facts the Retrieved Context
      doesn't support just to color the industry in more, and never
      substitute a different industry.
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
    if industry and industry.strip():
        user_prompt += (
            f"\n\nIndustry: {industry.strip()}\n"
            f"This request's professional/workplace context (explicitly given, or "
            f"randomly selected for this request if none was given — either way, treat "
            f"it as fixed for this request). The content must actually take place in "
            f"and reflect this industry's realistic day-to-day work, environment, or "
            f"hazards — do not just name-drop the industry once and otherwise stay "
            f"generic. Never invent facts the Retrieved Context doesn't support, and "
            f"never substitute a different industry."
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
    industry: Optional[str] = None,
) -> str:
    """Build the user-turn prompt for the Image Prompt Generation Agent.

    industry: the professional/workplace context resolved ONCE for this
      request by ContentGenerationEngine.generate() — either explicitly
      given by the caller, or randomly selected via pick_industry() when
      not given. Either way, by the time this function is called it is a
      concrete, resolved value and is authoritative: all three images
      must use it as the professional context/setting, exactly as given.
      It must never be invented, randomly re-picked here, or swapped for
      a different industry just because the content commonly associates
      with another one. It is also the exact same value sent to the
      Content Generation Agent for this request — never a different one.
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
    if industry and industry.strip():
        user_prompt += (
            f"\n\nIndustry: {industry.strip()}\n"
            f"(AUTHORITATIVE for this request — resolved once, before either agent ran, "
            f"and identical to what the Content Generation Agent received. All three "
            f"images must be set in this industry's professional context, exactly as "
            f"given. Do not substitute, mix in, or drift toward a different industry, "
            f"even if the Generated Content's topic is more commonly associated with "
            f"one — this Industry value always wins. Do not invent industry-specific "
            f"facts, objects, or hazards beyond what the Generated Content/Retrieved "
            f"Context supports; this field sets WHERE the content is shown, not what "
            f"additional details to add.)"
        )
    return user_prompt


def build_image_prompt_validation_request(
    content_type: str,
    mode: str,
    generated_content: str,
    draft_entries: List[dict],
    negative_prompt: str,
    alt_text: str,
    tags: List[str],
    summary: str,
    industry: Optional[str] = None,
) -> str:
    """Build the user-turn prompt for the Image Prompt Validator — the
    second-pass QA agent that checks the drafting agent's three
    content_anchor/visual_concept/prompt entries against the Generated
    Content (see prompts/image_prompt_validator_system.txt for the seven
    checks) before anything reaches Freepik.

    draft_entries: exactly three dicts, each with keys content_anchor,
      visual_concept, prompt — the drafting agent's raw output, as parsed
      by BaseLLMService.generate_image_prompt_package().
    """
    if mode not in _VALID_MODES:
        mode = "infographic"

    def _format_entry(i: int, entry: dict) -> str:
        return (
            f"Draft {i}:\n"
            f"  content_anchor: {entry.get('content_anchor', '')}\n"
            f"  visual_concept: {entry.get('visual_concept', '')}\n"
            f"  prompt: {entry.get('prompt', '')}"
        )

    drafts_block = "\n\n".join(_format_entry(i, entry) for i, entry in enumerate(draft_entries, start=1))

    user_prompt = (
        f"Output Mode: {mode}\n"
        f"Content Type: {content_type}\n\n"
        f"Generated Content:\n{generated_content}\n\n"
    )
    if industry and industry.strip():
        user_prompt += f"Industry: {industry.strip()}\n\n"
    user_prompt += (
        f"Shared negative_prompt: {negative_prompt}\n"
        f"Shared alt_text: {alt_text}\n"
        f"Shared tags: {', '.join(tags) if tags else ''}\n"
        f"Shared summary: {summary}\n\n"
        f"Draft image prompts to validate (exactly three, same order must "
        f"be preserved in your response):\n\n{drafts_block}"
    )
    return user_prompt