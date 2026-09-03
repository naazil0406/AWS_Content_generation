"""
Input validation + prompt understanding pipeline (see the validation
flow diagram in the project spec: basic -> special-char -> meaningful ->
spelling -> moderation -> understanding -> [content generation happens
in the caller] -> content relevance -> [image generation happens in the
caller] -> image-prompt relevance).

Design rationale, stated plainly:
- Steps that can be answered with a regex/heuristic (empty input,
  special-characters-only, obvious keyboard-mash/repeated-character
  gibberish) run FIRST and never touch the LLM — this satisfies the
  explicit performance requirement ("do not call expensive generation
  APIs for obviously invalid input") for the cheapest, most common
  garbage-input cases.
- Meaningful-word validation (beyond the obvious-gibberish heuristic),
  content moderation, spelling correction, and prompt understanding are
  combined into ONE LLM call (analyze_prompt()) rather than four
  separate round trips — they all need to read the same prompt once and
  produce a few related judgments, and every extra round trip is real,
  billed latency this project's own performance requirements explicitly
  ask to avoid.
- Content relevance checking (after content is generated) is a SEPARATE,
  smaller LLM call, because it depends on output that doesn't exist yet
  at prompt-analysis time.
- There is deliberately NO separate "image relevance" check that
  inspects actual rendered image pixels — this codebase has no
  vision-capable model call wired up, and this module will not fake one.
  Image-prompt relevance is instead addressed structurally: the image
  prompt is built from the SAME validated/corrected prompt and the
  content that already passed the relevance check above, and
  image_prompt_validator_system.txt's Image Prompt Validator Agent
  step in this project's existing image-prompt pipeline. See the
  project's test-report for exactly which acceptance-test cases this
  means are marked "manual verification required" rather than
  automatically tested.
"""

import json
import logging
import os
import re
import unicodedata
from typing import Callable, List, Optional, TypedDict

from app.services.llm_service import BaseLLMService

logger = logging.getLogger(__name__)

# --- Externalized validation rules (validation_rules.txt) -------------------
# Per explicit requirement: warning MESSAGES and simple thresholds live in
# an editable TXT file, not hardcoded in this module, so they can be
# changed without touching Python. What deliberately stays IN CODE: the
# actual detection logic (regexes, the moderation category list given to
# the LLM, obfuscation normalization) — moving detection logic itself
# into a plain-text file that ships alongside the app would make it
# trivially readable/editable by anyone with file access, which is
# exactly wrong for a safety-moderation mechanism.

_RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "validation_rules.txt")


def load_rules(path: str = _RULES_PATH) -> dict:
    """Parses validation_rules.txt into {section: {key: value}}. Called
    fresh on every use of get_rule() below (the file is small and this
    keeps local-dev editing responsive — save the file, next request
    picks it up immediately, no server restart needed) rather than
    cached at import time."""
    rules: dict = {}
    if not os.path.exists(path):
        logger.error("validation_rules.txt not found at %s — using built-in fallback messages.", path)
        return rules
    current_section = None
    current_key = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            if line.strip().startswith("[") and line.strip().endswith("]"):
                current_section = line.strip()[1:-1].strip()
                rules[current_section] = {}
                current_key = None
                continue
            if line.startswith((" ", "\t")) and current_section and current_key:
                # Continuation line for a multi-line value.
                rules[current_section][current_key] += " " + line.strip()
                continue
            if "=" in line and current_section:
                key, _, value = line.partition("=")
                key = key.strip()
                rules[current_section][key] = value.strip()
                current_key = key
    return rules


def get_rule(section: str, key: str, default: str) -> str:
    rules = load_rules()
    return rules.get(section, {}).get(key, default)


class ValidationError(Exception):
    """Raised for any input the pipeline refuses to generate for.
    `warning` is the exact user-facing message (see the spec's PART 15);
    `error_type` distinguishes validation ("empty"/"special_chars"/
    "meaningless"/"insufficient_detail") from moderation, so a caller
    (app/routes/content.py) can return a structured
    {success: false, warning, error_type} response without needing to
    parse the message text."""

    def __init__(self, warning: str, error_type: str):
        super().__init__(warning)
        self.warning = warning
        self.error_type = error_type


# --- Cheap, no-LLM-call heuristics ------------------------------------------

_SPECIAL_CHARS_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)  # no letters/digits anywhere
_DIGITS_ONLY_RE = re.compile(r"^\d+$")
_SINGLE_REPEATED_CHAR_RE = re.compile(r"^(.)\1{2,}$")  # e.g. "aaaaaaa", "zzzzzzz"
_REPEATED_SUBSTRING_RE = re.compile(r"^(\w{2,})\1+$")  # e.g. "testtesttest" = "test"*3
_VOWELS = set("aeiouAEIOU")

# QWERTY row-adjacency map, used to detect literal keyboard-mash input
# like "asdfgh" (home row) or "qwerty" (top row) without needing a real
# dictionary/spellcheck API (none is available in this environment).
_KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_ADJACENT_PAIRS = set()
for _row in _KEYBOARD_ROWS:
    for _i in range(len(_row) - 1):
        _ADJACENT_PAIRS.add((_row[_i], _row[_i + 1]))
        _ADJACENT_PAIRS.add((_row[_i + 1], _row[_i]))


def _is_keyboard_mash(word: str) -> bool:
    """True if nearly every consecutive letter pair in `word` is
    adjacent on a QWERTY row — catches "asdfgh", "qwerty", "zxcvbn"
    style input generically, rather than via a fixed word blocklist."""
    w = word.lower()
    if len(w) < 4 or not w.isalpha():
        return False
    pairs = list(zip(w, w[1:]))
    adjacent_count = sum(1 for pair in pairs if pair in _ADJACENT_PAIRS)
    return (adjacent_count / len(pairs)) >= 0.8


def is_empty_or_blank(prompt: str) -> bool:
    return not (prompt or "").strip()


def is_numbers_only(prompt: str) -> bool:
    """True for "123456", "2026", "5" — numbers WITH other meaningful
    words ("Create 5 safety tips", "16:9 infographic") never match this,
    since those aren't the ENTIRE (stripped, space-removed) prompt."""
    stripped = (prompt or "").strip()
    compact = stripped.replace(" ", "")
    return bool(compact) and bool(_DIGITS_ONLY_RE.match(compact))


def is_special_characters_only(prompt: str) -> bool:
    """True only when the prompt has NO letters or digits anywhere —
    "@", "!!!", "@#$%" all match; "AI & Cybersecurity", "16:9 safety
    infographic", "50% workplace safety improvement" all have letters/
    digits and correctly do NOT match, regardless of how much
    punctuation they also contain."""
    stripped = (prompt or "").strip()
    return bool(stripped) and bool(_SPECIAL_CHARS_ONLY_RE.match(stripped))


def is_obvious_gibberish(prompt: str) -> bool:
    """Fast pre-filter for the most blatant meaningless-input cases from
    the spec (digit strings, single-repeated-character runs, immediately
    repeated substrings, QWERTY keyboard-mash) — deliberately narrow so
    it never false-positives on short real words like "Safety" or
    "Cybersecurity". Anything that survives this still goes through
    analyze_prompt()'s LLM-based meaningful-word judgment below, which is
    the authoritative check for anything less blatant than these
    patterns."""
    stripped = (prompt or "").strip()
    if not stripped:
        return False
    compact = stripped.replace(" ", "")
    if _DIGITS_ONLY_RE.match(compact):
        return True
    if _SINGLE_REPEATED_CHAR_RE.match(compact.lower()):
        return True
    if _REPEATED_SUBSTRING_RE.match(compact.lower()):
        return True
    # Single-token inputs only (multi-word prompts are never keyboard-mash
    # as a whole — check each token instead, since a keyboard-mash TOKEN
    # embedded in an otherwise meaningful multi-word prompt should not
    # fail the whole prompt here; that nuance is left to the LLM check).
    tokens = stripped.split()
    if len(tokens) == 1 and _is_keyboard_mash(tokens[0]):
        return True
    return False


# --- Combined LLM call: meaningful check + moderation + spelling + intent --

_ANALYSIS_SYSTEM_PROMPT = """You are a strict input-validation and intent-understanding component for an educational content generation tool. You will be given ONE user prompt describing what content/image they want generated.

Analyze it and respond with ONLY a single JSON object (no markdown fences, no commentary), with exactly these keys:

{
  "is_meaningful": true/false,
  "is_appropriate": true/false,
  "moderation_reason": "short internal reason if is_appropriate is false, else null",
  "corrected_prompt": "the prompt with ONLY obvious spelling/typing mistakes fixed",
  "corrections": [{"from": "saftey", "to": "safety"}, ...],
  "intent": {
    "topic": "...", "audience": "... or null", "format": "... or null",
    "requirements": ["...", ...], "visual_style": "... or null",
    "colors": ["...", ...], "aspect_ratio_hint": "... or null",
    "required_text": ["...", ...],
    "kb_search_query": "...",
    "suggested_industry": "exact name from AVAILABLE INDUSTRIES below, or null"
  }
}

kb_search_query is a SEPARATE field from topic, used only for searching the Knowledge Base — it must preserve ALL substantive concepts from the prompt (subject, action/behavior, context, industry, who's involved) and strip ONLY the conversational/instructional wrapper (e.g. "give me an image related to", "create a poster about", "I want"). Do NOT reduce it down to a single word or the bare topic alone if the prompt named other substantive concepts too — e.g. for "a woman rushing in healthcare industry", kb_search_query must be something like "woman rushing healthcare industry" (preserving the person, the action, AND the industry), NOT reduced to just "rushing" or just "healthcare" — losing any of those concepts loses real semantic information the search needs. topic itself can stay a short label (used elsewhere, e.g. "rushing"); kb_search_query is deliberately fuller.

suggested_industry: return the exact industry name (character-for-character) from the AVAILABLE INDUSTRIES list below ONLY if the topic clearly, unambiguously implies that real-world setting (e.g. "car"/"driving" clearly implies a transportation/fleet setting). Return null if the topic is genuinely generic/industry-agnostic (e.g. "workplace safety", "fire safety") — these are intentionally meant to be illustrated across ANY industry for variety, not pinned to one. When unsure, prefer null over guessing.

AVAILABLE INDUSTRIES:
{available_industries}

RULES:
- is_meaningful: false for input that does not name or clearly imply ANY topic/subject to generate content about. This has two distinct failure categories, both false:
  (a) no discernible real-world topic at all (keyboard mash, random characters, pure noise), OR
  (b) PURE SOCIAL SPEECH-ACTS that carry no subject-matter content at all — greetings ("hello", "hi", "hey there"), acknowledgments/fillers ("thanks", "ok", "yes", "sure", "no"), or requests that name no subject whatsoever ("make me something", "create it", "help me").
  The test for (b) is narrow and specific: is this ONLY a social gesture/filler with zero identifiable subject, or does it name/imply an actual real-world thing, behavior, state, activity, emotion, or phenomenon? The latter is ALWAYS meaningful (true), no matter how short, common, or abstract that single word is — "rushing", "fatigue", "complacency", "frustration", "safety", "cybersecurity", "cars", "certs" are all meaningful, because each names a real concept/behavior/subject someone could genuinely write content about.
  EXPLICITLY DO NOT REJECT (is_meaningful must be true) input merely because it:
  * is short — a single word or a few words is enough if a real subject is identifiable
  * contains only one or two meaningful words
  * contains no verb at all — a bare noun, gerund, or adjective naming a real thing/behavior/state counts fully (e.g. "rushing", "fatigue", "cars")
  * does not contain the words "create", "generate", "make", "poster", "image", or any other action/format word — the input is a TOPIC/IDEA, not required to be phrased as a request
  * is phrased as a topic instead of a full sentence
  * is a creative or descriptive idea rather than a literal instruction
  * has no exact Knowledge Base match — that is a separate, later concern (Knowledge Base relevance), never a reason to call the input itself not meaningful
  Do not require the word to look like a formal "topic" or be a noun — a single verb/gerund naming a real behavior (e.g. "rushing") is exactly as valid a topic as a noun (e.g. "safety"). Judge semantic meaning, never by checking for specific keywords. When in doubt between "this is a real subject" and "this is just filler," prefer true (meaningful) — false is reserved for the narrow, unambiguous cases of pure gibberish or pure social nicety with no subject at all.
- is_appropriate: false for any request in these categories, regardless of phrasing, misspelling, unusual spacing/punctuation, character substitution, or other obfuscation attempts:
  * instructions or facilitation for harming people, self-harm, or violent wrongdoing
  * facilitating dangerous criminal activity, weapons, or explosives
  * poisoning or other toxic/chemical harm
  * sexual content that is explicit, pornographic, or intended for explicit sexual depiction
  * sexual exploitation, sexual solicitation, or non-consensual sexual content
  * any sexual content involving minors, under any framing including fictional
  Judge the SEMANTIC INTENT behind the text, not just literal exact words — a request written with misspellings, extra spaces/punctuation between letters, or character substitutions (e.g. "@" for "a", "3" for "e") that clearly still means one of the above categories is still is_appropriate: false. Conversely, do NOT flag ordinary educational, medical, safety-training, or health terminology as inappropriate merely because a word could theoretically have another meaning out of context — judge the actual request as a whole.
- corrected_prompt: fix ONLY obvious spelling/typing errors. NEVER change technical terms, brand/product/company names, acronyms, abbreviations, numbers, model names, or URLs. If you are not certain something is a typo, leave it unchanged. If there are no corrections, corrected_prompt must be identical to the input and corrections must be an empty list.
- intent: extract only what is explicitly present or clearly implied by the prompt; use null/empty for anything not specified. Never invent requirements the user didn't state.
- The user's original prompt is the source of truth for topic/intent — never substitute a different topic.
"""


def normalize_for_moderation(text: str) -> str:
    """Produces a normalized VIEW of the text used only as an extra
    signal alongside the LLM's own semantic judgment above — never used
    to alter the actual prompt sent to generation (see run_input_validation:
    the ORIGINAL/corrected prompt, not this normalized form, is what's
    ever passed onward). Reverses common obfuscation attempts (PART 15):
    case, repeated characters, whitespace-between-letters, and a small
    set of unambiguous character substitutions — without changing intent
    or meaning of ordinary text passed through it, since this output is
    never shown to anyone or used for anything but a coarse secondary
    moderation signal."""
    text = unicodedata.normalize("NFKC", text).lower()
    substitutions = {
        "@": "a", "4": "a", "3": "e", "1": "i", "!": "i",
        "0": "o", "$": "s", "5": "s", "7": "t", "+": "t",
    }
    for src, dst in substitutions.items():
        text = text.replace(src, dst)
    # Collapse whitespace/punctuation deliberately inserted between
    # letters of a single word (e.g. "s e x", "s.e.x", "s-e-x").
    text = re.sub(r"(?<=[a-z])[\s._\-]+(?=[a-z])", "", text)
    # Collapse runs of 3+ identical letters down to 1 (e.g. "haaarm").
    text = re.sub(r"([a-z])\1{2,}", r"\1", text)
    return text


class IntentModel(TypedDict, total=False):
    topic: str
    audience: Optional[str]
    format: Optional[str]
    requirements: List[str]
    visual_style: Optional[str]
    colors: List[str]
    aspect_ratio_hint: Optional[str]
    required_text: List[str]
    kb_search_query: str
    suggested_industry: Optional[str]


class AnalysisResult(TypedDict):
    is_meaningful: bool
    is_appropriate: bool
    moderation_reason: Optional[str]
    corrected_prompt: str
    corrections: List[dict]
    intent: IntentModel


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def analyze_prompt(
    llm: BaseLLMService, original_prompt: str,
    kb_sample_fn: Optional[Callable[[], List[dict]]] = None,
    available_industries: Optional[List[str]] = None,
) -> AnalysisResult:
    """The ONE LLM call covering meaningful-word validation (beyond the
    cheap heuristics above), content moderation, spelling correction,
    prompt understanding, AND industry suggestion (folded in here
    specifically to avoid a SEPARATE sequential LLM call — see
    app/routes/content.py, which used to call suggest_consistent_industry()
    as its own round trip after this one; merging them cuts one full
    LLM call's latency off the happy path, which matters because this
    whole chain runs BEFORE Freepik/image rendering even starts, inside
    API Gateway's fixed ~29s ceiling). Raises ValidationError directly
    for is_meaningful=false / is_appropriate=false, so callers only need
    to handle the success path plus ValidationError.

    The moderation judgment is given BOTH the raw prompt and a
    normalized form (normalize_for_moderation() above) that reverses
    common obfuscation (case, character substitution, inserted spacing)
    — this is a defense-in-depth measure per PART 14/15: an obfuscated
    prompt that reads as garbled nonsense in its raw form should still
    be caught once its normalized form reveals clear harmful/sexual
    intent. The NORMALIZED form is used only for this judgment; it is
    never used as corrected_prompt or passed to generation.

    available_industries, if given, enables suggested_industry in the
    returned intent dict (see _ANALYSIS_SYSTEM_PROMPT's schema) — pass
    the same list load_available_industries() returns. Omit it (None)
    to skip asking for an industry suggestion at all (e.g. if a caller
    doesn't need it)."""
    normalized = normalize_for_moderation(original_prompt)
    user_message = original_prompt.strip()
    if normalized != original_prompt.strip().lower():
        user_message += f"\n\n[Normalized form for moderation purposes only, ignore for spelling/intent extraction: {normalized}]"
    industries_text = "\n".join(f"- {name}" for name in (available_industries or [])) or "(not provided — leave suggested_industry null)"
    # .replace(), NOT .format() — this system prompt contains many
    # literal JSON curly braces (the schema example itself) that would
    # break str.format()'s placeholder parsing.
    system_prompt = _ANALYSIS_SYSTEM_PROMPT.replace("{available_industries}", industries_text)
    raw = llm._call_llm(system_prompt, user_message)
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Fail OPEN on a parse error rather than blocking every request
        # if the model ever returns malformed JSON — treat as "meaningful,
        # no corrections, no understood intent" so generation can still
        # proceed using the raw prompt, and log loudly so this is visible
        # in CloudWatch rather than silently degrading validation quality.
        logger.error("analyze_prompt(): failed to parse LLM JSON output (%s). Raw: %.300s", exc, raw)
        return {
            "is_meaningful": True,
            "is_appropriate": True,
            "moderation_reason": None,
            "corrected_prompt": original_prompt.strip(),
            "corrections": [],
            "intent": {},
        }

    if not data.get("is_meaningful", True):
        raise ValidationError(_build_meaningless_warning(llm, kb_sample_fn), error_type="meaningless")
    if not data.get("is_appropriate", True):
        logger.info("Content moderation blocked a request. Internal reason: %s", data.get("moderation_reason"))
        raise ValidationError(
            get_rule(
                "CONTENT_MODERATION", "warning",
                "This request cannot be processed because it contains content that is not allowed.",
            ),
            error_type="moderation",
        )

    return {
        "is_meaningful": True,
        "is_appropriate": True,
        "moderation_reason": None,
        "corrected_prompt": (data.get("corrected_prompt") or original_prompt).strip(),
        "corrections": data.get("corrections") or [],
        "intent": data.get("intent") or {},
    }


def format_spelling_warning(corrections: List[dict]) -> Optional[str]:
    if not corrections:
        return None
    pairs = ", ".join(f"{c.get('from')} → {c.get('to')}" for c in corrections if c.get("from") and c.get("to"))
    if not pairs:
        return None
    return f"⚠️ Spelling corrections were made: {pairs}"


# --- Post-generation content relevance check --------------------------------

_RELEVANCE_SYSTEM_PROMPT = """You check whether generated content actually matches what a user asked for. You will be given the user's ORIGINAL request and a piece of GENERATED CONTENT. Respond with ONLY a single JSON object: {"is_relevant": true/false, "reason": "short reason"}.

is_relevant must be false if ANY of these are true:
- The generated content is about a substantially different topic/subject than the request (e.g. request was about fire safety but content is about cybersecurity).
- The content REFRAMES or SUBSTITUTES the literal subject with a different scenario, industry, or setting that contradicts it — e.g. a request about "car" being turned into an aviation/pilot scenario, or a request about "office safety" being turned into a construction-site scenario. The literal subject the user named must remain the actual subject of the content, not merely be mentioned once while a different setting dominates.
- The content only superficially mentions the requested topic (e.g. one passing reference) while being mainly about something else.

Minor phrasing, emphasis, or added illustrative detail that stays clearly ABOUT the same literal subject does not make content irrelevant — only judge false when the subject itself has been changed or diluted."""


def check_content_relevance(llm: BaseLLMService, original_prompt: str, generated_content: str) -> bool:
    """Returns True if relevant. On any parse/call failure, fails OPEN
    (returns True) rather than blocking a successful generation on a
    broken relevance check — logged loudly either way."""
    user_prompt = f"ORIGINAL REQUEST:\n{original_prompt}\n\nGENERATED CONTENT:\n{generated_content}"
    try:
        raw = llm._call_llm(_RELEVANCE_SYSTEM_PROMPT, user_prompt)
        data = json.loads(_strip_code_fences(raw))
        is_relevant = bool(data.get("is_relevant", True))
        if not is_relevant:
            logger.info("Content relevance check FAILED. Reason: %s", data.get("reason"))
        return is_relevant
    except Exception as exc:  # noqa: BLE001 - relevance check is best-effort, never fatal
        logger.error("check_content_relevance(): call/parse failed (%s) — failing open.", exc)
        return True


_IMAGE_PROMPT_RELEVANCE_SYSTEM_PROMPT = """You check whether an image-generation prompt (text that will be sent to an image renderer) actually depicts what a user asked for AND matches the content that was generated for them. You will be given the user's ORIGINAL request, the GENERATED CONTENT (the text already produced for this request), and one or more IMAGE PROMPT(S) that are about to be rendered. Respond with ONLY a single JSON object: {"is_relevant": true/false, "reason": "short reason"}.

is_relevant must be false if EITHER of these is true:
1. BROAD SUBJECT MISMATCH: the image prompt(s) describe a scene/subject that does not match EITHER the user's literal request OR the generated content's actual subject matter — e.g. the user asked about "car"/driving and the image prompt describes an office worker at a computer; or the generated content is about 5 specific driving-safety habits but the image prompt describes something generic and unrelated to driving. A generic professional/office setting used as backdrop when neither the request nor the content has anything to do with an office is a common failure mode — flag it as not relevant.
2. EXPLICIT DETAIL DROPPED: the user's original request explicitly specified a concrete, checkable detail about who/what should be depicted — a person's gender ("a woman", "a man"), a specific count of people/objects, a specific color, a specific age group, or similar explicit specifics — and the image prompt's TEXT does not actually specify that same detail (e.g. user said "a woman" but the image prompt just says "a worker" or "a person" with no gender specified, or explicitly says "a man"). This is false even if the overall scene/topic otherwise matches — an explicitly requested demographic/count/color detail silently dropped or changed is exactly the kind of mismatch this check exists to catch. Only flag this for details the user ACTUALLY stated explicitly — never invent a requirement they didn't mention.

Minor stylistic details (lighting, composition, exact pose, background specifics) that the user did NOT explicitly specify do not make it irrelevant — this check is about explicitly-stated requirements being honored, not about demanding creative specificity the user never asked for."""


def check_image_prompt_relevance(
    llm: BaseLLMService, original_prompt: str, generated_content: str, image_prompts: List[str],
) -> bool:
    """Checks the TEXT PROMPT(S) about to be sent to the image renderer
    against BOTH the user's original request AND the content that was
    actually generated for it (PART 11: image prompt must be based on
    original prompt + generated content, not the prompt alone) — NOT a
    check of the actual rendered image pixels (no vision-model call
    exists in this codebase; see this module's top docstring). Catching
    a drifted image PROMPT before it's ever rendered is strictly better
    than catching nothing: it costs one small text-only LLM call instead
    of a full Freepik render, and it directly prevents the exact failure
    mode where content generation stays on-topic but the image prompt
    (often influenced by a randomly-selected 'industry' context — see
    ContentGenerationEngine.generate()) drifts into a generic/unrelated
    scene. Fails OPEN (returns True) on any call/parse error."""
    if not image_prompts:
        return True
    joined = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(image_prompts))
    user_message = (
        f"ORIGINAL REQUEST:\n{original_prompt}\n\n"
        f"GENERATED CONTENT:\n{generated_content}\n\n"
        f"IMAGE PROMPT(S):\n{joined}"
    )
    try:
        raw = llm._call_llm(_IMAGE_PROMPT_RELEVANCE_SYSTEM_PROMPT, user_message)
        data = json.loads(_strip_code_fences(raw))
        is_relevant = bool(data.get("is_relevant", True))
        if not is_relevant:
            logger.info("Image prompt relevance check FAILED. Reason: %s", data.get("reason"))
        return is_relevant
    except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
        logger.error("check_image_prompt_relevance(): call/parse failed (%s) — failing open.", exc)
        return True


_INFOGRAPHIC_TEXT_COMPLETENESS_SYSTEM_PROMPT = """You check an image-generation prompt written for an INFOGRAPHIC, specifically for a common failure mode: the prompt describes several distinct text elements (a title, a heading, and especially a list of numbered/bulleted items), but only SOME of them are given actual literal text in quotes — the rest are left as vague style/length instructions ("a short description, no longer than 4 words", "wide-set bold lettering") with no real content specified. When this happens, an image renderer has nothing real to draw for those elements and typically renders fragments of the STYLE INSTRUCTION ITSELF as if it were the label (e.g. literally rendering the words "no longer than 4 words" onto the image instead of an actual sign/tip).

Respond with ONLY a single JSON object: {"is_complete": true/false, "missing_elements": ["short description of each element lacking real quoted text", ...]}

is_complete is false if the prompt describes ANY distinct text element (a title, heading, list item, label, callout, etc.) WITHOUT providing its actual literal text in quotes for that specific element — even if OTHER elements in the same prompt do have real quoted text. Having a properly quoted title is not sufficient if the list items beneath it are left as generic style/length descriptions with no real words specified.

is_complete is true only if EVERY distinct text element the prompt describes has its own actual literal text already written out in quotes — nothing left for the renderer to invent."""


def check_infographic_text_completeness(llm: BaseLLMService, image_prompts: List[str]) -> bool:
    """Infographic-specific check (only meaningful for content_type ==
    "Infographic" — callers should gate accordingly) catching a distinct
    failure mode from check_image_prompt_relevance() above: not a wrong
    SUBJECT, but MISSING literal text for one or more described elements
    (typically list items), causing the renderer to draw fragments of a
    style/length instruction as if it were the actual label — e.g. "no
    longer than 4 words" or "wide-set lettering" appearing verbatim on
    the rendered image instead of real content.

    This is a real, observed failure mode (confirmed from an actual
    rendered image during this project, not speculation) that the
    existing code-level safety net
    (content_generation_engine.py's infographic_prompt_has_literal_text())
    does NOT catch, because that check only requires ANY quoted text to
    exist somewhere in the prompt — it passes as soon as a title/heading/
    callout is properly quoted, even if list items beneath them are not.
    This function checks EVERY described element individually instead.

    Fails open (returns True) on any call/parse error — this is an
    additional precision check, never something that should block
    generation entirely if it breaks."""
    if not image_prompts:
        return True
    joined = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(image_prompts))
    try:
        raw = llm._call_llm(_INFOGRAPHIC_TEXT_COMPLETENESS_SYSTEM_PROMPT, joined)
        data = json.loads(_strip_code_fences(raw))
        is_complete = bool(data.get("is_complete", True))
        if not is_complete:
            logger.info("Infographic text-completeness check FAILED. Missing: %s", data.get("missing_elements"))
        return is_complete
    except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
        logger.error("check_infographic_text_completeness(): call/parse failed (%s) — failing open.", exc)
        return True


STRICT_REGENERATION_SUFFIX = (
    "\n\nIMPORTANT: The previous result did not sufficiently follow the user's original "
    "request. Regenerate using ONLY the topic, intent, requirements, and visual instructions "
    "from the original user prompt below. Do not introduce unrelated topics. Do not reframe "
    "or substitute the literal subject with a different scenario, industry, or setting — for "
    "example, if the topic is 'car', the content must be about cars/driving, NOT aviation, "
    "pilots, or any other vehicle/industry substituted in its place. Keep the exact subject "
    "the user named as the actual subject of the content throughout."
)

STRICT_INFOGRAPHIC_TEXT_SUFFIX = (
    "\n\nIMPORTANT: The previous image prompt left one or more text elements (a title, "
    "heading, list item, or callout) without actual literal text — only a vague style or "
    "length instruction (e.g. 'a short description, no longer than 4 words') with no real "
    "words specified. Regenerate the image prompt so EVERY distinct text element has its own "
    "actual literal text written out in quotes, with real content (not a placeholder or a "
    "restated style/length instruction) for each one, including every numbered/bulleted item."
)

# Retry limit shared by content-relevance regeneration (see
# app/routes/content.py) — one retry only, per the explicit "do not
# create an infinite regeneration loop" / performance requirements.
_INDUSTRY_SUGGESTION_SYSTEM_PROMPT = """You decide whether a user's content-generation topic clearly implies a specific real-world professional/industry setting, from a fixed list. You will be given the TOPIC and the list of AVAILABLE INDUSTRIES.

Respond with ONLY a single JSON object: {"industry": "<exact name from the list>" or null}

Rules:
- Return the exact industry name (character-for-character from the list) ONLY if the topic clearly, unambiguously implies that real-world setting — e.g. "car"/"driving"/"vehicle maintenance" clearly implies a transportation/fleet setting; "patient care" clearly implies healthcare; "warehouse forklift safety" clearly implies warehouse & logistics.
- Return null if the topic is genuinely generic/industry-agnostic (e.g. "workplace safety", "fire safety", "cybersecurity", "healthy eating") — these are intentionally meant to be illustrated across ANY of the industries for variety, not pinned to one. Do not force a match that isn't clearly there.
- Return null if the topic could plausibly fit several very different industries with no clear single best fit.
- When genuinely unsure, prefer null over guessing — a wrong forced match is worse than staying generic."""


def suggest_consistent_industry(llm: BaseLLMService, topic: str, available_industries: List[str]) -> Optional[str]:
    """General-purpose replacement for keyword-list-based industry
    biasing: rather than maintaining an ever-incomplete hand-curated
    list of keywords per industry (which will always have gaps for
    whatever specific topic wasn't anticipated — e.g. "car" not being in
    a keyword list, causing a random, potentially wildly mismatched
    industry like Aviation to get picked for it), this asks the LLM
    directly whether ANY topic clearly implies one of the fixed
    industries. Works the same way regardless of what the topic actually
    is — no topic-specific data to maintain.

    Returns the industry name to pass as ContentGenerationEngine.generate()'s
    `industry` param (taking top priority over random selection — see
    that method's priority chain) when there's a clear match, or None
    when the topic is genuinely industry-agnostic (in which case the
    EXISTING random/keyword-hint selection in
    app/services/prompt_builder.py runs unchanged, preserving the
    intentional content-variety behavior for generic topics).

    Fails toward None (i.e. "let the existing random selection run") on
    any call/parse error — this is a nice-to-have precision improvement,
    never something that should block generation if it breaks."""
    industries_list = "\n".join(f"- {name}" for name in available_industries)
    user_message = f"TOPIC:\n{topic}\n\nAVAILABLE INDUSTRIES:\n{industries_list}"
    try:
        raw = llm._call_llm(_INDUSTRY_SUGGESTION_SYSTEM_PROMPT, user_message)
        data = json.loads(_strip_code_fences(raw))
        industry = data.get("industry")
        if industry and industry in available_industries:
            return industry
        return None
    except Exception as exc:  # noqa: BLE001 - precision improvement only, never fatal
        logger.error("suggest_consistent_industry(): call/parse failed (%s) — falling back to random selection.", exc)
        return None


_KB_TOPIC_SUGGESTION_SYSTEM_PROMPT = """You will be given several excerpts from a Knowledge Base. Extract 3-5 short, distinct topic names that these excerpts are actually about — concise noun phrases a user could type as a content request (e.g. "Fire Safety", "Forklift Operation", "Chemical Handling"). Respond with ONLY a single JSON object: {"topics": ["...", "..."]}. If the excerpts are too sparse/repetitive to identify several distinct topics, return fewer — never invent a topic that isn't actually reflected in the excerpts."""


def suggest_kb_topics(llm: BaseLLMService, sample_chunks: List[dict]) -> List[str]:
    """Used only to make a "no meaningful match" response more useful
    (PART 3: show helpful suggestions) — extracts a handful of REAL
    topic names from a broad sample of the Knowledge Base, so a rejected
    user sees "here's what you CAN ask about" instead of a dead-end
    "try another prompt." Not exhaustive — just a representative sample
    from whatever chunks were retrieved. Fails open (empty list) on any
    call/parse error, since this is a nice-to-have, not load-bearing."""
    if not sample_chunks:
        return []
    excerpts = "\n\n".join(f"[{i+1}] {c.get('text', '')[:400]}" for i, c in enumerate(sample_chunks[:8]))
    try:
        raw = llm._call_llm(_KB_TOPIC_SUGGESTION_SYSTEM_PROMPT, excerpts)
        data = json.loads(_strip_code_fences(raw))
        topics = data.get("topics") or []
        return [t for t in topics if isinstance(t, str) and t.strip()][:5]
    except Exception as exc:  # noqa: BLE001
        logger.error("suggest_kb_topics(): call/parse failed (%s) — returning no suggestions.", exc)
        return []


MAX_RELEVANCE_RETRIES = 1


# --- Knowledge Base match classification (exact / similar / no match) ------

_KB_MATCH_SYSTEM_PROMPT = """You classify how a Knowledge Base's retrieved content relates to a user's content-generation request. You will be given the USER PROMPT, the TOP RETRIEVAL SCORE (a 0.0-1.0 semantic similarity score from the vector search that already found these excerpts), and the top retrieved KNOWLEDGE BASE EXCERPTS (may be empty).

Respond with ONLY a single JSON object:
{"match_type": "exact" | "similar" | "none", "matched_topic": "short topic name or null", "reason": "short reason"}

THE CORE QUESTION THAT DECIDES exact vs. similar: does using this Knowledge Base content require CHANGING the user's actual subject to something else, or does it genuinely SUPPORT the subject they already asked about?
- If the excerpts are relevant supporting information for the SAME underlying subject/domain the user asked about — even if the exact wording isn't there, even if the excerpts are more generic than the user's specific framing, even if the user added a persona/industry/scenario the excerpts don't spell out verbatim — that is "exact". The user's subject is NOT changing; the Knowledge Base is just supplying supporting facts/context/terminology for it. Do NOT require the excerpts to already contain the user's exact phrase, persona, or industry framing — a general resource on the user's actual topic (e.g. generic material on a behavior/state/hazard the user asked about, applied to whatever specific scenario they described) is exactly what "relevant supporting knowledge" means, not a reason to downgrade.
- Only use "similar" when the Knowledge Base has NOTHING relevant to the user's actual subject, but happens to contain a genuinely DIFFERENT subject that could plausibly be offered as an alternative instead — i.e. generating from it would mean swapping the user's topic for a different one, not merely supporting the one they asked for. matched_topic must name that different topic concisely.
- Use "none" only when there is no relevant support for the user's subject AND no plausible alternative topic either.

Do NOT ask for approval ("similar") merely because exact wording, exact phrasing, or the user's specific persona/industry/scenario framing isn't literally present in the excerpts — that is normal, expected supporting-knowledge behavior and counts as "exact". Reserve "similar" strictly for genuine subject substitution.

CRITICAL — trust a strong retrieval score as real evidence, and do not let score alone force a topic change either direction:
The TOP RETRIEVAL SCORE already reflects genuine semantic similarity computed by the vector search. A high score (roughly 0.5+) is strong evidence the excerpts are relevant support for the user's actual subject — treat that as evidence FOR "exact", not something to second-guess by demanding stricter textual overlap than the score already shows. Conversely, do not let a high score alone justify changing the user's subject to whatever the top-scoring document happens to be about if that document is actually a different subject — score indicates relevance, not that the subject should change.

Be honest and conservative only about GENUINE subject mismatches (e.g. user asked for a cake recipe, excerpts are about workplace safety — that's "none" regardless of score) — but be generous everywhere else: differing terminology, missing scenario framing, and generic-vs-specific phrasing are all still "exact" when the underlying subject is the same."""


class KBMatchResult(TypedDict):
    match_type: str  # "exact" | "similar" | "none"
    matched_topic: Optional[str]
    reason: str


def classify_kb_match(llm: BaseLLMService, prompt: str, chunks: List[dict]) -> KBMatchResult:
    """PART 2/3: classifies retrieved Knowledge Base chunks against the
    user's prompt into exact/similar/none, WITHOUT generating anything.
    Called by POST /api/check-topic before content generation ever
    starts — a "similar" result must get explicit user approval (PART 4)
    before any generation call is made; "none" stops generation
    entirely; "exact" proceeds directly. On any parse/call failure, this
    fails toward "exact" (proceed normally) rather than blocking every
    request on a broken classifier — logged loudly either way, since
    failing toward "none" would incorrectly block ALL generation on an
    infrastructure hiccup, which is a worse failure mode for this
    specific check than the (rare, self-correcting on the next request)
    alternative."""
    if not chunks:
        return {"match_type": "none", "matched_topic": None, "reason": "Knowledge Base returned no results."}

    # similar_match_score_floor is a coarse sanity floor (see
    # validation_rules.txt) — if EVERY retrieved chunk's own relevance
    # score is below it, don't even ask the LLM to consider "similar";
    # go straight to "none". Chunks without a score (older/other KB
    # backends) skip this floor entirely rather than being penalized for
    # a field that isn't populated.
    floor = float(get_rule("KNOWLEDGE_BASE_RELEVANCE", "similar_match_score_floor", "0.35"))
    scored = [c.get("score") for c in chunks if isinstance(c.get("score"), (int, float))]
    if scored and max(scored) < floor:
        return {"match_type": "none", "matched_topic": None, "reason": "All retrieved chunks scored below the relevance floor."}

    excerpts = "\n\n".join(f"[{i+1}] {c.get('text', '')[:600]}" for i, c in enumerate(chunks[:5]))
    top_score = max(scored) if scored else None
    score_line = f"{top_score:.3f}" if top_score is not None else "not available"
    user_message = f"USER PROMPT:\n{prompt}\n\nTOP RETRIEVAL SCORE:\n{score_line}\n\nKNOWLEDGE BASE EXCERPTS:\n{excerpts}"
    try:
        raw = llm._call_llm(_KB_MATCH_SYSTEM_PROMPT, user_message)
        data = json.loads(_strip_code_fences(raw))
        match_type = data.get("match_type")
        if match_type not in ("exact", "similar", "none"):
            raise ValueError(f"unexpected match_type: {match_type!r}")
        return {
            "match_type": match_type,
            "matched_topic": data.get("matched_topic"),
            "reason": data.get("reason", ""),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("classify_kb_match(): call/parse failed (%s) — failing toward 'exact' (proceed).", exc)
        return {"match_type": "exact", "matched_topic": None, "reason": "classification unavailable, proceeding normally"}


def _build_meaningless_warning(llm: BaseLLMService, kb_sample_fn: Optional[Callable[[], List[dict]]]) -> str:
    """Builds the "not a content-generation request" message. Prefers
    REAL topic suggestions sampled from the Knowledge Base (same
    mechanism as the "no KB match" case in POST /api/check-topic — see
    suggest_kb_topics()) over the generic static template, per the
    explicit requirement to suggest actual usable topics. Falls back to
    validation_rules.txt's generic template examples ONLY if no sampler
    is given or KB sampling/suggestion fails for any reason — better to
    show a generic example than no example at all, but a real one is
    always preferred when available."""
    base_message = get_rule(
        "MEANINGFUL_WORDS", "warning",
        "Please provide a specific topic or content-generation request.",
    )
    if kb_sample_fn is not None:
        try:
            sample_chunks = kb_sample_fn()
            suggested_topics = suggest_kb_topics(llm, sample_chunks)
            if suggested_topics:
                return base_message + " For example, you could ask about: " + ", ".join(suggested_topics) + "."
        except Exception as exc:  # noqa: BLE001 - enrichment only, never fatal
            logger.warning("Failed to sample Knowledge Base for meaningless-input suggestions: %s", exc)
    fallback = get_rule(
        "MEANINGFUL_WORDS", "fallback_examples",
        "For example: \"Create an infographic about [topic]\", \"Create a poster about [topic]\".",
    )
    return base_message + " " + fallback


def run_input_validation(
    llm: BaseLLMService, raw_prompt: str,
    kb_sample_fn: Optional[Callable[[], List[dict]]] = None,
    available_industries: Optional[List[str]] = None,
) -> tuple:
    """The full pre-generation validation flow in one call: basic ->
    special-char -> meaningful (heuristic fast-path, then LLM) ->
    spelling correction -> moderation -> understanding (+ industry
    suggestion, if available_industries is given — see analyze_prompt()'s
    docstring for why this is folded in here instead of a separate call).
    Raises ValidationError for any rejection. On success, returns
    (corrected_prompt, spelling_warning_or_None, intent_dict) — callers
    use corrected_prompt for actual generation while keeping raw_prompt
    around as the source of truth for later relevance checks. intent_dict
    now includes "suggested_industry" when available_industries was given.

    kb_sample_fn, if given, is called ONLY when input fails the
    meaningful-content-request check (never for empty/special-chars/
    numbers-only, which don't need topic suggestions to be understood)
    — see _build_meaningless_warning() above. Passing None (the default)
    keeps this function fully decoupled from any Knowledge Base/AWS
    dependency, e.g. for tests."""
    if is_empty_or_blank(raw_prompt):
        raise ValidationError(
            get_rule("EMPTY_INPUT", "warning", "Please enter a prompt before generating content."),
            error_type="empty",
        )

    stripped = raw_prompt.strip()
    if is_special_characters_only(stripped):
        raise ValidationError(
            get_rule(
                "SPECIAL_CHARACTERS", "warning",
                "⚠️ Invalid prompt: Special characters alone are not allowed. Please enter meaningful words.",
            ),
            error_type="special_chars",
        )
    if is_numbers_only(stripped):
        raise ValidationError(
            get_rule(
                "NUMBERS_ONLY", "warning",
                "⚠️ Invalid prompt: Numbers alone are not a meaningful content request.",
            ),
            error_type="numbers_only",
        )
    if is_obvious_gibberish(stripped):
        raise ValidationError(_build_meaningless_warning(llm, kb_sample_fn), error_type="meaningless")

    analysis = analyze_prompt(llm, stripped, kb_sample_fn, available_industries)  # raises ValidationError for meaningless/moderation
    warning = format_spelling_warning(analysis["corrections"])
    return analysis["corrected_prompt"], warning, analysis["intent"]