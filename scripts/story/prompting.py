import ast
import json
import re
from typing import Any


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_MISSING_COMMA_OBJ_RE = re.compile(r"\}\s*\{")
_MISSING_COMMA_ARR_OBJ_RE = re.compile(r"\]\s*\{")
_MISSING_COMMA_KEY_RE = re.compile(r'(["\d\]\}])\s*\n\s*("(?:identity|style|frames|outfit|action|setting|camera|edit|subject_label)"\s*:)')
_CLOTHING_TAIL_RE = re.compile(
    r",?\s*(?:and\s+)?(?:she|he|they|it)\s+(?:wears?|is wearing|has on)\s+"
    r"(?:a|an|the|some|matching)?\s*(?:coat|jacket|tunic|shirt|blouse|dress|suit|"
    r"hoodie|sweater|cardigan|trousers|pants|jeans|skirt|boots|shoes|hat|scarf|gloves|armor|robe|cloak)\b.*$",
    re.IGNORECASE,
)
BANNED_SHORTHAND_RE = re.compile(
    r"\b("
    r"the same|same (?:woman|man|person|character|girl|boy|guy|lady|figure|subject|outfit|clothes|clothing|room|place|scene|angle|shot)|"
    r"previous (?:frame|slide|image|shot|prompt|scene)|"
    r"as before|once again|still|aforementioned|above-mentioned"
    r")\b",
    re.IGNORECASE,
)
STORY_STYLE_GUIDANCE: dict[str, dict[str, str]] = {
    "Default": {
        "pacing": "Clear, balanced visual progression across frames.",
        "visuals": "Natural, cohesive framing and lighting matched to the story and reference.",
        "camera": "Vary shot scale and angle naturally to match each story beat.",
    },
    "Cinematic": {
        "pacing": "Dramatic filmic arc with tension, visual payoff, and theatrical blocking.",
        "visuals": "Filmic lighting (motivated rim/key lights, depth, atmospheric haze), rich contrast, anamorphic/cinema composition.",
        "camera": "Use cinematic shot language such as wide establishing shots, dramatic low/high angles, over-the-shoulder shots, and tight close-ups.",
    },
    "Creative": {
        "pacing": "Inventive, imaginative progression with unexpected visual twists and playful staging.",
        "visuals": "Expressive color interplay, imaginative props/environments, and striking artistic flair.",
        "camera": "Use dynamic, unconventional angles, bold perspective shifts, and expressive framing.",
    },
    "Comic": {
        "pacing": "Punchy, high-impact visual beats designed like sequential comic panels.",
        "visuals": "Bold graphic outlines, clear figure-ground separation, strong focal silhouettes, and vibrant panel-ready readability.",
        "camera": "Use dynamic action angles, dramatic foreshortening, expressive character reactions, and clear focal hierarchy.",
    },
    "Documentary": {
        "pacing": "Grounded, observational, real-world progression capturing authentic moments.",
        "visuals": "Naturalistic available light, lived-in realistic environments, unposed candid body language, and authentic textures.",
        "camera": "Favor eye-level candid framing, medium/wide observational shots, and natural handheld-style perspectives.",
    },
}

def get_style_guidance(story_style: str | None) -> tuple[str, dict[str, str]]:
    story_style = (story_style or "Default").strip() or "Default"
    if story_style in STORY_STYLE_GUIDANCE:
        return story_style, STORY_STYLE_GUIDANCE[story_style]
    return story_style, {
        "pacing": f"Shape the story progression in a {story_style} tone.",
        "visuals": f"Infuse every frame with {story_style} visual atmosphere and styling.",
        "camera": f"Choose camera angles and framing that fit a {story_style} aesthetic.",
    }


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])\s*([,;:])", r"\1", cleaned)
    return cleaned.strip(" ,;")


def _sanitize_identity(identity: str) -> str:
    cleaned = _CLOTHING_TAIL_RE.sub("", identity or "")
    cleaned = BANNED_SHORTHAND_RE.sub("", cleaned)
    return _normalize_text(cleaned)


def _clean_frame_field(value: str, fallback: str = "") -> str:
    text = _normalize_text(value)
    if not text or BANNED_SHORTHAND_RE.search(text):
        text = BANNED_SHORTHAND_RE.sub("", text)
        text = _normalize_text(text)
        if not text:
            return _normalize_text(fallback)
    return text


def _join_prompt_clauses(parts: list[str]) -> str:
    clauses: list[str] = []
    for part in parts:
        p = _normalize_text(part)
        p = re.sub(r"[.]+$", "", p).strip()
        if p:
            clauses.append(p)
    if not clauses:
        return ""
    return ", ".join(clauses) + "."


def build_story_prompt(
    slide_count: int,
    is_edit_model: bool,
    use_reference: bool,
    has_control_image: bool,
    story_style: str,
) -> tuple[str, bool]:
    is_instruct_edit = bool(is_edit_model and use_reference and has_control_image)
    style_name, style_cfg = get_style_guidance(story_style)

    style_block = (
        f"Story Style: {style_name}\n"
        f"- Pacing & Narrative: {style_cfg['pacing']}\n"
        f"- Visual Direction: {style_cfg['visuals']}\n"
        f"- Camera Direction: {style_cfg['camera']}\n\n"
    )

    if is_instruct_edit:
        system = (
            "You plan visual story frames for an instruction-based image editor.\n"
            "Each frame instruction is applied independently to the original reference image.\n"
            f"{style_block}"
            f"Break the story into {slide_count} ordered frames and return ONLY valid JSON.\n\n"
            "Schema:\n"
            "{\n"
            '  "subject_label": "concise unambiguous visual identifier from the reference image (e.g., \'the woman in the red coat\'), never a pronoun",\n'
            '  "frames": [\n'
            '    {"edit": "concise concrete change to pose, action, expression, outfit, props, or camera angle relative to the reference image"}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            f"- `frames` must contain exactly {slide_count} items.\n"
            "- Keep each edit concise (1-2 short sentences) and describe only what changes.\n"
            f"- Apply the {style_name} story style to how moments and camera/action shifts are chosen.\n"
            "- Refer to the subject using `subject_label`, never 'she', 'he', 'they', 'the same', or 'previous'.\n"
            "- Output raw JSON only."
        )
    else:
        ref_note = (
            "Match the subject's physical identity and artistic medium visible in the supplied reference image.\n"
            if (use_reference and has_control_image)
            else ""
        )
        system = (
            "You plan visual story frames for a stateless image generator that has NO memory between frames.\n"
            f"{ref_note}"
            f"{style_block}"
            f"Break the story into {slide_count} ordered frames and return ONLY valid JSON.\n\n"
            "Schema:\n"
            "{\n"
            '  "identity": "unchanging physical traits ONLY: age, gender, face, skin, hair style/color, build (NO clothing or poses)",\n'
            f'  "style": "overall artistic medium, rendering style, color palette, and lighting mood reflecting {style_name} (NO camera angles)",\n'
            '  "frames": [\n'
            "    {\n"
            '      "outfit": "concise explicit clothing and worn accessories visible in this specific frame",\n'
            '      "action": "concise visible pose, expression, and action in this single moment (use explicit subject/character nouns)",\n'
            '      "setting": "concise visible environment, background, and props",\n'
            '      "camera": "concise shot framing and camera angle matching the style and story beat"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            f"- `frames` must contain exactly {slide_count} items.\n"
            "- Keep every field brief and concrete so prompts stay compact and clear.\n"
            "- Always state the full `outfit` explicitly in every frame, updating it when clothing changes (e.g., jacket removed).\n"
            f"- Reflect {style_name} in `style`, `action`, `setting`, and `camera`.\n"
            "- Never use 'the same', 'same woman/man/character/outfit', 'still', 'again', 'previous', or vague back-references.\n"
            "- Output raw JSON only."
        )

    return system, is_instruct_edit


def assemble_story_prompts(raw_llm_output: str, is_instruct_edit: bool) -> list[str]:
    data = parse_story_payload(raw_llm_output)
    raw_frames = data.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("Story: payload missing valid 'frames' list")

    prompts: list[str] = []

    if is_instruct_edit:
        label = _normalize_text(str(data.get("subject_label", "")))
        for frame in raw_frames:
            edit = frame.get("edit", "") if isinstance(frame, dict) else str(frame)
            edit = _normalize_text(str(edit))
            if label:
                edit = BANNED_SHORTHAND_RE.sub(label, edit)
            edit = _normalize_text(edit)
            if edit and not edit.endswith((".", "!", "?")):
                edit += "."
            prompts.append(edit)
    else:
        identity = _sanitize_identity(str(data.get("identity", "")))
        style = _normalize_text(str(data.get("style", "")))
        last_outfit = ""

        for frame in raw_frames:
            if isinstance(frame, dict):
                raw_outfit = str(frame.get("outfit", ""))
                raw_action = str(frame.get("action", ""))
                raw_setting = str(frame.get("setting", ""))
                raw_camera = str(frame.get("camera", ""))

                # Carry forward the last known explicit outfit if the LLM abbreviated or left it empty
                if not _normalize_text(raw_outfit) or BANNED_SHORTHAND_RE.search(raw_outfit):
                    outfit = last_outfit
                else:
                    outfit = _normalize_text(raw_outfit)
                    last_outfit = outfit

                action = _clean_frame_field(raw_action)
                setting = _clean_frame_field(raw_setting)
                camera = _clean_frame_field(raw_camera)

                subject_phrase = (f"{identity}, {outfit}" if (identity and outfit) else (identity or outfit))
                prompt = _join_prompt_clauses([subject_phrase, action, setting, camera, style])
                prompts.append(prompt)
            else:
                text = BANNED_SHORTHAND_RE.sub(identity, str(frame).strip())
                text = _join_prompt_clauses([text, style])
                prompts.append(text)

    return prompts


def parse_json(response):
    response = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', response.strip(), flags=re.IGNORECASE)
    last_error = None
    for match in re.finditer(r'[\[{]', response):
        start = match.start()
        stack = []
        quote = None
        escaped = False
        for index in range(start, len(response)):
            char = response[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char in '[{':
                stack.append(']' if char == '[' else '}')
            elif char in ']}':
                if not stack or char != stack.pop():
                    break
                if not stack:
                    candidate = response[start:index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError as e:
                        last_error = e
                    try:
                        return ast.literal_eval(candidate)
                    except (SyntaxError, ValueError) as e:
                        last_error = e
                    break
    if last_error is not None:
        raise ValueError(f'Story: LLM response did not contain valid JSON or a Python literal: {last_error}') from last_error
    raise ValueError('Story: LLM response did not contain a JSON value')


def parse_frame_prompts(response, slide_count):
    data = parse_json(response)
    if isinstance(data, dict):
        data = data.get('prompts', data.get('frames'))
    if not isinstance(data, list):
        raise ValueError('Story: LLM response must contain a list of frame prompts')
    prompts = []
    for item in data:
        if isinstance(item, str):
            prompt = item
        elif isinstance(item, dict):
            prompt = item.get('prompt')
        else:
            continue
        if isinstance(prompt, str) and prompt.strip():
            prompts.append(prompt.strip())
    if len(prompts) != slide_count:
        raise ValueError(f'Story: LLM returned {len(prompts)} frame prompts; expected {slide_count}')
    return prompts


def _extract_object_candidate(raw_text: str) -> str:
    s = _THINK_BLOCK_RE.sub("", raw_text or "").strip()
    s = _CODE_FENCE_RE.sub("", s).strip()

    # Case 1: Output started right after `"identity": ` (e.g. `"A young woman...", "style": ...}`)
    if s.startswith('"') and re.search(r'"\s*,\s*"(?:style|frames)"\s*:', s):
        s = '{"identity": ' + s
    # Case 2: Output started with `"identity": ...` without outer `{`
    elif re.match(r'^"?identity"?\s*:', s):
        s = "{" + s

    # Trim any leading text before first `{`
    first_brace = s.find("{")
    if first_brace > 0:
        s = s[first_brace:]

    # Repair missing commas and trailing commas from small-model output
    s = _MISSING_COMMA_OBJ_RE.sub("}, {", s)
    s = _MISSING_COMMA_ARR_OBJ_RE.sub("], {", s)
    s = _MISSING_COMMA_KEY_RE.sub(r"\1,\n\2", s)
    s = _TRAILING_COMMA_RE.sub(r"\1", s)

    # Balance unclosed `{` / `[` if generation truncated early
    if s.startswith("{"):
        open_braces = s.count("{") - s.count("}")
        open_brackets = s.count("[") - s.count("]")
        if open_brackets > 0:
            s += "]" * open_brackets
        if open_braces > 0:
            s += "}" * open_braces

    return s


def parse_story_payload(raw_llm_output: str) -> dict[str, Any]:
    candidate = _extract_object_candidate(raw_llm_output)

    # 1. Standard JSON (use raw_decode so trailing text after `}` doesn't fail)
    decoder = json.JSONDecoder()
    for text in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            obj, _ = decoder.raw_decode(text.lstrip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 2. Optional `json_repair` if available in the SD.Next environment
    try:
        import json_repair  # type: ignore

        repaired = json_repair.loads(candidate)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    # 3. Python literal fallback (handles single quotes / True / None if model emitted Python dict syntax)
    py_candidate = re.sub(r"\bnull\b", "None", candidate, flags=re.IGNORECASE)
    py_candidate = re.sub(r"\btrue\b", "True", py_candidate, flags=re.IGNORECASE)
    py_candidate = re.sub(r"\bfalse\b", "False", py_candidate, flags=re.IGNORECASE)
    try:
        obj = ast.literal_eval(py_candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    raise ValueError(f"Could not parse story JSON from output: {raw_llm_output[:300]!r}")
