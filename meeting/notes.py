"""Offline, extractive meeting-note suggestions; no model or network calls.

Suggestions deliberately retain whole transcript segments. Keyword matches are
only navigation aids: they cannot establish an agreement, owner, or deadline.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any


_ACTION_CUES = re.compile(
    r"\b(?:action\s+items?|next\s+steps?|follow[ -]?up|to[ -]?do|"
    r"(?:i|we)[’']ll|(?:i|we)\s+will|(?:need|needs|have|has)\s+to|"
    r"(?:can|could|would)\s+you|please|let[’']s|should|must|going\s+to)\b"
    r"|我(?:们)?会|我们将|需要|请|麻烦|负责|跟进|下一步|待办|行动项"
    r"|计划|记得|安排|确认|发送|提交|完成|截止",
    re.IGNORECASE,
)
_NOTE_CUES = re.compile(
    r"\b(?:decid(?:e|ed)|decision|agree(?:d|ment)?|important|risk|"
    r"blocker|concern|question|proposal|recommend|conclusion)\b"
    r"|决定|决策|同意|结论|重点|关键|风险|阻碍|问题|建议|方案",
    re.IGNORECASE,
)
_MAX_QUOTE_NOTES = 8
_MAX_ACTION_NOTES = 12


def _segments(
    segments: Iterable[Mapping[str, Any]] | None,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Ignore blank or malformed entries without changing transcript text."""
    if segments is None:
        return []
    return [
        (index, segment)
        for index, segment in enumerate(segments)
        if isinstance(segment, Mapping)
        and isinstance(segment.get("text"), str)
        and segment["text"].strip()
    ]


def _seconds(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _time(value: Any) -> str:
    seconds = _seconds(value)
    if seconds is None:
        return "unknown"
    milliseconds = round(seconds * 1000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds_part = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


def _label(value: Any) -> str:
    """Escape provenance metadata so it cannot turn into Markdown markup."""
    if value is None or str(value).strip() == "":
        return "unknown"
    text = " ".join(str(value).split())
    return re.sub(r"([\\`*_{}\[\]()#+.!|<>~-])", r"\\\1", text)


def _provenance(index: int, segment: Mapping[str, Any]) -> str:
    start = segment.get("start", segment.get("start_seconds"))
    end = segment.get("end", segment.get("end_seconds"))
    segment_id = segment.get("segment_id", segment.get("id"))
    identity = f"transcript segment {index + 1}"
    if segment_id is not None:
        identity += f" (ID: {_label(segment_id)})"
    return (
        f"{_time(start)} – {_time(end)} | "
        f"Speaker: {_label(segment.get('speaker'))} | "
        f"Source: {_label(segment.get('source'))} | {identity}"
    )


def _quote(text: str) -> str:
    """Fence whole text verbatim, including text containing Markdown fences."""
    longest_fence = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest_fence + 1)
    # The final newline separates the quote from the closing fence; no content
    # is stripped, normalized, paraphrased, or truncated.
    separator = "" if text.endswith("\n") else "\n"
    return f"{fence}text\n{text}{separator}{fence}"


def suggest_actions(
    segments: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return action-language candidates in original transcript order.

    Input fields: ``text`` (required), ``start``/``end`` (seconds), ``speaker``,
    ``source``, and optional ``id``/``segment_id``. ``start_seconds`` and
    ``end_seconds`` are accepted aliases. Empty/invalid entries are skipped.

    ``quote`` and ``text`` contain the exact original text; ``segment_index``
    is zero-based in the input. All candidates require human verification.
    ``owner`` and ``deadline`` are always None, even when mentioned in a quote:
    this function does not infer commitments or parse responsibility/dates.
    """
    candidates: list[dict[str, Any]] = []
    for index, segment in _segments(segments):
        text = segment["text"]
        matches = list(_ACTION_CUES.finditer(text))
        if not matches:
            continue
        candidates.append(
            {
                "text": text,
                "quote": text,
                "segment_index": index,
                "segment_id": segment.get("segment_id", segment.get("id")),
                "start": segment.get("start", segment.get("start_seconds")),
                "end": segment.get("end", segment.get("end_seconds")),
                "speaker": segment.get("speaker"),
                "source": segment.get("source"),
                "matched_cues": list(dict.fromkeys(match.group() for match in matches)),
                "status": "candidate",
                "needs_review": True,
                "owner": None,
                "deadline": None,
            }
        )
    return candidates


def suggest_notes(segments: Iterable[Mapping[str, Any]] | None) -> str:
    """Build deterministic Markdown suggestions from transcript quotations.

    Up to eight quote candidates favor segments containing decision, issue, or
    proposal language; ties use transcript order. Up to twelve action-language
    candidates are shown in transcript order. These are excerpts for review,
    not a comprehensive summary. Missing provenance is explicitly unknown.
    """
    # Materialize once so generators work and original positions are retained.
    materialized = list(segments) if segments is not None else []
    valid = _segments(materialized)
    lines = [
        "# Meeting note candidates / 会议笔记候选",
        "",
        "Generated locally from transcript quotes using English/Chinese keyword rules. "
        "Verify every candidate against the recording and surrounding context. "
        "These excerpts are not a complete summary or confirmed decisions/actions. "
        "Transcription and speaker labels may contain errors.",
        "",
        "以下内容为原文摘录候选，请核对录音与上下文；不自动认定决定、负责人或截止日期。",
        "",
    ]
    if not valid:
        lines.extend(["No transcript text available / 暂无可用转录文本。", ""])
        return "\n".join(lines)

    ranked = sorted(
        valid,
        key=lambda entry: (not bool(_NOTE_CUES.search(entry[1]["text"])), entry[0]),
    )
    selected = sorted(ranked[:_MAX_QUOTE_NOTES], key=lambda entry: entry[0])
    lines.extend(
        [
            "## Quote candidates / 摘录候选",
            "",
            f"Showing {len(selected)} of {len(valid)} nonempty transcript segments; "
            "selected quotes retain their original transcript order.",
            "",
        ]
    )
    for number, (index, segment) in enumerate(selected, 1):
        lines.extend(
            [
                f"### Quote candidate {number} — verify / 请核实",
                "",
                _provenance(index, segment),
                "",
                _quote(segment["text"]),
                "",
            ]
        )

    actions = suggest_actions(materialized)
    lines.extend(
        [
            "## Possible action items / 待核实行动项",
            "",
            "Keyword matches can include questions, suggestions, negations, or past events. "
            "Confirm whether an action was agreed and enter any owner/deadline yourself.",
            "",
        ]
    )
    if not actions:
        lines.extend(
            [
                "No action-language matches found. This does not establish that there "
                "were no action items / 未匹配到行动用语，不代表没有行动项。",
                "",
            ]
        )
    else:
        lines.extend([f"Showing {min(len(actions), _MAX_ACTION_NOTES)} of {len(actions)} candidates.", ""])
        for number, candidate in enumerate(actions[:_MAX_ACTION_NOTES], 1):
            lines.extend(
                [
                    f"### Action candidate {number} — unverified / 未核实",
                    "",
                    _provenance(candidate["segment_index"], candidate),
                    "",
                    _quote(candidate["quote"]),
                    "",
                    "Owner: unassigned; deadline: unassigned. Verify from the source before assigning.",
                    "",
                ]
            )
    return "\n".join(lines)
