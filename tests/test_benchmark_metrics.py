"""Pure metric checks: no backend, model, device, or GUI is started."""

import pytest

from scripts.replay_benchmark import (
    edit_distance,
    normalized_tokens,
    percentile,
    quality_score,
    summarize_latency,
)


def test_english_normalization_preserves_words_and_normalizes_apostrophes():
    assert normalized_tokens("WE’LL send Version 2 — don't WAIT!") == [
        "we'll", "send", "version", "2", "don't", "wait"
    ]
    assert normalized_tokens("  HELLO,\nworld...  ", "en") == ["hello", "world"]
    assert normalized_tokens("…！？ —", "en") == []


def test_chinese_normalization_uses_characters_including_mixed_latin_text():
    assert normalized_tokens("我们用 GPU 4070！", "zh") == [
        "我", "们", "用", "g", "p", "u", "4", "0", "7", "0"
    ]
    assert normalized_tokens("你好，世界。\n", "zh") == list("你好世界")
    assert normalized_tokens("，。！？", "zh") == []


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ([], [], 0),
        (["one", "two"], [], 2),
        ([], ["one", "two"], 2),
        (["same"], ["same"], 0),
        (["one", "two"], ["one", "too"], 1),
        (["a", "b", "c"], ["a", "x", "c", "d"], 2),
        (list("今天开会"), list("今天会"), 1),
        (list("你好"), list("您好"), 1),
    ],
)
def test_edit_distance_counts_known_edits(reference, hypothesis, expected):
    assert edit_distance(reference, hypothesis) == expected


def test_word_error_rate_counts_insertions_and_substitutions():
    assert quality_score("we ship friday", "we shipped on friday", "en") == {
        "metric": "WER",
        "errors": 2,
        "reference_units": 3,
        "error_rate": pytest.approx(2 / 3),
        "empty_output": False,
    }


def test_character_error_rate_counts_unicode_deletion():
    assert quality_score("今天开会。", "今天会", "zh") == {
        "metric": "CER",
        "errors": 1,
        "reference_units": 4,
        "error_rate": 0.25,
        "empty_output": False,
    }


@pytest.mark.parametrize("observed", ["", "   \n", "…!?"])
def test_empty_recognition_is_all_deletions_not_a_success(observed):
    result = quality_score("hello world", observed, "en")
    assert result["errors"] == result["reference_units"] == 2
    assert result["error_rate"] == 1.0
    assert result["empty_output"] is True


@pytest.mark.parametrize("observed,errors,empty", [("", 0, True), ("extra", 1, False)])
def test_empty_reference_has_no_defined_error_rate(observed, errors, empty):
    result = quality_score("", observed, "en")
    assert result["reference_units"] == 0
    assert result["errors"] == errors
    assert result["error_rate"] is None
    assert result["empty_output"] is empty


def test_wer_can_exceed_one_when_recognition_adds_words():
    assert quality_score("one", "one two three", "en")["error_rate"] == 2.0


def test_percentiles_handle_empty_singleton_and_linear_interpolation():
    assert percentile([], 50) is None
    for quantile in (0, 50, 95, 100):
        assert percentile([7.5], quantile) == 7.5
    values = [40.0, 10.0, 30.0, 20.0]
    assert percentile(values, 0) == 10.0
    assert percentile(values, 50) == 25.0
    assert percentile(values, 95) == pytest.approx(38.5)
    assert percentile(values, 100) == 40.0


def test_latency_summary_counts_each_metric_independently():
    result = summarize_latency([
        dict(caption_delay_ms=10, latency_ms=20, processing_ms=8,
             queue_wait_ms=2, first_caption_delay_ms=11),
        dict(caption_delay_ms=30, latency_ms=60, processing_ms=16, queue_wait_ms=6),
        dict(caption_delay_ms=50, latency_ms=100, processing_ms=24,
             queue_wait_ms=10, first_caption_delay_ms=51),
    ])
    expected = {
        "caption_delay_ms": dict(count=3, p50=30, p95=48, max=50),
        "latency_ms": dict(count=3, p50=60, p95=96, max=100),
        "processing_ms": dict(count=3, p50=16, p95=23.2, max=24),
        "queue_wait_ms": dict(count=3, p50=6, p95=9.6, max=10),
        "first_caption_delay_ms": dict(count=2, p50=31, p95=49, max=51),
    }
    assert result.keys() == expected.keys()
    for metric, statistics in expected.items():
        assert result[metric] == pytest.approx(statistics)


def test_latency_summary_ignores_nonfinite_nonnumeric_and_boolean_samples():
    invalid = [None, "12", True, False, float("nan"), float("inf"), -float("inf"), [], {}]
    events = [dict(latency_ms=value, processing_ms=value) for value in invalid]
    events += [{}, {"irrelevant": 99}, {"latency_ms": 12.5}]
    assert summarize_latency(events) == {
        "latency_ms": {"count": 1, "p50": 12.5, "p95": 12.5, "max": 12.5}
    }
    assert summarize_latency([]) == {}


def test_invalid_caption_delay_does_not_discard_other_event_metrics():
    result = summarize_latency([
        {"caption_delay_ms": 1000, "caption_delay_valid": False, "processing_ms": 20},
        {"caption_delay_ms": 10, "caption_delay_valid": True},
        {"caption_delay_ms": 30},
    ])
    assert result["caption_delay_ms"] == pytest.approx(
        {"count": 2, "p50": 20, "p95": 29, "max": 30}
    )
    assert result["processing_ms"] == {
        "count": 1, "p50": 20, "p95": 20, "max": 20
    }


@pytest.mark.parametrize("metric", [
    "asr_ms", "speaker_ms", "worker_ms", "final_caption_delay_ms",
    "caption_delay_from_speech_end_ms", "caption_delay_from_utterance_start_ms",
])
def test_engine_timing_fields_are_summarized(metric):
    assert summarize_latency([{metric: 10}, {metric: 30}])[metric] == pytest.approx(
        {"count": 2, "p50": 20, "p95": 29, "max": 30}
    )
