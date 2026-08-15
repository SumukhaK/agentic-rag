from unittest.mock import patch

from agentic_rag.evaluation.judge import FaithfulnessCheckResult, check_faithfulness

KWARGS = dict(model="qwen2.5:14b-instruct", base_url="http://localhost:11434", timeout=60, temperature=0.0)


@patch("agentic_rag.orchestration.judge.generate")
def test_check_faithfulness_clears_a_response_starting_with_clean(mock_generate):
    mock_generate.return_value = "CLEAN"

    result = check_faithfulness(
        "Who won?", "Arsenal won [1].", "[1] Arsenal beat Chelsea 3-0.", **KWARGS
    )

    assert result == FaithfulnessCheckResult(is_faithful=True, raw_judge_response="CLEAN")


@patch("agentic_rag.orchestration.judge.generate")
def test_check_faithfulness_flags_a_response_not_starting_with_clean(mock_generate):
    mock_generate.return_value = "UNSUPPORTED"

    result = check_faithfulness(
        "Who won?", "Arsenal won 5-0 [1].", "[1] Arsenal beat Chelsea 3-0.", **KWARGS
    )

    assert result == FaithfulnessCheckResult(is_faithful=False, raw_judge_response="UNSUPPORTED")


@patch("agentic_rag.orchestration.judge.generate")
def test_check_faithfulness_passes_query_answer_and_sources_into_the_prompt(mock_generate):
    mock_generate.return_value = "CLEAN"

    check_faithfulness(
        "Who won the derby?",
        "Arsenal won [1].",
        "[1] Arsenal beat Chelsea 3-0 on Saturday.",
        **KWARGS,
    )

    prompt = mock_generate.call_args.args[0]
    assert "Who won the derby?" in prompt
    assert "Arsenal won [1]." in prompt
    assert "Arsenal beat Chelsea 3-0 on Saturday." in prompt


@patch("agentic_rag.orchestration.judge.generate")
def test_check_faithfulness_fails_closed_on_an_unparseable_response(mock_generate):
    mock_generate.return_value = ""

    result = check_faithfulness("q", "a", "s", **KWARGS)

    assert result.is_faithful is False
