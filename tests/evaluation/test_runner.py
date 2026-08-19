from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import PointStruct, SparseVector

from agentic_rag.config import Settings
from agentic_rag.embedding.sparse_client import SparseEmbeddingError, embed_sparse_texts
from agentic_rag.evaluation.judge import FaithfulnessCheckResult
from agentic_rag.evaluation.questions import EvalQuestion
from agentic_rag.evaluation.report import EvalQuestionResult, EvaluationReport
from agentic_rag.evaluation.runner import (
    EvaluationIndexingError,
    _fetch_cited_source_text,
    _report_path_for,
    _run_question,
    main,
    run_evaluation,
)
from agentic_rag.indexing.qdrant_setup import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    get_client,
)
from agentic_rag.indexing.upsert import _point_id
from agentic_rag.ingestion.pipeline import IngestionFailure
from agentic_rag.ingestion.scheduler import SyncCycleResult
from agentic_rag.orchestration.answer import AnswerResult, Citation
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE
from access_tiers import TIER_EMPLOYEE

SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = "eval_documents"


@pytest.fixture(scope="module", autouse=True)
def _require_sparse_model():
    try:
        embed_sparse_texts(["warmup"], model_name=SPARSE_MODEL)
    except SparseEmbeddingError as exc:
        pytest.skip(f"sparse embedding model unavailable: {exc}")


def _settings(tmp_path) -> Settings:
    return Settings(
        watched_folder_path=tmp_path / "corpus",
        qdrant_storage_path=tmp_path / "qdrant",
        sync_snapshot_path=tmp_path / "sync_snapshot.json",
        evaluation_corpus_path=tmp_path / "eval_corpus",
        evaluation_questions_path=tmp_path / "questions.json",
        evaluation_qdrant_storage_path=tmp_path / "eval_qdrant",
        evaluation_qdrant_collection_name=COLLECTION,
        sparse_embedding_model=SPARSE_MODEL,
        _env_file=None,
    )


def _question(**overrides) -> EvalQuestion:
    defaults = dict(
        id="q1",
        query="Who won the derby?",
        user_tier=TIER_EMPLOYEE,
        expected_answerable=True,
        expected_source_paths=["employee/derby.md"],
    )
    defaults.update(overrides)
    return EvalQuestion(**defaults)


def _empty_sync_result() -> SyncCycleResult:
    return SyncCycleResult(
        indexed=[], deleted=[], ingestion_failures=[], indexing_failures=[], deletion_failures=[]
    )


def _dummy_result(question_id="q1", error=None) -> EvalQuestionResult:
    return EvalQuestionResult(
        question_id=question_id,
        query="q",
        expected_answerable=True,
        expected_source_paths=["employee/a.md"],
        answer_text="a",
        cited_paths=["employee/a.md"],
        retrieval_hit=True,
        answered=True,
        faithfulness=FaithfulnessCheckResult(is_faithful=True, raw_judge_response="CLEAN"),
        hallucinated=False,
        error=error,
        duration_seconds=1.0,
    )


# --- _fetch_cited_source_text -----------------------------------------------


def test_fetch_cited_source_text_returns_the_indexed_chunk_text(tmp_path):
    client = get_client(tmp_path / "qdrant")
    ensure_collection(client, collection_name=COLLECTION, vector_size=3)
    citation = Citation(
        number=1, relative_path="employee/derby.md", chunk_index=0, access_tier=TIER_EMPLOYEE
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=_point_id(citation.relative_path, citation.chunk_index),
                vector={
                    DENSE_VECTOR_NAME: [0.1, 0.2, 0.3],
                    SPARSE_VECTOR_NAME: SparseVector(indices=[], values=[]),
                },
                payload={
                    "relative_path": "employee/derby.md",
                    "chunk_index": 0,
                    "text": "Arsenal beat Chelsea 3-0.",
                    "access_tier": TIER_EMPLOYEE,
                },
            )
        ],
    )

    text = _fetch_cited_source_text(client, COLLECTION, citation)

    assert text == "Arsenal beat Chelsea 3-0."


def test_fetch_cited_source_text_returns_empty_when_the_chunk_is_missing(tmp_path):
    client = get_client(tmp_path / "qdrant")
    ensure_collection(client, collection_name=COLLECTION, vector_size=3)
    citation = Citation(number=1, relative_path="employee/nope.md", chunk_index=0, access_tier=TIER_EMPLOYEE)

    text = _fetch_cited_source_text(client, COLLECTION, citation)

    assert text == ""


# --- _run_question -----------------------------------------------------------


@patch("agentic_rag.evaluation.runner.check_faithfulness")
@patch("agentic_rag.evaluation.runner._fetch_cited_source_text")
@patch("agentic_rag.evaluation.runner.answer_with_cache")
def test_run_question_marks_a_retrieval_hit_when_an_expected_source_is_cited(
    mock_answer, mock_fetch_text, mock_faithfulness, tmp_path
):
    mock_answer.return_value = AnswerResult(
        text="Arsenal won [1].",
        citations=[Citation(number=1, relative_path="employee/derby.md", chunk_index=0, access_tier=TIER_EMPLOYEE)],
    )
    mock_fetch_text.return_value = "Arsenal beat Chelsea 3-0."
    mock_faithfulness.return_value = FaithfulnessCheckResult(is_faithful=True, raw_judge_response="CLEAN")
    settings = _settings(tmp_path)

    result = _run_question(_question(), settings=settings, client=object(), embedding_cache=object())

    assert result.retrieval_hit is True
    assert result.answered is True
    assert result.hallucinated is False
    assert result.error is None


@patch("agentic_rag.evaluation.runner.check_faithfulness")
@patch("agentic_rag.evaluation.runner._fetch_cited_source_text")
@patch("agentic_rag.evaluation.runner.answer_with_cache")
def test_run_question_marks_a_retrieval_miss_when_no_expected_source_is_cited(
    mock_answer, mock_fetch_text, mock_faithfulness, tmp_path
):
    mock_answer.return_value = AnswerResult(
        text="Chelsea won [1].",
        citations=[Citation(number=1, relative_path="employee/wrong.md", chunk_index=0, access_tier=TIER_EMPLOYEE)],
    )
    mock_fetch_text.return_value = "some other text"
    mock_faithfulness.return_value = FaithfulnessCheckResult(is_faithful=True, raw_judge_response="CLEAN")
    settings = _settings(tmp_path)

    result = _run_question(_question(), settings=settings, client=object(), embedding_cache=object())

    assert result.retrieval_hit is False


@patch("agentic_rag.evaluation.runner.check_faithfulness")
@patch("agentic_rag.evaluation.runner.answer_with_cache")
def test_run_question_normalizes_windows_path_separators_before_comparing(
    mock_answer, mock_faithfulness, tmp_path
):
    # relative_path values from the real ingestion pipeline use os.sep
    # (backslash on Windows); expected_source_paths in the JSON fixture
    # are written with forward slashes for portability - comparing them
    # raw would spuriously report a miss on Windows for a correct hit.
    mock_answer.return_value = AnswerResult(
        text="Arsenal won [1].",
        citations=[
            Citation(number=1, relative_path="employee\\derby.md", chunk_index=0, access_tier=TIER_EMPLOYEE)
        ],
    )
    mock_faithfulness.return_value = FaithfulnessCheckResult(is_faithful=True, raw_judge_response="CLEAN")
    settings = _settings(tmp_path)

    with patch("agentic_rag.evaluation.runner._fetch_cited_source_text", return_value="text"):
        result = _run_question(
            _question(expected_source_paths=["employee/derby.md"]),
            settings=settings,
            client=object(),
            embedding_cache=object(),
        )

    assert result.retrieval_hit is True


def test_run_question_does_not_judge_faithfulness_for_no_expected_source_paths_case(tmp_path):
    # A question that isn't expected to be answerable at all, and the
    # system correctly falls back - nothing was generated, so there's
    # nothing to judge.
    with patch("agentic_rag.evaluation.runner.answer_with_cache") as mock_answer, patch(
        "agentic_rag.evaluation.runner.check_faithfulness"
    ) as mock_faithfulness:
        mock_answer.return_value = AnswerResult(text=CANNOT_ANSWER_MESSAGE, citations=[])
        settings = _settings(tmp_path)

        result = _run_question(
            _question(expected_answerable=False, expected_source_paths=[]),
            settings=settings,
            client=object(),
            embedding_cache=object(),
        )

    mock_faithfulness.assert_not_called()
    assert result.answered is False
    assert result.faithfulness is None


def test_run_question_recognizes_the_fallback_even_with_surrounding_whitespace(tmp_path):
    # Live-verified real bug: mistral sometimes wraps the canonical
    # fallback in a leading space (" I do not know..."), which an exact
    # `!=` comparison against CANNOT_ANSWER_MESSAGE fails to recognize as
    # the fallback at all - miscounting a correct "I don't know" as a
    # fabricated answer. answer_with_cache() itself already uses substring
    # containment (`CANNOT_ANSWER_MESSAGE not in answer.text`), not exact
    # equality, for exactly this reason - this must match that convention.
    with patch("agentic_rag.evaluation.runner.answer_with_cache") as mock_answer, patch(
        "agentic_rag.evaluation.runner.check_faithfulness"
    ) as mock_faithfulness:
        mock_answer.return_value = AnswerResult(
            text=f" {CANNOT_ANSWER_MESSAGE}", citations=[]
        )
        settings = _settings(tmp_path)

        result = _run_question(
            _question(expected_answerable=False, expected_source_paths=[]),
            settings=settings,
            client=object(),
            embedding_cache=object(),
        )

    mock_faithfulness.assert_not_called()
    assert result.answered is False
    assert result.hallucinated is False
    assert result.retrieval_hit is None


def test_run_question_marks_hallucinated_when_answered_but_should_have_been_unanswerable(tmp_path):
    with patch("agentic_rag.evaluation.runner.answer_with_cache") as mock_answer, patch(
        "agentic_rag.evaluation.runner.check_faithfulness"
    ) as mock_faithfulness:
        mock_answer.return_value = AnswerResult(text="A fabricated answer.", citations=[])
        settings = _settings(tmp_path)

        result = _run_question(
            _question(expected_answerable=False, expected_source_paths=[]),
            settings=settings,
            client=object(),
            embedding_cache=object(),
        )

    mock_faithfulness.assert_not_called()
    assert result.hallucinated is True


@patch("agentic_rag.evaluation.runner.check_faithfulness")
@patch("agentic_rag.evaluation.runner._fetch_cited_source_text")
@patch("agentic_rag.evaluation.runner.answer_with_cache")
def test_run_question_marks_hallucinated_when_the_answer_is_judged_unfaithful(
    mock_answer, mock_fetch_text, mock_faithfulness, tmp_path
):
    mock_answer.return_value = AnswerResult(
        text="Arsenal won 5-0 [1].",
        citations=[Citation(number=1, relative_path="employee/derby.md", chunk_index=0, access_tier=TIER_EMPLOYEE)],
    )
    mock_fetch_text.return_value = "Arsenal beat Chelsea 3-0."
    mock_faithfulness.return_value = FaithfulnessCheckResult(
        is_faithful=False, raw_judge_response="UNSUPPORTED"
    )
    settings = _settings(tmp_path)

    result = _run_question(_question(), settings=settings, client=object(), embedding_cache=object())

    assert result.hallucinated is True
    assert result.faithfulness.is_faithful is False


@patch("agentic_rag.evaluation.runner.SemanticCache")
@patch("agentic_rag.evaluation.runner.answer_with_cache")
def test_run_question_uses_a_fresh_semantic_cache_per_question(
    mock_answer, mock_cache_cls, tmp_path
):
    # A cache shared across questions in the same run risks a later
    # question being silently served an earlier, unrelated question's
    # cached answer if their embeddings happen to be similar enough -
    # each question must be measured independently.
    mock_answer.return_value = AnswerResult(text=CANNOT_ANSWER_MESSAGE, citations=[])
    settings = _settings(tmp_path)

    _run_question(
        _question(expected_answerable=False, expected_source_paths=[]),
        settings=settings,
        client=object(),
        embedding_cache=object(),
    )
    _run_question(
        _question(expected_answerable=False, expected_source_paths=[]),
        settings=settings,
        client=object(),
        embedding_cache=object(),
    )

    assert mock_cache_cls.call_count == 2


def test_run_question_isolates_a_pipeline_failure_into_the_error_field(tmp_path):
    # A judge-model timeout/OOM (this session's hardware has a documented
    # history of exactly this) must not raise out of _run_question and
    # take the whole eval run's already-computed results down with it.
    with patch(
        "agentic_rag.evaluation.runner.answer_with_cache",
        side_effect=RuntimeError("Ollama timed out"),
    ), patch("agentic_rag.evaluation.runner.time.monotonic", side_effect=[10.0, 34.0]):
        settings = _settings(tmp_path)

        result = _run_question(
            _question(), settings=settings, client=object(), embedding_cache=object()
        )

    assert result.error == "RuntimeError: Ollama timed out"
    assert result.answered is False
    assert result.hallucinated is False
    assert result.retrieval_hit is None
    assert result.faithfulness is None
    assert result.duration_seconds == 24.0


@patch("agentic_rag.evaluation.runner.check_faithfulness")
@patch("agentic_rag.evaluation.runner._fetch_cited_source_text")
@patch("agentic_rag.evaluation.runner.answer_with_cache")
def test_run_question_records_the_duration_of_a_successful_run(
    mock_answer, mock_fetch_text, mock_faithfulness, tmp_path
):
    # duration_seconds must come from a real time.monotonic() delta, not a
    # hardcoded/derivable constant - mocking the clock (rather than relying
    # on real wall-clock timing, which this repo's CLAUDE.md forbids in
    # tests) lets this assert an exact value instead of a tautological
    # ">= 0.0" that could never catch e.g. duration_seconds being wired to
    # the wrong pair of timestamps.
    mock_answer.return_value = AnswerResult(
        text="Arsenal won [1].",
        citations=[Citation(number=1, relative_path="employee/derby.md", chunk_index=0, access_tier=TIER_EMPLOYEE)],
    )
    mock_fetch_text.return_value = "Arsenal beat Chelsea 3-0."
    mock_faithfulness.return_value = FaithfulnessCheckResult(is_faithful=True, raw_judge_response="CLEAN")
    settings = _settings(tmp_path)

    with patch("agentic_rag.evaluation.runner.time.monotonic", side_effect=[100.0, 105.25]):
        result = _run_question(
            _question(), settings=settings, client=object(), embedding_cache=object()
        )

    assert result.duration_seconds == 5.25


# --- run_evaluation ------------------------------------------------------------


@patch("agentic_rag.evaluation.runner._run_question")
@patch("agentic_rag.evaluation.runner.run_sync_cycle")
@patch("agentic_rag.evaluation.runner.load_questions")
def test_run_evaluation_indexes_the_eval_corpus_with_the_eval_settings(
    mock_load_questions, mock_sync_cycle, mock_run_question, tmp_path
):
    mock_load_questions.return_value = [_question()]
    mock_sync_cycle.return_value = (_empty_sync_result(), {})
    mock_run_question.return_value = _dummy_result()
    settings = _settings(tmp_path)

    run_evaluation(settings=settings)

    sync_kwargs = mock_sync_cycle.call_args.kwargs
    eval_settings = sync_kwargs["settings"]
    assert eval_settings.watched_folder_path == settings.evaluation_corpus_path
    assert eval_settings.qdrant_collection_name == settings.evaluation_qdrant_collection_name


@patch("agentic_rag.evaluation.runner._run_question")
@patch("agentic_rag.evaluation.runner.run_sync_cycle")
@patch("agentic_rag.evaluation.runner.load_questions")
def test_run_evaluation_runs_every_loaded_question_and_builds_a_report(
    mock_load_questions, mock_sync_cycle, mock_run_question, tmp_path
):
    mock_load_questions.return_value = [_question(id="q1"), _question(id="q2")]
    mock_sync_cycle.return_value = (_empty_sync_result(), {})
    mock_run_question.side_effect = [
        _dummy_result(question_id="q1"),
        _dummy_result(question_id="q2"),
    ]
    settings = _settings(tmp_path)

    report = run_evaluation(settings=settings)

    assert [r.question_id for r in report.results] == ["q1", "q2"]


@patch("agentic_rag.evaluation.runner._run_question")
@patch("agentic_rag.evaluation.runner.run_sync_cycle")
@patch("agentic_rag.evaluation.runner.load_questions")
def test_run_evaluation_recreates_the_eval_collection_before_indexing(
    mock_load_questions, mock_sync_cycle, mock_run_question, tmp_path
):
    # A stale collection left over from a previous run (a renamed/removed
    # corpus file) must not leak into this run's measurements - each run
    # starts from a guaranteed-empty collection, not whatever the last
    # run happened to leave behind.
    mock_load_questions.return_value = [_question()]
    mock_sync_cycle.return_value = (_empty_sync_result(), {})
    mock_run_question.return_value = _dummy_result()
    settings = _settings(tmp_path)
    client = get_client(settings.evaluation_qdrant_storage_path)
    ensure_collection(client, settings.evaluation_qdrant_collection_name, vector_size=999)
    client.close()

    run_evaluation(settings=settings)

    fresh_client = get_client(settings.evaluation_qdrant_storage_path)
    try:
        info = fresh_client.get_collection(settings.evaluation_qdrant_collection_name)
        assert info.config.params.vectors[DENSE_VECTOR_NAME].size == settings.embedding_dimensions
    finally:
        fresh_client.close()


@patch("agentic_rag.evaluation.runner._run_question")
@patch("agentic_rag.evaluation.runner.run_sync_cycle")
@patch("agentic_rag.evaluation.runner.load_questions")
def test_run_evaluation_raises_loudly_when_corpus_indexing_has_failures(
    mock_load_questions, mock_sync_cycle, mock_run_question, tmp_path
):
    mock_load_questions.return_value = [_question()]
    mock_sync_cycle.return_value = (
        SyncCycleResult(
            indexed=[],
            deleted=[],
            ingestion_failures=[IngestionFailure(relative_path="bad.txt", reason="boom")],
            indexing_failures=[],
            deletion_failures=[],
        ),
        {},
    )
    settings = _settings(tmp_path)

    with pytest.raises(EvaluationIndexingError, match="bad.txt|ingestion"):
        run_evaluation(settings=settings)

    mock_run_question.assert_not_called()


@patch("agentic_rag.evaluation.runner._run_question")
@patch("agentic_rag.evaluation.runner.run_sync_cycle")
@patch("agentic_rag.evaluation.runner.load_questions")
def test_run_evaluation_closes_the_qdrant_client_on_success(
    mock_load_questions, mock_sync_cycle, mock_run_question, tmp_path
):
    mock_load_questions.return_value = [_question()]
    mock_sync_cycle.return_value = (_empty_sync_result(), {})
    mock_run_question.return_value = _dummy_result()
    settings = _settings(tmp_path)
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = False

    with patch("agentic_rag.evaluation.runner.get_client", return_value=fake_client):
        run_evaluation(settings=settings)

    fake_client.close.assert_called_once()


@patch("agentic_rag.evaluation.runner._run_question")
@patch("agentic_rag.evaluation.runner.run_sync_cycle")
@patch("agentic_rag.evaluation.runner.load_questions")
def test_run_evaluation_closes_the_qdrant_client_even_when_indexing_fails(
    mock_load_questions, mock_sync_cycle, mock_run_question, tmp_path
):
    mock_load_questions.return_value = [_question()]
    mock_sync_cycle.return_value = (
        SyncCycleResult(
            indexed=[],
            deleted=[],
            ingestion_failures=[IngestionFailure(relative_path="bad.txt", reason="boom")],
            indexing_failures=[],
            deletion_failures=[],
        ),
        {},
    )
    settings = _settings(tmp_path)
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = False

    with patch("agentic_rag.evaluation.runner.get_client", return_value=fake_client):
        with pytest.raises(EvaluationIndexingError):
            run_evaluation(settings=settings)

    fake_client.close.assert_called_once()


# --- _report_path_for ---------------------------------------------------------


def test_report_path_for_is_deterministic_given_a_fixed_time(tmp_path):
    now = datetime(2026, 8, 15, 10, 30, 45)

    path = _report_path_for(tmp_path, now=now)

    assert path == tmp_path / "eval-20260815T103045.json"


# --- main -----------------------------------------------------------------------


def _dummy_report() -> EvaluationReport:
    return EvaluationReport(
        results=[],
        retrieval_precision=1.0,
        faithfulness_rate=0.75,
        hallucination_rate=0.167,
        errored_count=0,
        average_duration_seconds=76.7,
    )


@patch("agentic_rag.evaluation.runner.log_evaluation_run")
@patch("agentic_rag.evaluation.runner.configure_eval_logging")
@patch("agentic_rag.evaluation.runner.run_evaluation")
@patch("agentic_rag.evaluation.runner.Settings")
def test_main_configures_logging_before_running_the_evaluation(
    mock_settings_cls, mock_run_evaluation, mock_configure, mock_log_run, tmp_path
):
    mock_settings_cls.return_value = _settings(tmp_path)
    mock_run_evaluation.return_value = _dummy_report()

    main()

    mock_configure.assert_called_once()


@patch("agentic_rag.evaluation.runner.time.monotonic", side_effect=[100.0, 145.5])
@patch("agentic_rag.evaluation.runner.log_evaluation_run")
@patch("agentic_rag.evaluation.runner.configure_eval_logging")
@patch("agentic_rag.evaluation.runner.run_evaluation")
@patch("agentic_rag.evaluation.runner.Settings")
def test_main_logs_the_run_with_the_reports_metrics_and_report_path(
    mock_settings_cls, mock_run_evaluation, mock_configure, mock_log_run, mock_monotonic, tmp_path
):
    mock_settings_cls.return_value = _settings(tmp_path)
    mock_run_evaluation.return_value = _dummy_report()

    main()

    mock_log_run.assert_called_once()
    kwargs = mock_log_run.call_args.kwargs
    assert kwargs["retrieval_precision"] == 1.0
    assert kwargs["faithfulness_rate"] == 0.75
    assert kwargs["hallucination_rate"] == 0.167
    assert kwargs["errored_count"] == 0
    assert kwargs["average_duration_seconds"] == 76.7
    assert kwargs["report_path"].endswith(".json")
    assert kwargs["run_duration_seconds"] == 45.5
