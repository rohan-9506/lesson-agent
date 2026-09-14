"""Unit tests for the two fully-deterministic rubric checks. No network,
no LLM -- these must pass with zero API key configured."""
from lesson_agent.rubric import check_jargon, check_readability
from lesson_agent.topics import get_knowledge_card

CARD = get_knowledge_card("rag")


def test_jargon_check_fails_on_unexplained_term():
    text = "The ranking step relies on cosine similarity to sort the candidates."
    result = check_jargon(text, CARD.jargon_glossary)
    assert result.passed is False
    assert "cosine similarity" in result.reason


def test_jargon_check_passes_when_term_explained():
    text = (
        "The ranking step relies on cosine similarity, which means comparing how similar two "
        "embeddings are, to sort the candidates."
    )
    result = check_jargon(text, CARD.jargon_glossary)
    assert result.passed is True


def test_jargon_check_ignores_terms_never_used():
    text = "This sentence does not mention any glossary term at all."
    result = check_jargon(text, CARD.jargon_glossary)
    assert result.passed is True


def test_readability_check_fails_on_dense_academic_prose():
    text = (
        "Consequently, the aforementioned methodological paradigm necessitates the orchestration of "
        "multidimensional vector representations, which, notwithstanding their computational overhead, "
        "facilitate an epistemologically robust retrieval mechanism that substantially ameliorates the "
        "hallucinatory propensities intrinsic to autoregressive language modeling architectures."
    )
    result = check_readability(text)
    assert result.passed is False


def test_readability_check_passes_on_simple_prose():
    text = (
        "RAG helps a computer find the right notes first. Then it writes an answer using those notes. "
        "This makes the answer more likely to be true and up to date."
    )
    result = check_readability(text)
    assert result.passed is True
