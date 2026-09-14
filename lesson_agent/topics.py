"""Per-topic grounding material.

Two things live here that make the evaluator's job tractable instead of vague
vibes-based judging:

1. `grounding_facts` -- a short list of facts we assert are true. The generator
   is told to only make factual claims consistent with these (plus common,
   safe background knowledge), and the evaluator checks generated claims
   against this list. This is what "accurate & grounded" cashes out to
   concretely, rather than just asking a model "is this accurate?" and hoping.
2. `required_points` -- the key points a beginner lesson on this topic MUST
   cover. This is what "covers the key points" checks against.

Add a new dict entry here to support a new topic; the rest of the pipeline is
topic-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeCard:
    topic: str
    grounding_facts: tuple[str, ...]
    required_points: tuple[str, ...]
    jargon_glossary: dict[str, str]  # term -> beginner-friendly one-line meaning


KNOWLEDGE_CARDS: dict[str, KnowledgeCard] = {
    "rag": KnowledgeCard(
        topic="Introduction to RAG (Retrieval-Augmented Generation)",
        grounding_facts=(
            "RAG stands for Retrieval-Augmented Generation.",
            "RAG combines a retrieval step (fetching relevant documents/passages from an external "
            "knowledge source) with a generation step (an LLM writing an answer using those passages).",
            "The core motivation for RAG is that LLMs have a fixed training cutoff and can hallucinate; "
            "RAG lets them answer using up-to-date or private/domain-specific information without retraining.",
            "A typical RAG pipeline: the source documents are split into chunks, each chunk is converted "
            "into a vector embedding, and embeddings are stored in a vector database.",
            "At query time, the user's question is also embedded, and the vector database returns the "
            "most similar (semantically closest) chunks -- this is the 'retrieval' step.",
            "The retrieved chunks are inserted into the LLM's prompt as context, and the LLM generates an "
            "answer grounded in that context -- this is the 'augmented generation' step.",
            "RAG reduces (but does not eliminate) hallucination, and can cite/attribute its sources because "
            "the retrieved passages are known.",
            "RAG is different from fine-tuning: fine-tuning changes the model's weights; RAG changes what "
            "information the model sees at inference time, without retraining it.",
            "Common building blocks of a RAG system include: a document loader/chunker, an embedding model, "
            "a vector database (e.g. FAISS, Pinecone, Chroma), and an LLM.",
        ),
        required_points=(
            "what RAG stands for and the two-step idea (retrieve, then generate)",
            "why RAG exists / the problem it solves (stale knowledge, hallucination, private data)",
            "how retrieval works at a high level (chunking, embeddings, vector similarity search)",
            "how the retrieved information is used by the generator (inserted as context into the prompt)",
            "at least one concrete example or analogy that makes the retrieve+generate flow tangible",
            "how RAG differs from just fine-tuning an LLM",
        ),
        jargon_glossary={
            "embedding": "a list of numbers that represents the meaning of a piece of text, so a computer can "
            "compare how similar two pieces of text are",
            "vector database": "a database built to store embeddings and quickly find the ones most similar "
            "to a new piece of text",
            "chunking": "splitting a long document into smaller pieces so each piece can be searched and "
            "retrieved on its own",
            "retrieval": "the step where the system searches for and fetches the most relevant pieces of "
            "text for a given question",
            "generation": "the step where the language model writes an answer in natural language",
            "fine-tuning": "further training an existing model on new examples so its weights (internal "
            "settings) change",
            "hallucination": "when a language model confidently states something false or made up",
            "semantic similarity": "how close two pieces of text are in *meaning*, not just in exact words",
            "context window": "the amount of text a language model can 'see' at once when answering",
            "cosine similarity": "a common way to measure how similar two embeddings are by comparing their "
            "direction, not their length",
            "LLM": "large language model -- an AI model trained on huge amounts of text to understand and "
            "generate human-like language",
            "inference": "the step of actually using a trained model to produce an output, as opposed to "
            "training it",
            "prompt": "the text instructions and context given to a language model before it generates a "
            "response",
            "token": "a small chunk of text (often part of a word) that a language model reads or writes one "
            "unit at a time",
            "corpus": "a large collection of text documents used as a knowledge source",
            "top-k retrieval": "fetching the k most similar chunks (e.g. the top 5) instead of just one",
        },
    ),
}


def get_knowledge_card(topic: str) -> KnowledgeCard:
    key = topic.strip().lower()
    for alias in ("rag", "retrieval-augmented generation", "retrieval augmented generation"):
        if alias in key:
            return KNOWLEDGE_CARDS["rag"]
    raise KeyError(
        f"No knowledge card registered for topic {topic!r}. "
        f"Add one to lesson_agent/topics.py -- this keeps grounding/evaluation "
        f"honest instead of letting the model invent its own 'key points'."
    )
