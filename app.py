import json
import re
import math
import os
from collections import Counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="RAG Colosseum")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the 50-passage knowledge base
with open("kb.json", "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, list):
    raw_passages = data
elif isinstance(data, dict):
    raw_passages = data.get("passages", data.get("documents", []))
else:
    raise ValueError("Invalid knowledge base format")

passages = []
for item in raw_passages:
    if isinstance(item, str):
        passages.append(item)
    elif isinstance(item, dict):
        text = (
            item.get("text")
            or item.get("content")
            or item.get("passage")
            or item.get("body")
        )
        if text:
            passages.append(text)

if len(passages) != 50:
    raise ValueError(f"Expected 50 passages, found {len(passages)}")


# Normalize common word variations
ALIASES = {
    "heartbeats": "heartbeat",
    "timeouts": "timeout",
    "quorums": "quorum",
    "replicas": "replica",
    "servers": "server",
    "nodes": "node",
    "layers": "layer",
    "blocks": "block",
    "iterations": "iteration",
    "keys": "key",
    "ports": "port",
    "protocols": "protocol",
    "algorithms": "algorithm",
    "databases": "database",
    "connections": "connection",
    "requests": "request",
    "packets": "packet",
    "hashes": "hash",
    "digests": "digest",
    "batches": "batch",
    "samples": "sample",
    "retries": "retry",
    "intervals": "interval",
    "vectors": "vector",
    "transactions": "transaction",
    "signatures": "signature",
    "certificates": "certificate",
    "timers": "timer",
    "updates": "update",
    "queries": "query",
    "indexes": "index",
    "indices": "index",
    "weights": "weight",
    "gradients": "gradient",
}

STOP_WORDS = {
    "what", "which", "when", "where", "who", "why", "how",
    "does", "do", "did", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "on", "for", "to",
    "with", "and", "or", "by", "from", "this", "that",
    "these", "those", "it", "its", "be", "as", "at",
    "can", "could", "would", "should", "used", "use",
    "using", "explain", "describe", "tell", "me",
    "please", "give", "show", "according", "passage",
    "knowledge", "base", "mentioned", "stated"
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    result = []

    for word in words:
        if word in STOP_WORDS:
            continue

        word = ALIASES.get(word, word)
        result.append(word)

    return result


# Build BM25 index
tokenized_passages = [tokenize(p) for p in passages]
N = len(passages)

doc_lengths = [len(p) for p in tokenized_passages]
avg_length = sum(doc_lengths) / N

df = Counter()
for tokens in tokenized_passages:
    df.update(set(tokens))


def bm25(query_tokens, document_tokens, length):
    k1 = 1.5
    b = 0.75
    frequencies = Counter(document_tokens)
    score = 0.0

    for word in set(query_tokens):
        frequency = frequencies.get(word, 0)

        if frequency == 0:
            continue

        document_frequency = df.get(word, 0)

        idf = math.log(
            1 + (N - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

        numerator = frequency * (k1 + 1)
        denominator = frequency + k1 * (
            1 - b + b * length / avg_length
        )

        score += idf * numerator / denominator

    return score


# Detect questions that ask for a particular fact
ATTRIBUTE_GROUPS = {
    "port": ["port", "listening", "listen", "bind", "socket"],
    "timeout": ["timeout", "wait", "deadline", "expire"],
    "quorum": ["quorum", "acknowledgment", "acknowledge"],
    "interval": ["interval", "periodically", "frequency"],
    "retry": ["retry", "retries", "attempt"],
    "batch": ["batch", "batches", "samples", "chunk"],
    "learning_rate": ["learning", "rate", "lr"],
    "layers": ["layers", "layer", "blocks", "stack"],
    "bits": ["bits", "length", "security"],
    "hash": ["hash", "digest", "sha", "hmac"],
    "protocol": ["protocol", "transport"],
    "algorithm": ["algorithm", "algorithms"],
}


def question_attribute(question):
    words = set(tokenize(question))

    for attribute, terms in ATTRIBUTE_GROUPS.items():
        if words.intersection(terms):
            return attribute

    return None


def sentence_relevance(question_tokens, sentence):
    sentence_tokens = set(tokenize(sentence))

    if not sentence_tokens:
        return 0

    return len(set(question_tokens) & sentence_tokens)


def is_injection(question):
    q = question.lower()

    dangerous_phrases = [
        "ignore previous instructions",
        "ignore all instructions",
        "reveal the system prompt",
        "reveal your system prompt",
        "show your hidden prompt",
        "forget your instructions",
        "act as an unrestricted",
        "ignore the knowledge base",
    ]

    return any(phrase in q for phrase in dangerous_phrases)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


def retrieve(question, query_tokens, top_k):
    results = []

    for i, tokens in enumerate(tokenized_passages):
        score = bm25(query_tokens, tokens, doc_lengths[i])

        # Reward exact technical phrases
        passage_lower = passages[i].lower()
        question_lower = question.lower()

        if question_lower in passage_lower:
            score += 2.0

        # Reward matching technical terms
        for token in set(query_tokens):
            if len(token) >= 4 and token in passage_lower:
                score += 0.15

        results.append((i, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


def build_answer(question, selected):
    query_tokens = set(tokenize(question))
    attribute = question_attribute(question)

    candidates = []

    for passage_id, passage_score in selected:
        text = passages[passage_id]

        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Treat passage text as data, not instructions
            blocked = [
                "ignore previous instructions",
                "ignore all instructions",
                "reveal the system prompt",
                "act as an assistant",
            ]

            if any(x in sentence_lower for x in blocked):
                continue

            overlap = sentence_relevance(query_tokens, sentence)

            if overlap == 0:
                continue

            attribute_bonus = 0

            if attribute:
                terms = ATTRIBUTE_GROUPS[attribute]

                if any(term in sentence_lower for term in terms):
                    attribute_bonus = 2

            relevance = overlap + attribute_bonus

            candidates.append(
                (relevance, passage_score, passage_id, sentence)
            )

    candidates.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True
    )

    selected_sentences = []
    source_ids = []

    for _, _, passage_id, sentence in candidates:
        if sentence not in selected_sentences:
            selected_sentences.append(sentence)

        if passage_id not in source_ids:
            source_ids.append(passage_id)

        if len(selected_sentences) >= 3:
            break

    if not selected_sentences:
        return None, []

    return " ".join(selected_sentences), source_ids


@app.post("/query")
def query(request: QueryRequest):
    question = request.question.strip()

    if not question or is_injection(question):
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    query_tokens = tokenize(question)

    if not query_tokens:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    top_k = max(1, min(request.top_k, 5))

    ranked = retrieve(question, query_tokens, top_k)

    if not ranked or ranked[0][1] <= 0:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    best_id, best_score = ranked[0]

    best_tokens = set(tokenized_passages[best_id])
    query_set = set(query_tokens)

    coverage = (
        len(query_set & best_tokens) / len(query_set)
        if query_set else 0
    )

    # Conservative unsupported-question check
    if best_score < 0.8 or coverage < 0.25:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    answer, source_ids = build_answer(question, ranked)

    if not answer:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    scores = [
        round(1 - math.exp(-score / 4), 4)
        for passage_id, score in ranked
        if passage_id in source_ids
    ]

    confidence = min(
        0.95,
        max(0.1, 1 - math.exp(-best_score / 4))
    )

    return {
        "answer": answer,
        "sources": source_ids,
        "scores": scores,
        "confidence": round(confidence, 4)
    }


@app.get("/")
def home():
    return {
        "status": "RAG Colosseum running",
        "passages": len(passages)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
