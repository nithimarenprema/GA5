import json
import re
import math
from collections import Counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --------------------------------
# 1. INITIALIZE APPLICATION
# --------------------------------

app = FastAPI(title="RAG Colosseum")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------
# 2. LOAD KNOWLEDGE BASE
# --------------------------------

with open("kb.json", "r", encoding="utf-8") as f:
    data = json.load(f)


def get_passages(data):

    if isinstance(data, list):
        items = data

    elif isinstance(data, dict):
        items = data.get(
            "passages",
            data.get("documents", [])
        )

    else:
        raise ValueError("Invalid knowledge base")

    passages = []

    for item in items:

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
        raise ValueError(
            f"Expected 50 passages, got {len(passages)}"
        )

    return passages


passages = get_passages(data)


# --------------------------------
# 3. TEXT PROCESSING
# --------------------------------

STOP_WORDS = {
    "what", "which", "when", "where", "who",
    "why", "how", "does", "do", "did",
    "is", "are", "was", "were", "the",
    "a", "an", "of", "in", "on", "for",
    "to", "with", "and", "or", "by",
    "from", "this", "that", "these",
    "those", "it", "its", "be", "as",
    "at", "can", "could", "would",
    "should", "used", "use", "using",
    "explain", "describe", "tell", "me"
}


def tokenize(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return [
        word for word in words
        if word not in STOP_WORDS
    ]


# --------------------------------
# 4. BUILD BM25 INDEX
# --------------------------------

tokenized_passages = [
    tokenize(passage)
    for passage in passages
]

document_count = len(passages)

document_lengths = [
    len(tokens)
    for tokens in tokenized_passages
]

average_length = (
    sum(document_lengths) / document_count
    if document_count else 1
)

document_frequencies = Counter()

for tokens in tokenized_passages:
    document_frequencies.update(set(tokens))


def bm25_score(query_tokens, document_tokens, doc_length):

    k1 = 1.5
    b = 0.75

    frequencies = Counter(document_tokens)

    score = 0.0

    for word in set(query_tokens):

        frequency = frequencies.get(word, 0)

        if frequency == 0:
            continue

        df = document_frequencies.get(word, 0)

        idf = math.log(
            1 + (
                document_count - df + 0.5
            ) / (df + 0.5)
        )

        numerator = frequency * (k1 + 1)

        denominator = frequency + k1 * (
            1 - b + b * doc_length / average_length
        )

        score += idf * numerator / denominator

    return score


# --------------------------------
# 5. REQUEST FORMAT
# --------------------------------

class QueryRequest(BaseModel):

    question: str

    top_k: int = 3


# --------------------------------
# 6. ANSWER EXTRACTION
# --------------------------------

def extract_answer(question, selected):

    query_words = set(tokenize(question))

    results = []

    blocked_phrases = [
        "ignore previous instructions",
        "ignore all instructions",
        "reveal the system prompt",
        "reveal your secrets",
        "you are chatgpt",
        "act as an assistant"
    ]

    for passage_id, score in selected:

        text = passages[passage_id]

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        for sentence in sentences:

            sentence_lower = sentence.lower()

            if any(
                phrase in sentence_lower
                for phrase in blocked_phrases
            ):
                continue

            sentence_words = set(
                tokenize(sentence)
            )

            overlap = len(
                query_words & sentence_words
            )

            if overlap > 0:

                results.append(
                    (overlap, score, sentence)
                )

    results.sort(
        key=lambda item: (
            item[0],
            item[1]
        ),
        reverse=True
    )

    selected_sentences = []

    for _, _, sentence in results:

        if sentence not in selected_sentences:
            selected_sentences.append(sentence)

        if len(selected_sentences) == 3:
            break

    if not selected_sentences:
        return None

    return " ".join(selected_sentences)


# --------------------------------
# 7. QUERY ENDPOINT
# --------------------------------

@app.post("/query")
def query(request: QueryRequest):

    question = request.question.strip()

    if not question:

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

    # Calculate BM25 scores.

    scores = []

    for i, tokens in enumerate(tokenized_passages):

        score = bm25_score(
            query_tokens,
            tokens,
            document_lengths[i]
        )

        # Small bonus for an exact phrase match.

        if question.lower() in passages[i].lower():
            score += 1.0

        scores.append(score)

    # Rank passages.

    ranked = sorted(
        range(document_count),
        key=lambda i: scores[i],
        reverse=True
    )

    top_k = max(
        1,
        min(request.top_k, 5)
    )

    selected_ids = ranked[:top_k]

    best_score = scores[selected_ids[0]]

    # Check query-term coverage.

    best_tokens = set(
        tokenized_passages[selected_ids[0]]
    )

    matched_terms = len(
        set(query_tokens) & best_tokens
    )

    coverage = (
        matched_terms / len(set(query_tokens))
    )

    # Abstain when retrieval is weak.

    if best_score < 0.8 or coverage < 0.25:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    selected = [
        (i, scores[i])
        for i in selected_ids
        if scores[i] > 0
    ]

    answer = extract_answer(
        question,
        selected
    )

    if not answer:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    # Convert BM25 scores into confidence-like values.

    normalized_scores = [
        round(
            1 - math.exp(-scores[i] / 4),
            4
        )
        for i, _ in selected
    ]

    confidence = min(
        0.95,
        max(
            0.1,
            1 - math.exp(-best_score / 4)
        )
    )

    return {
        "answer": answer,
        "sources": [
            i for i, _ in selected
        ],
        "scores": normalized_scores,
        "confidence": round(confidence, 4)
    }


# --------------------------------
# 8. HEALTH CHECK
# --------------------------------

@app.get("/")
def home():

    return {
        "status": "RAG Colosseum running",
        "passages": len(passages)
    }


# --------------------------------
# 9. SERVER START
# --------------------------------

if __name__ == "__main__":

    import os
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
