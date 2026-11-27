import json
import re
import os
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# -------------------------------
# 1. INITIALIZE APPLICATION
# -------------------------------

app = FastAPI(title="RAG Colosseum")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# 2. LOAD KNOWLEDGE BASE
# -------------------------------

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

    for i, item in enumerate(items):

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


# -------------------------------
# 3. LOAD EMBEDDING MODEL
# -------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------
# 4. GENERATE EMBEDDINGS
# -------------------------------

passage_embeddings = model.encode(
    passages,
    normalize_embeddings=True,
    convert_to_numpy=True
)


# -------------------------------
# 5. REQUEST FORMAT
# -------------------------------

class QueryRequest(BaseModel):

    question: str

    top_k: int = 3


# -------------------------------
# 6. EXTRACT FACTUAL SENTENCES
# -------------------------------

def extract_answer(question, selected):

    question_words = set(
        re.findall(r"[a-zA-Z0-9]+", question.lower())
    )

    stop_words = {
        "what", "which", "when", "where",
        "does", "do", "is", "are", "the",
        "a", "an", "of", "in", "for",
        "to", "how", "many", "much",
        "used", "use", "using", "and"
    }

    keywords = question_words - stop_words

    results = []

    for passage_id, score in selected:

        text = passages[passage_id]

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        for sentence in sentences:

            words = set(
                re.findall(
                    r"[a-zA-Z0-9]+",
                    sentence.lower()
                )
            )

            overlap = len(words & keywords)

            if overlap > 0:
                results.append(
                    (overlap, score, sentence)
                )

    results.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True
    )

    if not results:
        return None

    # Select relevant factual sentences.
    selected_sentences = []

    for _, _, sentence in results[:3]:

        if sentence not in selected_sentences:
            selected_sentences.append(sentence)

    return " ".join(selected_sentences)


# -------------------------------
# 7. RAG QUERY ENDPOINT
# -------------------------------

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

    # Convert question into embedding.
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    # Calculate cosine similarity.
    similarities = np.dot(
        passage_embeddings,
        query_embedding
    )

    # Find most relevant passages.
    top_k = max(1, min(request.top_k, 5))

    indices = np.argsort(
        similarities
    )[::-1][:top_k]

    selected = [
        (int(i), float(similarities[i]))
        for i in indices
    ]

    best_score = selected[0][1]

    # Abstain when relevance is too low.
    if best_score < 0.30:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.15
        }

    # Extract answer from retrieved passages.
    answer = extract_answer(
        question,
        selected
    )

    if not answer:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.15
        }

    # Calculate confidence.
    confidence = min(
        0.95,
        max(0.65, best_score)
    )

    return {
        "answer": answer,
        "sources": [i for i, s in selected],
        "scores": [
            round(max(0.0, min(1.0, s)), 4)
            for i, s in selected
        ],
        "confidence": round(confidence, 4)
    }


# -------------------------------
# 8. HEALTH CHECK
# -------------------------------

@app.get("/")
def home():

    return {
        "status": "RAG Colosseum running",
        "passages": len(passages)
    }


# -------------------------------
# 9. START SERVER
# -------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
