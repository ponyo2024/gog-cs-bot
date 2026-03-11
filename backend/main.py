import time
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from long_context_service import ask_long_context
from rag_service import ask_rag


app = FastAPI(title="GoG CS Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    model_used: str | None = None
    latency_seconds: float | None = None
    tokens_used: int | None = None
    method: Literal["rag", "long_context"]
    status: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/rag", response_model=QueryResponse)
def rag_query(req: QueryRequest) -> QueryResponse:
    start = time.time()
    result = ask_rag(req.question)
    latency = round(time.time() - start, 2)
    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        model_used=result.get("model_used"),
        latency_seconds=latency,
        tokens_used=result.get("total_tokens"),
        method="rag",
        status=result.get("status", "error"),
    )


@app.post("/api/long-context", response_model=QueryResponse)
def long_context_query(req: QueryRequest) -> QueryResponse:
    start = time.time()
    result = ask_long_context(req.question)
    latency = round(time.time() - start, 2)
    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        model_used=result.get("model_used"),
        latency_seconds=latency,
        tokens_used=result.get("total_tokens"),
        method="long_context",
        status=result.get("status", "error"),
    )
