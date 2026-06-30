"""
Evaluation service — relevance grading, evidence verification, RAGAS metrics,
SQLite evaluation logging, and A/B testing hooks.

Improvements implemented:
    RAGAS metrics: faithfulness, answer_relevancy, context_precision
     computed as lightweight LLM-based proxies (no external ragas library required).
     Evaluation logging: every grading and verification result persisted to
     SQLite at data/eval_log.db for offline benchmarking.
     A/B testing hooks: runs tagged with an experiment_id so different
     retrieval strategies can be compared from stored traces.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from langchain_core.documents import Document

from app.config.settings import RELEVANCE_THRESHOLD, MIN_RESEARCH_CONFIDENCE


_DB_PATH = Path(__file__).resolve().parents[4] / "data" / "eval_log.db"


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_log (
            id            TEXT PRIMARY KEY,
            ts            REAL,
            event         TEXT,
            question      TEXT,
            score         REAL,
            web_triggered INTEGER,
            experiment_id TEXT,
            metadata      TEXT
        )
    """)
    conn.commit()
    return conn


def _log(event: str, question: str, score: float,
         web_triggered: bool = False,
         experiment_id: str = "default",
         metadata: dict | None = None) -> None:
    """Persist an evaluation event to SQLite (best-effort — never raises)."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO eval_log VALUES (?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                time.time(),
                event,
                question,
                score,
                int(web_triggered),
                experiment_id,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # logging must never break the pipeline


def query_eval_log(experiment_id: str | None = None, limit: int = 100) -> list[dict]:
    """
    Read stored evaluation traces.

    Args:
        experiment_id: Filter to a specific A/B experiment (None = all).
        limit:         Max rows to return.

    Returns:
        List of row dicts ordered newest-first.
    """
    try:
        conn = _get_conn()
        if experiment_id:
            rows = conn.execute(
                "SELECT * FROM eval_log WHERE experiment_id=? ORDER BY ts DESC LIMIT ?",
                (experiment_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM eval_log ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        conn.close()
        cols = ["id", "ts", "event", "question", "score", "web_triggered", "experiment_id", "metadata"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []



def _parse_score(raw: str) -> float:
    match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
    return max(0.0, min(float(match.group(0)), 1.0)) if match else 0.0



def _build_relevance_prompt(question: str, rewritten: str, docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Document {i}: {source}, page {page}]\n{doc.page_content}")
    return f"""
You are grading whether retrieved PDF chunks are relevant to a user's question.
Return a single numeric score between 0 and 1.

Scoring guide:
0.0 – retrieved text is completely unrelated.
0.5 – partially related but likely insufficient.
1.0 – directly contains enough information to answer.

Return only the number.

Original question: {question}
Optimised retrieval query: {rewritten}

Retrieved PDF chunks:
{chr(10).join(blocks)}
"""


def grade_relevance(
    llm,
    question: str,
    rewritten: str,
    docs: list[Document],
    experiment_id: str = "default",
) -> float:
    """Score document relevance (0–1). Returns 0.0 for empty doc list."""
    if not docs:
        _log("grade_relevance", question, 0.0, experiment_id=experiment_id)
        return 0.0
    prompt = _build_relevance_prompt(question, rewritten, docs)
    response = llm.invoke(prompt)
    score = _parse_score(response.content)
    _log("grade_relevance", question, score,
         web_triggered=score < RELEVANCE_THRESHOLD,
         experiment_id=experiment_id)
    return score


def should_web_search(score: float) -> bool:
    """Return True when relevance is strictly below threshold (not equal)."""
    return score < RELEVANCE_THRESHOLD



def _build_verification_prompt(question, plan, context, sources) -> str:
    source_lines = "\n".join(f"* {s}" for s in sources)
    return f"""
You are a verification agent.
Check whether the gathered evidence is consistent, sufficient, and relevant.
Return JSON only with these keys:
- confidence_score: number 0–1
- consistency_passed: boolean
- verification_report: short explanation of strengths, conflicts, and gaps

Research question: {question}
Research plan: {plan}
Sources:
{source_lines}

Gathered evidence:
{context}
"""


def _parse_verification(raw: str) -> dict:
    data = {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            data = parsed
        else:
            # json.loads returned a scalar (float/int/str) — try regex fallback
            raise ValueError("not a dict")
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                pass

    score = max(0.0, min(float(data.get("confidence_score", 0) or 0), 1.0))
    return {
        "confidence_score": score,
        "consistency_passed": bool(data.get("consistency_passed", False)),
        "verification_report": str(data.get("verification_report", raw)).strip(),
    }


def verify_evidence(
    llm,
    question: str,
    plan: str,
    context: str,
    sources: list[str],
    experiment_id: str = "default",
) -> dict:
    """Verify evidence quality. Returns confidence_score, consistency_passed, verification_report."""
    prompt = _build_verification_prompt(question, plan, context, sources)
    response = llm.invoke(prompt)
    result = _parse_verification(response.content)
    _log("verify_evidence", question, result["confidence_score"],
         experiment_id=experiment_id,
         metadata={"consistency_passed": result["consistency_passed"]})
    return result



def compute_ragas_metrics(
    llm,
    question: str,
    answer: str,
    context: str,
    experiment_id: str = "default",
) -> dict[str, float]:
    """
    Compute three RAGAS-style quality metrics via LLM prompts.

    Returns:
        {
          "faithfulness":       float 0-1  # Is the answer grounded in context?
          "answer_relevancy":   float 0-1  # Does the answer address the question?
          "context_precision":  float 0-1  # Is the context focused (low noise)?
        }
    """
    prompts = {
        "faithfulness": f"""
Rate how faithfully the answer is grounded in the provided context.
Score 1.0 if every claim is supported; 0.0 if the answer contradicts or ignores context.
Return only a number between 0 and 1.

Question: {question}
Context: {context[:1500]}
Answer: {answer[:800]}
""",
        "answer_relevancy": f"""
Rate how directly and completely the answer addresses the question.
Score 1.0 if fully answered; 0.0 if the answer is off-topic.
Return only a number between 0 and 1.

Question: {question}
Answer: {answer[:800]}
""",
        "context_precision": f"""
Rate how focused and relevant the context is for answering the question.
Score 1.0 if all context is useful; 0.0 if most context is noise.
Return only a number between 0 and 1.

Question: {question}
Context: {context[:1500]}
""",
    }

    metrics: dict[str, float] = {}
    for metric_name, prompt in prompts.items():
        try:
            resp = llm.invoke(prompt)
            score = _parse_score(resp.content)
        except Exception:
            score = 0.0
        metrics[metric_name] = score
        _log(f"ragas_{metric_name}", question, score, experiment_id=experiment_id)

    return metrics
