"""
Prompt templates for the RAG and Deep Research pipelines.
Centralised here so prompt wording can be iterated without touching node logic.

Future improvements:
  - Version prompt templates (v1, v2 …) and A/B test them against stored
    evaluation traces to measure quality improvements empirically.
  - Support multi-language prompt variants selected at runtime based on the
    user's detected locale.
  - Load templates from YAML/JSON files so non-engineers can edit prompts
    without touching Python source.
"""

QUERY_REWRITE_TEMPLATE = """
Rewrite the user's question into a concise search query for retrieving relevant PDF chunks.
Use the recent chat only to resolve references like "it", "that", or "the previous topic".
Do not answer the question. Return only the rewritten search query.

Recent chat:
{memory_context}

User question:
{question}
"""

RELEVANCE_GRADER_TEMPLATE = """
You are grading whether retrieved PDF chunks are relevant to a user's question.
Return a single numeric score between 0 and 1.

Scoring guide:
0.0 – retrieved text is unrelated.
0.5 – partially related but likely insufficient.
1.0 – directly contains enough information to answer.

Do not explain your score. Return only the number.

Original question:
{question}

Optimized retrieval query:
{rewritten_question}

Retrieved PDF chunks:
{context}
"""

ANSWER_GENERATION_TEMPLATE = """
You are a helpful question-answering assistant.
Answer the user's question using only the provided fused context.
If the answer is not present in the context, say: "I don't know from the uploaded documents."
Include short source citations when you use context facts.

Recent chat:
{memory_context}

Question:
{question}

Fused context:
{final_context}
"""

RESEARCH_PLAN_TEMPLATE = """
Create a concise research plan for answering the user's question.
Break the work into focused search and evidence-gathering steps.
Return the plan as short bullets.

Recent chat:
{memory_context}

Research question:
{question}
"""

VERIFICATION_TEMPLATE = """
You are a verification agent.
Check whether the gathered evidence is consistent, sufficient, and relevant to the research question.
Return JSON only with these keys:
- confidence_score: number between 0 and 1
- consistency_passed: boolean
- verification_report: short explanation of strengths, conflicts, and gaps

Research question:
{question}

Research plan:
{research_plan}

Sources:
{sources}

Gathered evidence:
{final_context}
"""

REPORT_GENERATION_TEMPLATE = """
You are a deep research assistant.
Generate a well-structured research report using only the evidence in the final context.
Do not invent facts. If evidence is missing, state the limitation clearly.

Your report must use exactly these sections:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Sources

Recent chat:
{memory_context}

Research question:
{question}

Research plan:
{research_plan}

Verification confidence:
{confidence_score}

Verification notes:
{verification_report}

Final evidence context:
{final_context}
"""
