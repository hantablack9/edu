"""Run the RAG++ Weave evaluation comparison in Codespaces."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cohere
import nltk
import weave

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
PROMPTS = NOTEBOOKS / "prompts"
sys.path.insert(0, str(NOTEBOOKS))

from scripts.response_generator import SimpleResponseGenerator  # noqa: E402
from scripts.response_metrics import (  # noqa: E402
    compute_bleu,
    compute_diff,
    compute_levenshtein,
    compute_rouge,
    llm_response_scorer,
)
from scripts.retrieval_metrics import IR_METRICS, LLM_METRICS as RETRIEVAL_LLM_METRICS  # noqa: E402
from scripts.retriever import TFIDFRetriever  # noqa: E402

DATASET_URI = "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
CHUNKED_DATA_URI = "weave:///rag-course/rag-course/object/chunked_data:Lt6M8qCUICD1JZTlMYzuDLTVtvFYESxvj3tcAIoPrtE"


def required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing {name}; set it as a Codespaces secret, never commit it.")
    return value


def json_result(text: str) -> dict[str, Any]:
    match = re.search(r"\\{.*\\}", text.strip(), flags=re.S)
    if not match:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    return json.loads(match.group(0))


async def cohere_judge(instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
    client = cohere.AsyncClientV2(api_key=required("COHERE_API_KEY"))
    response = await client.chat(
        model=os.getenv("COHERE_JUDGE_MODEL", "command-r-plus"),
        temperature=0.0,
        max_tokens=250,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return json_result(response.message.content[0].text)


def normalize(result: dict[str, Any]) -> float:
    return max(0.0, min(2.0, float(result.get("score", 0)))) / 2.0


@weave.op
async def relevance_judge(output: dict[str, Any], question: str) -> float:
    result = await cohere_judge(
        "Judge answer relevance and completeness for the question. Return JSON only with score 0, 1, or 2 and reason. Use 2 for a direct complete answer, 1 for partial relevance, 0 for irrelevant.",
        {"question": question, "answer": output["answer"]},
    )
    return normalize(result)


@weave.op
async def faithfulness_judge(output: dict[str, Any], question: str) -> float:
    context = "\\n\\n".join(f"[{x.get('source', 'unknown')}] {x.get('text', '')}" for x in output.get("contexts", []))
    result = await cohere_judge(
        "Judge whether every material answer claim is supported by the context. Return JSON only with score 0, 1, or 2 and reason. Use 2 for fully supported, 1 for mixed, 0 for unsupported or contradicted.",
        {"question": question, "context": context, "answer": output["answer"]},
    )
    return normalize(result)


class InspectableRAG(weave.Model):
    retriever: weave.Model
    response_generator: weave.Model
    top_k: int = 5

    @weave.op
    def predict(self, question: str) -> dict[str, Any]:
        contexts = self.retriever.predict(question, self.top_k)
        return {"answer": self.response_generator.predict(question, contexts), "contexts": contexts}


@weave.op
def response_diff(output: dict[str, Any], answer: str) -> float:
    return compute_diff(output["answer"], answer)


@weave.op
def response_levenshtein(output: dict[str, Any], answer: str) -> float:
    return compute_levenshtein(output["answer"], answer)


@weave.op
def response_rouge(output: dict[str, Any], answer: str) -> float:
    return compute_rouge(output["answer"], answer)


@weave.op
def response_bleu(output: dict[str, Any], answer: str) -> float:
    return compute_bleu(output["answer"], answer)


@weave.op
async def correctness_judge(output: dict[str, Any], question: str, answer: str) -> float:
    result = await llm_response_scorer(question=question, answer=answer, output=output["answer"])
    return float(result["score"]) / 2.0


async def main() -> None:
    required("WANDB_API_KEY")
    required("COHERE_API_KEY")
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT", "rag-course-assignment")
    weave_project = f"{entity}/{project}" if entity else project
    weave.init(weave_project)

    dataset = weave.ref(DATASET_URI).get()
    chunks = weave.ref(CHUNKED_DATA_URI).get()
    retriever = TFIDFRetriever()
    retriever.index_data([dict(row) for row in chunks.rows])

    await weave.Evaluation(
        name="retrieval-ir-evaluation",
        dataset=dataset,
        scorers=IR_METRICS,
        preprocess_model_input=lambda row: {"query": row["question"], "k": 5},
    ).evaluate(retriever)
    await weave.Evaluation(
        name="retrieval-llm-judge-evaluation",
        dataset=dataset,
        scorers=RETRIEVAL_LLM_METRICS,
        preprocess_model_input=lambda row: {"query": row["question"], "k": 5},
    ).evaluate(retriever)

    rag = InspectableRAG(
        retriever=retriever,
        response_generator=SimpleResponseGenerator(
            model=os.getenv("COHERE_GENERATION_MODEL", "command-r"),
            prompt=(PROMPTS / "initial_system.txt").read_text(),
        ),
    )
    await weave.Evaluation(
        name="response-evaluation-comparison",
        dataset=dataset,
        scorers=[response_diff, response_levenshtein, response_rouge, response_bleu, correctness_judge, relevance_judge, faithfulness_judge],
        preprocess_model_input=lambda row: {"question": row["question"]},
    ).evaluate(rag)
    print(f"Weave project: https://wandb.ai/{weave_project}/weave")
    print("Copy that URL into assignment_submission.txt before uploading it to the course.")


if __name__ == "__main__":
    asyncio.run(main())
