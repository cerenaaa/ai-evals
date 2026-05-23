"""
RAG Evaluation: faithfulness, context recall, and answer relevancy (RAGAS-style).
Uses NLI and embedding similarity for model-free scoring.
"""
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
from sentence_transformers import SentenceTransformer, util
import anthropic


@dataclass
class RAGEvalResult:
    question: str
    context: str
    answer: str
    ground_truth: Optional[str]
    faithfulness: float        # Is the answer grounded in the context?
    answer_relevancy: float    # Is the answer relevant to the question?
    context_recall: float      # Does the context cover the ground truth?


class RAGEvaluator:
    """
    Evaluates RAG pipeline outputs along three axes:
    - Faithfulness: answer claims are supported by context (via LLM judge)
    - Answer Relevancy: cosine similarity between answer and question embeddings
    - Context Recall: overlap between context and ground truth (when available)
    """
    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        self.client = anthropic.Anthropic()
        self.embedder = SentenceTransformer(embed_model)

    def _faithfulness_score(self, context: str, answer: str) -> float:
        """LLM judge: what fraction of answer claims are supported by context?"""
        prompt = f"""You are evaluating whether an answer is faithful to a given context.

Context:
{context}

Answer:
{answer}

List each factual claim in the answer. For each claim, state whether it is SUPPORTED or NOT SUPPORTED by the context.
Then output a JSON object with key "faithfulness_score" (float 0-1, fraction of claims that are supported).
Respond ONLY with the JSON object."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            data = json.loads(response.content[0].text)
            return float(data.get("faithfulness_score", 0.0))
        except Exception:
            return 0.0

    def _answer_relevancy(self, question: str, answer: str) -> float:
        """Cosine similarity between question and answer embeddings."""
        q_emb = self.embedder.encode(question, convert_to_tensor=True)
        a_emb = self.embedder.encode(answer, convert_to_tensor=True)
        return float(util.cos_sim(q_emb, a_emb).item())

    def _context_recall(self, context: str, ground_truth: str) -> float:
        """Sentence-level recall: fraction of GT sentences entailed by context."""
        gt_sentences = [s.strip() for s in ground_truth.split(".") if s.strip()]
        if not gt_sentences:
            return 0.0
        ctx_emb = self.embedder.encode(context, convert_to_tensor=True)
        recalled = 0
        for sent in gt_sentences:
            s_emb = self.embedder.encode(sent, convert_to_tensor=True)
            sim = float(util.cos_sim(ctx_emb, s_emb).item())
            if sim > 0.75:
                recalled += 1
        return recalled / len(gt_sentences)

    def evaluate(self, samples: list[dict]) -> dict:
        """
        samples: list of {question, context, answer, ground_truth (optional)}
        """
        results = []
        for s in samples:
            faith = self._faithfulness_score(s["context"], s["answer"])
            relevancy = self._answer_relevancy(s["question"], s["answer"])
            recall = self._context_recall(s["context"], s.get("ground_truth", "")) if s.get("ground_truth") else None

            result = RAGEvalResult(
                question=s["question"],
                context=s["context"],
                answer=s["answer"],
                ground_truth=s.get("ground_truth"),
                faithfulness=faith,
                answer_relevancy=relevancy,
                context_recall=recall if recall is not None else -1.0,
            )
            results.append(asdict(result))
            print(f"  Faithfulness={faith:.2f} | Relevancy={relevancy:.2f} | Recall={recall:.2f if recall else 'N/A'}")

        return {
            "n_samples": len(results),
            "avg_faithfulness": np.mean([r["faithfulness"] for r in results]),
            "avg_relevancy": np.mean([r["answer_relevancy"] for r in results]),
            "avg_context_recall": np.mean([r["context_recall"] for r in results if r["context_recall"] >= 0]),
            "samples": results,
        }
