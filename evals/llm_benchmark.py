"""
LLM Benchmarking: accuracy, BLEU, ROUGE, BERTScore, and latency metrics.
"""
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import anthropic


@dataclass
class BenchmarkResult:
    prompt: str
    expected: str
    generated: str
    bleu: float
    rouge_l: float
    bert_f1: float
    latency_ms: float
    model: str
    metadata: dict = field(default_factory=dict)


class LLMBenchmark:
    def __init__(self, model: str = "claude-sonnet-4-20250514", system_prompt: Optional[str] = None):
        self.model = model
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic()
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def _call_model(self, prompt: str) -> tuple[str, float]:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "max_tokens": 1024, "messages": messages}
        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        start = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        return response.content[0].text, latency_ms

    def _bleu(self, reference: str, hypothesis: str) -> float:
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        smoother = SmoothingFunction().method4
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)

    def _rouge_l(self, reference: str, hypothesis: str) -> float:
        scores = self.rouge.score(reference, hypothesis)
        return scores["rougeL"].fmeasure

    def evaluate_sample(self, prompt: str, expected: str) -> BenchmarkResult:
        generated, latency_ms = self._call_model(prompt)

        _, _, bert_f1 = bert_score([generated], [expected], lang="en", verbose=False)

        return BenchmarkResult(
            prompt=prompt,
            expected=expected,
            generated=generated,
            bleu=self._bleu(expected, generated),
            rouge_l=self._rouge_l(expected, generated),
            bert_f1=bert_f1.mean().item(),
            latency_ms=latency_ms,
            model=self.model,
        )

    def run(self, dataset: list[dict]) -> dict:
        """
        dataset: list of {"prompt": str, "expected": str}
        """
        results = []
        for item in dataset:
            result = self.evaluate_sample(item["prompt"], item["expected"])
            results.append(asdict(result))
            print(f"  BLEU={result.bleu:.3f} | ROUGE-L={result.rouge_l:.3f} | BERTScore={result.bert_f1:.3f} | {result.latency_ms:.0f}ms")

        bleus = [r["bleu"] for r in results]
        rouges = [r["rouge_l"] for r in results]
        berts = [r["bert_f1"] for r in results]
        latencies = [r["latency_ms"] for r in results]

        summary = {
            "model": self.model,
            "n_samples": len(results),
            "bleu": {"mean": np.mean(bleus), "std": np.std(bleus)},
            "rouge_l": {"mean": np.mean(rouges), "std": np.std(rouges)},
            "bert_f1": {"mean": np.mean(berts), "std": np.std(berts)},
            "latency_ms": {
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
            },
            "samples": results,
        }
        return summary


if __name__ == "__main__":
    dataset = [
        {"prompt": "What is the capital of France?", "expected": "The capital of France is Paris."},
        {"prompt": "Explain gradient descent in one sentence.", "expected": "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that minimizes the loss function."},
    ]
    bench = LLMBenchmark()
    summary = bench.run(dataset)
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))
