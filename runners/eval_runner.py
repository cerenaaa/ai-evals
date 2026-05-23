"""
Eval Runner: orchestrates benchmark, RAG, and bias eval pipelines from a YAML config.
"""
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime

from evals.llm_benchmark import LLMBenchmark
from evals.rag_eval import RAGEvaluator
from evals.bias_eval import BiasEvaluator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_evals(config: dict) -> dict:
    results = {"run_at": datetime.utcnow().isoformat(), "evals": {}}

    if "benchmark" in config:
        print("\n── LLM Benchmark ──")
        bench = LLMBenchmark(
            model=config["benchmark"].get("model", "claude-sonnet-4-20250514"),
            system_prompt=config["benchmark"].get("system_prompt"),
        )
        results["evals"]["benchmark"] = bench.run(config["benchmark"]["dataset"])

    if "rag" in config:
        print("\n── RAG Eval ──")
        rag = RAGEvaluator()
        results["evals"]["rag"] = rag.evaluate(config["rag"]["samples"])

    if "bias" in config:
        print("\n── Bias Eval ──")
        bias = BiasEvaluator()
        results["evals"]["bias"] = bias.run(config["bias"]["templates"])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--output", default="results/eval_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_evals(config)

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
