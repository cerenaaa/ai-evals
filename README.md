# AI Evals Framework

A modular evaluation framework for LLMs, RAG pipelines, and AI systems. Covers accuracy benchmarking, hallucination detection, bias auditing, and prompt regression testing.

## Structure

```
ai-evals/
├── evals/
│   ├── llm_benchmark.py       # Accuracy, BLEU, ROUGE, BERTScore
│   ├── rag_eval.py            # Faithfulness, context recall, answer relevancy
│   ├── hallucination_eval.py  # Factual consistency checks
│   └── bias_eval.py           # Demographic parity, counterfactual fairness
├── runners/
│   ├── eval_runner.py         # Orchestrates eval pipelines
│   └── prompt_regression.py   # Regression tests across prompt versions
├── utils/
│   ├── metrics.py             # Core metric implementations
│   └── data_loader.py         # Dataset utilities
├── configs/
│   └── eval_config.yaml       # Run configurations
├── results/                   # Output JSON/CSV reports
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python runners/eval_runner.py --config configs/eval_config.yaml
```

## Metrics Supported

| Category | Metrics |
|---|---|
| Text Quality | BLEU, ROUGE-L, BERTScore, METEOR |
| RAG | Faithfulness, Context Recall, Answer Relevancy (RAGAS-style) |
| Hallucination | NLI-based factual consistency, entity overlap |
| Bias | Demographic parity, counterfactual consistency, toxicity |
| Latency | p50/p95/p99 token latency, TTFT |
