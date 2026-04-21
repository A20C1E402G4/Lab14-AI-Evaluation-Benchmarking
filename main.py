import asyncio
import json
import os
import time

from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent

# Release-gate thresholds
_MIN_HIT_RATE = 0.50
_MIN_AGREEMENT_RATE = 0.60
_MAX_COST_USD = 5.00


async def run_benchmark_with_results(agent_version: str, agent: MainAgent | None = None):
    print(f"\n🚀 Starting benchmark: {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Missing data/golden_set.jsonl. Run 'python data/synthetic_gen.py' first.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ data/golden_set.jsonl is empty.")
        return None, None

    if agent is None:
        agent = MainAgent()

    judge = LLMJudge()
    retrieval_eval = RetrievalEvaluator()
    runner = BenchmarkRunner(agent, retrieval_eval, judge)
    results = await runner.run_all(dataset)

    total = len(results)
    agreement_stats = judge.get_agreement_stats()
    cost_summary = runner.get_cost_summary()

    metrics = {
        "avg_score": round(sum(r["judge"]["final_score"] for r in results) / total, 4),
        "hit_rate": round(
            sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total, 4
        ),
        "mrr": round(
            sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total, 4
        ),
        "agreement_rate": agreement_stats["agreement_rate"],
        "cohen_kappa": agreement_stats.get("cohen_kappa", 0.0),
        "pass_rate": round(sum(1 for r in results if r["status"] == "pass") / total, 4),
    }

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": metrics,
        "cost": cost_summary,
    }
    return results, summary


async def run_benchmark(version: str, agent: MainAgent | None = None):
    _, summary = await run_benchmark_with_results(version, agent)
    return summary


async def main():
    # V1: basic BM25 top_k=3
    v1_agent = MainAgent(top_k=3, version="v1")
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base", v1_agent)

    # V2: improved BM25 top_k=5 (more context)
    v2_agent = MainAgent(top_k=5, version="v2")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", v2_agent)

    if not v1_results or not v1_summary or not v2_results or not v2_summary:
        print("❌ Benchmark failed. Check data/golden_set.jsonl.")
        return

    # Regression comparison
    delta_score = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    delta_hit = v2_summary["metrics"]["hit_rate"] - v1_summary["metrics"]["hit_rate"]

    print("\n📊 --- REGRESSION COMPARISON ---")
    print(f"V1 avg_score : {v1_summary['metrics']['avg_score']:.4f}")
    print(f"V2 avg_score : {v2_summary['metrics']['avg_score']:.4f}")
    print(f"Delta score  : {'+' if delta_score >= 0 else ''}{delta_score:.4f}")
    print(f"V1 hit_rate  : {v1_summary['metrics']['hit_rate']:.4f}")
    print(f"V2 hit_rate  : {v2_summary['metrics']['hit_rate']:.4f}")
    print(f"Delta hit    : {'+' if delta_hit >= 0 else ''}{delta_hit:.4f}")
    print(f"V2 agreement : {v2_summary['metrics']['agreement_rate']:.4f}")
    print(f"V2 cost      : ${v2_summary['cost']['total_cost_usd']:.4f}")

    # Extended release gate
    gate_checks = {
        "score_improved": delta_score >= 0,
        "hit_rate_ok": v2_summary["metrics"]["hit_rate"] >= _MIN_HIT_RATE,
        "agreement_ok": v2_summary["metrics"]["agreement_rate"] >= _MIN_AGREEMENT_RATE,
        "cost_ok": v2_summary["cost"]["total_cost_usd"] <= _MAX_COST_USD,
    }
    approved = all(gate_checks.values())

    print("\n🔒 Release Gate Checks:")
    for check, passed in gate_checks.items():
        print(f"  {'✅' if passed else '❌'} {check}")

    # Persist results
    os.makedirs("reports", exist_ok=True)

    final_summary = {
        **v2_summary,
        "regression": {
            "v1_metrics": v1_summary["metrics"],
            "v2_metrics": v2_summary["metrics"],
            "delta_score": round(delta_score, 4),
            "delta_hit_rate": round(delta_hit, 4),
        },
        "gate": {
            "decision": "APPROVE" if approved else "BLOCK RELEASE",
            "checks": gate_checks,
        },
    }

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    benchmark_results = {
        "v1": {
            "version": "Agent_V1_Base",
            "summary": v1_summary["metrics"],
            "results": v1_results,
        },
        "v2": {
            "version": "Agent_V2_Optimized",
            "summary": v2_summary["metrics"],
            "results": v2_results,
        },
    }
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    if approved:
        print("\n✅ DECISION: APPROVE RELEASE")
    else:
        print("\n❌ DECISION: BLOCK RELEASE")

    print(f"\n📁 Reports saved to reports/")


if __name__ == "__main__":
    asyncio.run(main())
