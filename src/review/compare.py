# Script to review failed results
import gzip
import json

from human_eval_infilling.data import BENCHMARK_NAME_TO_PATH
from rich import print
from rich.console import Console

BENCHMARK_NAME = "single-line"
MODEL = "gpt"

test_path = BENCHMARK_NAME_TO_PATH[BENCHMARK_NAME]

console = Console()


def load_tasks():
    tasks = []
    with gzip.open(test_path, "rb") as f:
        for line in f:
            sample = json.loads(line)
            tasks.append(sample)
    return tasks


def load_results():
    generated_file = f"data/{MODEL}_{BENCHMARK_NAME}.jsonl_results.jsonl"
    results = []
    with open(generated_file, "rt") as f:
        for line in f:
            sample = json.loads(line)
            results.append(sample)
    return results


def main():
    tasks = {task["task_id"]: task for task in load_tasks()}
    results = load_results()
    print(f"Got {len(results)} results and {len(tasks)} tasks")
    attempted_tasks = set(result["task_id"] for result in results)
    passed_tasks = set(result["task_id"] for result in results if result["passed"])
    always_failed_tasks = set(
        task_id for task_id in attempted_tasks if task_id not in passed_tasks
    )
    print(f"Attempted tasks: {len(attempted_tasks)}")
    print(f"Passed tasks: {len(passed_tasks)}")
    print(f"Always failed tasks: {len(always_failed_tasks)}")

    # calculate pass@k
    for k in [1, 3, 5]:
        pass_at_k = sum(1 for result in results if result["passed"] and result["rank"] <= k) / len(results)


if __name__ == "__main__":
    main()
