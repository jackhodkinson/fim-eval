from human_eval_infilling.evaluate_functional_correctness import (
    evaluate_functional_correctness,
)

from src.generation.generate import generate_samples

# BENCHMARK_NAME: single-line, multi-line, random-span, random-span-light, test
BENCHMARK_NAME = "single-line"
MODEL = "gpt"  # dummy, gpt
NUM_TASKS = 10


def main():
    generated_file = f"data/{MODEL}_{BENCHMARK_NAME}.jsonl"

    generate_samples(
        BENCHMARK_NAME,
        num_samples_per_task=1,
        output_file=generated_file,
        model=MODEL,
        num_tasks=NUM_TASKS,
    )
    evaluate_functional_correctness(
        BENCHMARK_NAME,
        generated_file,
        n_workers=8,
        num_tasks=NUM_TASKS,
    )
    


if __name__ == "__main__":
    main()
