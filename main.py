import time

from human_eval_infilling.evaluate_functional_correctness import (
    evaluate_functional_correctness,
)

from src.generation.generate import generate_samples

# BENCHMARK_NAME: single-line, multi-line, random-span, random-span-light, test
BENCHMARK_NAME = "single-line"
MODEL = "codellama:7b-code"  # dummy, gpt, gemini, claude, llama
NUM_TASKS = 100
NUM_SAMPLES_PER_TASK = 5


def main():
    generated_file = f"data/{MODEL}_{BENCHMARK_NAME}.jsonl"

    start_time = time.time()
    generate_samples(
        BENCHMARK_NAME,
        num_samples_per_task=NUM_SAMPLES_PER_TASK,
        output_file=generated_file,
        model=MODEL,
        num_tasks=NUM_TASKS,
    )
    generation_time = time.time() - start_time
    print(f"Sample generation took {generation_time:.2f} seconds")

    evaluate_functional_correctness(
        BENCHMARK_NAME,
        generated_file,
        k=[1, 3, 5],
        n_workers=8,
        num_tasks=NUM_TASKS,
    )


if __name__ == "__main__":
    main()
