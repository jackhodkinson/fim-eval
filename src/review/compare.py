from human_eval_infilling.evaluate_functional_correctness import (
    evaluate_functional_correctness,
)

# BENCHMARK_NAME: single-line, multi-line, random-span, random-span-light, test
BENCHMARK_NAME = "single-line"
NUM_TASKS = 100
NUM_SAMPLES_PER_TASK = 5


def main():
    for model in ["codellama-7b-code", "gpt", "gemini", "claude", "llama"]:
        generated_file = f"./data/{model}_{BENCHMARK_NAME}.jsonl"

        evaluate_functional_correctness(
            BENCHMARK_NAME,
            generated_file,
            k=[1, 3, 5],
            n_workers=8,
            num_tasks=NUM_TASKS,
        )


if __name__ == "__main__":
    main()
