from human_eval_infilling.evaluate_functional_correctness import (
    evaluate_functional_correctness,
)

from src.generation.generate import generate_samples

BENCHMARK_NAME = "test"
MODEL = "dummy"


def main():
    generated_file = f"data/{MODEL}_{BENCHMARK_NAME}.jsonl"

    generate_samples(
        BENCHMARK_NAME,
        num_samples_per_task=1,
        output_file=generated_file,
        model=MODEL,
    )
    pass_at_k = evaluate_functional_correctness(
        BENCHMARK_NAME,
        generated_file,
        n_workers=8,
    )
    print(pass_at_k)


if __name__ == "__main__":
    main()
