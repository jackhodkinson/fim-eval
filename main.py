from human_eval_infilling.evaluate_functional_correctness import (
    evaluate_functional_correctness,
)

from src.generation.generate import generate_samples

BENCHMARK_NAME = "test"


def main():
    generate_samples(BENCHMARK_NAME, 1, "data/example_samples.jsonl")
    evaluate_functional_correctness(BENCHMARK_NAME, "data/example_samples.jsonl")


if __name__ == "__main__":
    main()
