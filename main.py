from human_eval_infilling.evaluate_functional_correctness import (
    evaluate_functional_correctness,
)
from human_eval_infilling.generate_samples import generate_samples


def main():
    generate_samples("test", 1)
    evaluate_functional_correctness("test", "data/samples.jsonl")


if __name__ == "__main__":
    main()
