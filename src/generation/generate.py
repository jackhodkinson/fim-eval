from human_eval_infilling.data import read_problems, write_jsonl

from src.generation.dummy import generate_completion as dummy_generate_completion
from src.generation.gpt import generate_completion as gpt_generate_completion

MODELS = {
    "gpt": gpt_generate_completion,
    "dummy": dummy_generate_completion,
}


def generate_samples(
    benchmark_name: str,
    num_samples_per_task: int,
    output_file: str,
    model: str,
) -> list[dict]:
    assert model in MODELS, f"Model {model} not supported"
    generate_completion = MODELS[model]
    problems = read_problems(benchmark_name=benchmark_name)

    samples = [
        dict(
            task_id=task_id,
            completion=generate_completion(
                problems[task_id]["prompt"], problems[task_id]["suffix"]
            ),
        )
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    write_jsonl(output_file, samples)
    return samples


if __name__ == "__main__":
    generate_samples(
        benchmark_name="test",
        num_samples_per_task=1,
        output_file="data/example_samples.jsonl",
        model="dummy",
    )
