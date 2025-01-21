from human_eval_infilling.data import read_problems, write_jsonl

from src.generation.gpt import generate_completion


def generate_samples(
    benchmark_name: str, num_samples_per_task: int, output_file: str
) -> list[dict]:
    problems = read_problems(benchmark_name=benchmark_name)
    num_samples_per_task = 1
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
    generate_samples("test", 1, "data/example_samples.jsonl")
