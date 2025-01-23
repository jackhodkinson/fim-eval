from itertools import islice

import tqdm
from human_eval_infilling.data import read_problems, write_jsonl

from src.generation.base import GenerateResponse
from src.generation.claude import generate_completion as claude_generate_completion
from src.generation.dummy import generate_completion as dummy_generate_completion
from src.generation.gemini import generate_completion as gemini_generate_completion
from src.generation.gpt import generate_completion as gpt_generate_completion

MODELS = {
    "gpt": gpt_generate_completion,
    "dummy": dummy_generate_completion,
    "gemini": gemini_generate_completion,
    "claude": claude_generate_completion,
}


def generate_samples(
    benchmark_name: str,
    num_samples_per_task: int,
    output_file: str,
    model: str,
    num_tasks: int = None,
) -> list[dict]:
    print(
        f"Generating {num_samples_per_task} X {num_tasks if num_tasks is not None else 'all'} samples for {benchmark_name} with {model} model"
    )
    assert model in MODELS, f"Model {model} not supported"
    generate_completion = MODELS[model]
    problems = read_problems(benchmark_name=benchmark_name)
    if num_tasks is not None:
        problems = dict(islice(problems.items(), num_tasks))

    samples = []
    for task_id in tqdm.tqdm(problems):
        for _ in range(num_samples_per_task):
            response: GenerateResponse = generate_completion(
                problems[task_id]["prompt"], problems[task_id]["suffix"]
            )
            sample = {**response, "task_id": task_id}
            samples.append(sample)

    write_jsonl(output_file, samples)
    return samples


if __name__ == "__main__":
    generate_samples(
        benchmark_name="test",
        num_samples_per_task=1,
        output_file="data/example_samples.jsonl",
        model="dummy",
    )
