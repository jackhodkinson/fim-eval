import json
import os
import time

from human_eval_infilling.evaluate_functional_correctness import (
    evaluate_functional_correctness,
)

from src.generation.generate import generate_samples

# BENCHMARK_NAME: single-line, multi-line, random-span, random-span-light, test
BENCHMARK_NAME = "single-line"
MODEL = "codellama-7b-code"  # dummy, gpt, gemini, claude, llama
NUM_TASKS = 200
NUM_SAMPLES_PER_TASK = 10
TEMPERATURES = [0.0, 0.25, 0.5, 0.75, 1.0]
K_VALUES = [1, 3, 5, 10]

results = []


def main():
    dirname = f"data/{MODEL}_temperature"
    os.makedirs(dirname, exist_ok=True)

    for temperature in TEMPERATURES:
        t0 = time.time()
        print(f"Generating samples with temperature {temperature}")
        generated_file = f"{dirname}/{BENCHMARK_NAME}.jsonl"

        start_time = time.time()
        generate_samples(
            BENCHMARK_NAME,
            num_samples_per_task=NUM_SAMPLES_PER_TASK,
            output_file=generated_file,
            model=MODEL,
            num_tasks=NUM_TASKS,
            temperature=temperature,
        )
        generation_time = time.time() - start_time
        print(f"Sample generation took {generation_time:.2f} seconds")

        pass_at_k = evaluate_functional_correctness(
            BENCHMARK_NAME,
            generated_file,
            k=K_VALUES,
            n_workers=8,
            num_tasks=NUM_TASKS,
        )
        pass_at_k["temperature"] = temperature
        results.append(pass_at_k)
        print(f"Pass@k for temperature {temperature}:")
        for k, v in pass_at_k.items():
            print(f"Pass@{k}: {v:.2f}")

        print(
            f"Total time for temperature {temperature}: {time.time() - t0:.2f} seconds"
        )

    with open(f"{dirname}/results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print("Summarizing results...")
    print(results)
    for r in results:
        print(f"Temperature {r['temperature']}:")
        for k in K_VALUES:
            print(f"Pass@{k}: {r[f'pass@{k}']:.2f}")
            print("------------------------")


if __name__ == "__main__":
    main()
