import os

from openai import OpenAI

from src.generation.base import GenerateResponse

client = OpenAI(
    base_url=os.getenv("REMOTE_MODEL_BASE_URL"),
    api_key=os.getenv("REMOTE_MODEL_API_KEY"),
)

with open("./src/generation/system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


def generate_completion(
    prefix: str, suffix: str, model: str, temperature: float = 0.2
) -> GenerateResponse:
    """
    Connect to a remote client serving an OpenAI compatible API
    Uses the completion endpoint
    You can host one on Modal: https://modal.com/docs/examples/vllm_inference
    """
    # You should change this depending on what model(s) you are hosting on your remote server
    assert model == "DeepSeek-Coder-V2-Lite-Base"
    # Use the completions endpoint instead of chat
    response = client.completions.create(
        model=model,
        prompt=f"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>",
        temperature=temperature,
        max_tokens=512,  # Adjust as needed
    )

    return GenerateResponse(
        completion=response.choices[0].text.replace(" <EOT>", ""),
        input_tokens=0,
        output_tokens=0,
        model=model,
        cost=0.0,  # Ollama is free to use locally
    )


if __name__ == "__main__":
    prefix = "def compute_gcd(x, y):"
    suffix = "return result"
    result = generate_completion(prefix, suffix, "DeepSeek-Coder-V2-Lite-Base")
    print(result)
    print()
    completion = result["completion"]
    print(prefix + completion + suffix)
