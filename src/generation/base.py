from typing import TypedDict


class GenerateResponse(TypedDict):
    completion: str
    input_tokens: int
    output_tokens: int
    model: str
    cost: float


def generate_completion(
    prefix: str, suffix: str, temperature: float
) -> GenerateResponse:
    raise NotImplementedError("Base class")
