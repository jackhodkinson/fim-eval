from src.generation.base import GenerateResponse


def generate_completion(
    prefix: str, suffix: str, temperature: float
) -> GenerateResponse:
    """
    Dummy generatino to test the pipeline without incurring costs.
    """
    return GenerateResponse(
        completion="<dummy>",
        input_tokens=0,
        output_tokens=0,
        model="dummy",
        cost=0.0,
    )
