from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="The text to check for AI-generated content (1 to 20000 characters).",
    )

    # the example request shown in /docs
    model_config = {
        "json_schema_extra": {
            "example": {"text": "The quick brown fox jumps over the lazy dog. It was a sunny day, so we went for a walk."}
        }
    }


class DetectResponse(BaseModel):
    check_id: str = Field(..., description="ID you can use to fetch this result again from GET /history/{check_id}.")
    label: str = Field(..., description="'ai' or 'human'.")
    ai_probability: float = Field(..., ge=0, le=1, description="Model's probability (0 to 1) that the text is AI-generated.")

    # the example response shown in /docs
    model_config = {
        "json_schema_extra": {
            "example": {"check_id": "11fe0d18-f679-4f89-8944-33e4c5c72954", "label": "human", "ai_probability": 0.31}
        }
    }


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Short machine-readable error type.")
    detail: str = Field(..., description="Safe human-readable message.")