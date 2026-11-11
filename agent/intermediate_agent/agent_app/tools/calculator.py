"""Native LangChain tools: schemas come from typed, validated arguments."""

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field


class TwoNumbers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: float = Field(strict=True, allow_inf_nan=False, ge=-1e100, le=1e100,
                     description="First number")
    b: float = Field(strict=True, allow_inf_nan=False, ge=-1e100, le=1e100,
                     description="Second number")


@tool(args_schema=TwoNumbers)
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b


@tool(args_schema=TwoNumbers)
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers and return their product."""
    return a * b


add_numbers.metadata = {"data_access": "read_only"}
multiply_numbers.metadata = {"data_access": "read_only"}
