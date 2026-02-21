from typing import Literal

from pydantic import BaseModel, Field


class YesNoResponse(BaseModel):
    """強制只回 yes/no 的回應"""
    answer: Literal["yes", "no"] = Field(description="只能回答 'yes' 或 'no'")
