import operator
from typing import Annotated

from pydantic import BaseModel, Field

from agent.analysts import Analyst


class ResearchGraphState(BaseModel):
  topic: str = Field(..., description="Research topic")
  max_analysts: int = Field(2, description="Maximum number of analysts to interview")
  human_feedback_for_analysts: str | None = Field(
    None, description="Human feedback to gather analyst team members"
  )
  analysts: list[Analyst] = Field([], description="Analysts asking questions")
  sections: Annotated[list[str], operator.add] = Field([], description="Send() API key")
  introduction: str = Field("", description="Introduction for the final report")
  content: str = Field("", description="Content for the final report")
  conclusion: str = Field("", description="Conclusion for the final report")
  final_report: str = Field("", description="Final report")
  translated_report: str = Field("", description="Translated report")
