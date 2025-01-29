from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class Analyst(BaseModel):
  affiliation: str = Field(description="Primary affiliation of the analyst.")
  name: str = Field(description="Name of the analyst.")
  role: str = Field(description="Role of the analyst in the context of the topic.")
  description: str = Field(
    description="Description of the analyst focus, concerns and motives."
  )

  @property
  def persona(self) -> str:
    return f"""\
      Name: {self.name}
      Role: {self.role}
      Affiliation: {self.affiliation}
      Description: {self.description}
    """


class Perspectives(BaseModel):
  analysts: list[Analyst] = Field(
    description="Comprehensive list of analysts with roles and affiliations."
  )


class GenerateAnalystsState(BaseModel):
  topic: str = Field(description="Research topic.")
  max_analysts: int = Field(description="Maximum number of analysts to generate.")
  human_feedback: str = Field(
    description="Human feedback on the generated analysts.", default=""
  )
  analysts: list[Analyst] = Field(
    description="Generated analysts.", default_factory=list
  )


instruction = """\
You are tasked with creating a set of AI analyst personas.
Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme.
"""


def generate_analysts(state: GenerateAnalystsState):
  """Create a set of AI analyst personas."""

  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", instruction),
      ("human", "Generate the set of analysts."),
    ]
  )

  structured_llm = llm.with_structured_output(Perspectives)
  chain = prompt | structured_llm

  perspectives = chain.invoke(
    {
      "topic": state.topic,
      "max_analysts": state.max_analysts,
      "human_feedback": state.human_feedback,
    }
  )
  return {"analysts": perspectives.analysts}  # type: ignore


def get_human_feedback(state: GenerateAnalystsState):
  """No-op node for human feedback."""
  pass


def should_continue(state: GenerateAnalystsState):
  """Return the next node to run."""
  feedback = state.human_feedback or None
  if feedback:
    return "generate_analysts"
  return END
