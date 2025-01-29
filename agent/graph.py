import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.analysts import (
  generate_analysts,
  human_feedback,
)
from agent.interview import interview_builder
from agent.report import (
  finalize_report,
  write_conclusion,
  write_introduction,
  write_report,
)
from agent.research import ResearchGraphState

load_dotenv()


def initiate_interviews(state: ResearchGraphState):
  """This is the "map" step where we run eatch interview sub-graph using Send API"""
  feedback = state.human_feedback_for_analysts or "approve"
  if feedback.lower() != "approve":
    return "create_analysts"
  else:
    return [
      Send(
        "conduct_interview",
        {
          "analyst": analyst,
          "messages": [
            HumanMessage(
              content=f"So you said you were writing an article on {state.topic}"
            )
          ],
        },
      )
      for analyst in state.analysts
    ]


# Define a new graph

builder = StateGraph(ResearchGraphState)

# add nodes
builder.add_node("create_analysts", generate_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# add edges
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
  "human_feedback",
  initiate_interviews,  # type: ignore ...Return values must be hashable but Send is not...
  ["create_analysts", "conduct_interview"],
)
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(
  ["write_report", "write_introduction", "write_conclusion"], "finalize_report"
)
builder.add_edge("finalize_report", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

file_paht = os.path.join(os.path.dirname(__file__), "graph.png")
graph.get_graph(xray=1).draw_mermaid_png(output_file_path=file_paht)
