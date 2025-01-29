import os

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.analysts import (
  GenerateAnalystsState,
  generate_analysts,
  get_human_feedback,
  should_continue,
)

load_dotenv()


# Define a new graph

builder = StateGraph(GenerateAnalystsState)
builder.add_node(generate_analysts)
builder.add_node(get_human_feedback)

builder.add_edge(START, "generate_analysts")
builder.add_edge("generate_analysts", "get_human_feedback")
builder.add_conditional_edges(
  "get_human_feedback", should_continue, ["generate_analysts", END]
)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["get_human_feedback"], checkpointer=memory)

file_paht = os.path.join(os.path.dirname(__file__), "graph.png")
graph.get_graph().draw_mermaid_png(output_file_path=file_paht)
