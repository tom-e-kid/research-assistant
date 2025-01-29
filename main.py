import argparse

from langchain_core.runnables import RunnableConfig

from agent.graph import graph


def parse_args():
  parser = argparse.ArgumentParser(
    description="Run conversation with configurable parameters"
  )
  parser.add_argument(
    "--thread-id", type=str, default="1", help="Thread ID for the conversation"
  )
  parser.add_argument(
    "--topic", type=str, default=None, help="Topic for the conversation"
  )
  return parser.parse_args()


def run():
  args = parse_args()
  config: RunnableConfig = {"configurable": {"thread_id": args.thread_id or "1"}}

  if args.topic:
    for event in graph.stream(
      {"topic": args.topic, "max_analysts": 3},
      config,
      stream_mode="values",
    ):
      analysts = event.get("analysts", None)
      if analysts:
        for analyst in analysts:
          print(f"Name: {analyst.name}")
          print(f"Affiliation: {analyst.affiliation}")
          print(f"Role: {analyst.role}")
          print(f"Description: {analyst.description}")
          print("-" * 50)

    state = graph.get_state(config)
    print(state.next)
  else:
    raise ValueError("Topic is required")


if __name__ == "__main__":
  run()
