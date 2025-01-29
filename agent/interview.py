import operator
from typing import Annotated

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import AIMessage, AnyMessage, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agent.analysts import Analyst
from agent.api import llm, web_seaerch


class InterviewState(BaseModel):
  messages: Annotated[list[AnyMessage], add_messages] = Field(
    [], description="List of messages in the conversation"
  )
  analyst: Analyst = Field(..., description="The analyst asking questions")
  max_num_turns: int = Field(
    2, description="The maximum number of turns in the interview"
  )
  context: Annotated[list[str], operator.add] = Field([], description="Source docs")
  interview: str = Field("", description="The interview transcript")
  sections: list[str] = Field(
    [], description="Final key we duplicate in outer state for Send() API"
  )


class SearchQuery(BaseModel):
  search_query: str | None = Field(
    default=None, description="Search query for retrieval"
  )


question_instruction = """\
You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you.
"""


def generate_question(state: InterviewState):
  """This is an analyst node that generates a question"""

  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", question_instruction),
      MessagesPlaceholder("messages"),
    ]
  )

  chain = prompt | llm
  question = chain.invoke({"goals": state.analyst.persona, "messages": state.messages})

  return {"messages": [question]}


search_instruction = """\
You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query
"""


def search_web(state: InterviewState):
  """Retrieve docs from  web search"""

  prompt = ChatPromptTemplate.from_messages(
    [("system", search_instruction), MessagesPlaceholder("messages")]
  )
  chain = prompt | llm.with_structured_output(SearchQuery)
  query = chain.invoke({"messages": state.messages})

  docs = web_seaerch.invoke(query.search_query)  # type: ignore

  formatted_docs = "\n\n---\n\n".join(
    [f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' for doc in docs]
  )

  return {"context": [formatted_docs]}


def search_wikipedia(state: InterviewState):
  """Retrieve docs from wikipedia"""

  prompt = ChatPromptTemplate.from_messages(
    [("system", search_instruction), MessagesPlaceholder("messages")]
  )
  chain = prompt | llm.with_structured_output(SearchQuery)
  query = chain.invoke({"messages": state.messages})

  docs = WikipediaLoader(query=query.search_query, load_max_docs=2).load()  # type: ignore

  formatted_docs = "\n\n---\n\n".join(
    [
      f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
      for doc in docs
    ]
  )

  return {"context": [formatted_docs]}


answer_instruction = """\
You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation.
"""


def generate_answer(state: InterviewState):
  """This is an export node that generates an answer to the question"""

  prompt = ChatPromptTemplate.from_messages(
    [("system", answer_instruction), MessagesPlaceholder("messages")]
  )

  chain = prompt | llm
  answer = chain.invoke(
    {
      "goals": state.analyst.persona,
      "context": state.context,
      "messages": state.messages,
    }
  )
  answer.name = "expert"

  return {"messages": [answer]}


def save_interview(state: InterviewState):
  """Save the interview transcript"""

  interview = get_buffer_string(state.messages)
  return {"interview": interview}


def route_messages(state: InterviewState, name: str = "expert"):
  """Route between question and answer"""

  messages = state.messages
  max_num_turns = state.max_num_turns

  num_responses = len(
    [m for m in messages if isinstance(m, AIMessage) and m.name == name]
  )

  if num_responses >= max_num_turns:
    return "save_interview"

  # This router is run after each question and answer pair
  # Get the last question asked to check if it signals the end of the interview
  last_question = messages[-2]

  if "Thank you so much for your help" in last_question.content:
    return "save_interview"
  return "ask_question"


section_writer_instruction = """\
You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed
"""


def write_section(state: InterviewState):
  """Node to write a section of the report from the interview transcript and context"""

  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", section_writer_instruction),
      ("human", "Use this source to write your section: {context}"),
    ]
  )

  chain = prompt | llm
  section = chain.invoke({"focus": state.analyst.description, "context": state.context})

  return {"sections": [section.content]}


interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node(search_web)
interview_builder.add_node(search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node(save_interview)
interview_builder.add_node(write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges(
  "answer_question", route_messages, ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)
