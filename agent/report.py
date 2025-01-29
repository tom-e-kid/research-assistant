from langchain_core.prompts import ChatPromptTemplate

from agent.api import llm
from agent.research import ResearchGraphState

report_writer_instruction = """\
You are a technical writer creating a report on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}
"""


def write_report(state: ResearchGraphState):
  """Write content for the final report"""

  context = "\n\n".join([f"{section}" for section in state.sections])
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", report_writer_instruction),
      ("human", "Write a report based upon these memos."),
    ]
  )
  chain = prompt | llm
  output = chain.invoke({"topic": state.topic, "context": context})
  return {"content": output.content}


intoro_conclusion_instruction = """\
You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {context}
"""


def write_introduction(state: ResearchGraphState):
  """Write the introduction for the final report"""

  context = "\n\n".join([f"{section}" for section in state.sections])
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", intoro_conclusion_instruction),
      ("human", "Write the report introduction."),
    ]
  )
  chain = prompt | llm
  output = chain.invoke({"topic": state.topic, "context": context})
  return {"introduction": output.content}


def write_conclusion(state: ResearchGraphState):
  """Write the conclusion for the final report"""

  context = "\n\n".join([f"{section}" for section in state.sections])
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", intoro_conclusion_instruction),
      ("human", "Write the report conclusion."),
    ]
  )
  chain = prompt | llm
  output = chain.invoke({"topic": state.topic, "context": context})
  return {"conclusion": output.content}


def finalize_report(state: ResearchGraphState):
  """\
  This is the "reduce" step where we gather all the sections,
  combine them, and reflect on them to write the introduction and conclusion
  """

  content = state.content
  if content.startswith("## Insights"):
    content = content.replace("## Insights", "")
  if "## Sources" in content:
    try:
      content, sources = content.split("## Sources")
    except ValueError:
      sources = None
  else:
    sources = None

  report = (
    state.introduction + "\n\n---\n\n" + content + "\n\n---\n\n" + state.conclusion
  )
  if sources is not None:
    report += "\n\n## Sources\n" + sources

  return {"final_report": report}


translate_instruction = """\
You are a professional translator.  
Your task is to translate the provided report into the appropriate target language.  

**Translation Rules:**  
- Determine the appropriate target language based on the topic provided by the user.  
  - If the target language is unclear, default to **English**.  
- **Output only the translated text. Do not include any extra commentary or explanations.**  
- **Use natural and clear expressions in the target language, avoiding literal translations that may sound unnatural.**  
- **For technical terms or specialized vocabulary, use the most appropriate equivalent in the target language.**  
- **Ensure that the translation accurately conveys the original meaning while maintaining readability.**  

"""

user_prompt = """\
Please translate the following report into the appropriate language.

**Topic:** {topic}  
**Report:**  
{report}

"""


def translate_report(state: ResearchGraphState):
  """Translate the report into Japanese."""

  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", translate_instruction),
      ("human", user_prompt),
    ]
  )
  chain = prompt | llm
  output = chain.invoke({"topic": state.topic, "report": state.final_report})
  return {"translated_report": output.content}
