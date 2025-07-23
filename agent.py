from typing import TypedDict
from typing import Annotated 
from langgraph.graph import START,END
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import tools_condition

from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="agentic-ai"

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

class Agentstate(TypedDict):
  messages: Annotated[list[BaseMessage],add_messages]

def make_tool_call():
  
  from langchain_core.tools import tool
  @tool
  def add_numbers(a: int, b: int) -> int:
      """Add two numbers."""
      return a + b

  tools = [add_numbers]

  llm_with_tools = llm.bind_tools(tools)

  def chat(state:Agentstate)->Agentstate:
    return {"messages":[llm_with_tools.invoke(state["messages"])]}
  
  graph=StateGraph(Agentstate)
  graph.add_node("chat",chat)
  graph.add_node("tools",ToolNode([add_numbers]))

  graph.add_conditional_edges(
      "chat",
      tools_condition,
  )

  graph.add_edge(START,"chat")
  graph.add_edge("tools",END)
  agent=graph.compile()
  return agent


tool_agent=make_tool_call()