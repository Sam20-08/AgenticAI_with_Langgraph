from langchain .document_loaders import YoutubeLoader
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from langchain.text_splitter import TokenTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from typing import TypedDict,Union,List
from langgraph.graph import StateGraph, START,END
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from typing import Annotated
from langgraph.graph.message import add_messages

youtube_url = input("Enter the YouTube URL: ")
# Basic transcript loading

loader = YoutubeLoaderDL.from_youtube_url(
    youtube_url, add_video_info=True
)

"""## Extracting Details of the YouTube Video"""

documents = loader.load()

documents[0].metadata

"""## Extracting Transcript of the video"""

loader=YoutubeLoader.from_youtube_url(youtube_url,language=["en","en-us"])

transcript=loader.load()

print(transcript[-1].page_content)

splitter=TokenTextSplitter(chunk_size=10000, chunk_overlap=100)
chunks=splitter.split_documents(transcript)

len(chunks)

"""#### Initializing the LLM for summarizing"""

llm=ChatGroq(model="llama3-8b-8192",api_key="api_key")

summaerizer_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)
summary=summaerizer_chain.run(chunks)

class Agentstate(TypedDict):
  Title:str
  message:List[Union[HumanMessage,AIMessage,SystemMessage]]

conversational_history=[SystemMessage(content="You are helpful AI assistant for writing better blog post and research about the topics")]

conversational_history.append(HumanMessage(content=transcript[-1].page_content))

def title_maker(state:Agentstate)->Agentstate:
  """I want to generate a title for the blog post based on the transcript."""
  state["Title"]=llm.invoke(documents[0].metadata["title"])
  print(f"\nTitle : {state['Title'].content}")
  return state

def summarizer_agent(state:Agentstate)->Agentstate:
  """I want to summarize the transcript and generate a blog post."""
  response=llm.invoke(state['message'])
  state['message'].append(AIMessage(content=response.content))
  print(f"\nAI : {response.content}")
  # print(f"\nOur current state {state['message']}")
  return state

graph=StateGraph(Agentstate)
graph.add_node("title_maker", title_maker)
graph.add_node("Chatnode",summarizer_agent)
graph.add_edge(START,"title_maker")
graph.add_edge("title_maker","Chatnode")
graph.add_edge("Chatnode",END)
agent=graph.compile()

from IPython.display import Image,display
display(Image(agent.get_graph().draw_mermaid_png()))

while True:
  user_input=input("Enter: ")
  conversational_history.append(HumanMessage(content=user_input))
  if user_input!="exit":
    result=agent.invoke({'message':conversational_history})
  else:
    break

conversational_history=result['message']

print(conversational_history)

