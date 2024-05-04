#!pip install langchain openai google-search-results
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent


search = SerpAPIWrapper(serpapi_api_key=serp_api)
tools = [
Tool(
name = "Current Search",
func=search.run,
description="useful for when you need to answer questions about current events or the current state of the world"
),
]

memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(openai_api_key=api_key)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

agent_chain.run(input="hi, i am bob")
agent_chain.run("what are some good dinners to make this week, if i like Italian food?")

