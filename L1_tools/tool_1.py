from langchain.tools import tool
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama

@tool("lower_case", return_direct=True)
def to_lower_case(input:str) -> str:
  """Returns the input as all lower case."""
  return input.lower()

tools = [
    Tool(
        name = "lower_case",
        func=to_lower_case,
        description="useful for when you need to return the input as all lower case."
    )
]
tools.append(to_lower_case)

llm = Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("write 5 sentences about Narendra Modi. Return the answer in all lower case")

