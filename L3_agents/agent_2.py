from langchain.tools import Tool, tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import os
from langchain import OpenAI

# # set_debug(True)
# # set_verbose(True)

# # Load environment variables from .env file
# load_dotenv()

# # Access the OPENAI API key securely
# api_key = os.getenv("OPENAI_API_KEY")

# # Ensure API key is set in environment (should be done securely, not hardcoded)
# if api_key:
#     os.environ["OPENAI_API_KEY"] = api_key
# else:
#     raise EnvironmentError("API key not found. Please check your .env file.")

# # openai
# llm = OpenAI(temperature=0)



# Define a function that will be converted into a tool
@tool
def lower_case_function(input_text: str) -> str:
    """Returns the lower case version of a string."""
    print(f"Converting the following string to lower case: {input_text}")
    return input_text.lower()

def _handle_error(error) -> str:
    return str(error)[:100]

# Convert the function into a Tool
lower_case_tool = Tool(
    func=lower_case_function,
    name="LowerCaseTool",
    description="Useful for when you need to convert text to lowercase"
)

# Initialize the chat model
llm = Ollama(model="openhermes", base_url="http://localhost:11434", verbose=True)

# Create a list of tools
tools = [lower_case_tool]

# # Initialize the agent with the list of tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=4,
    handle_parsing_errors=_handle_error
)

# Execute the tool within the agent
# input = "Convert 'THIS TEXT' to lowercase and stop. I want nothing else."
output = agent.run("Convert THIsdisjd hhuSFShdS dshASAd jsdis TEwadXT to lowercase and stop. I want nothing else.")
# output = agent.invoke({"input": input})
# print(output)


# agent_obj = StructuredChatAgent.from_llm_and_tools(
#     llm=llm,
#     tools=tools,
# )

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     return_intermediate_steps=True
# )

# output = agent_executor({"input": input})
print(output)