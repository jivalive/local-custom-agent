from langchain.tools import tool, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
from langchain import OpenAI
from typing import Tuple
import time

# Define a function that will be converted into a tool
@tool
def lower_case_function(input_text: str) -> str:
    """Returns the lower case version of a string."""
    return input_text.lower()

# Error handling function
def _handle_error(error) -> str:
    return str(error)[:100]

# Convert the function into a Tool
lower_case_tool = Tool(
    func=lower_case_function,
    name="LowerCaseTool",
    description="Convert text to lowercase"
)

# Initialize the chat model
llm = Ollama(model="openhermes", base_url="http://localhost:11434", verbose=True)

# Initialize the agent with the list of tools
agent = initialize_agent(
    tools=[lower_case_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=4,
    handle_parsing_errors=_handle_error
)

def format_time(seconds: float) -> str:
    """Formats time in MM:SS:ms format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

def run_query(query: str) -> Tuple[str, str]:
    """Executes the given query using the agent and measures the time taken."""
    start_time = time.time()
    output = agent.run(query)
    elapsed_time = time.time() - start_time
    formatted_time = format_time(elapsed_time)
    return output, formatted_time

# Example usage
query = "Convert THIsdisjd hhuSFShdS dshASAd jsdis TEwadXT to lowercase and stop. I want nothing else."
output, time_taken = run_query(query)
print(f"Output: {output}, Time taken: {time_taken}")
