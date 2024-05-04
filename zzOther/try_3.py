import os
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OPENAI API key securely
api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is set in environment (should be done securely, not hardcoded)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise EnvironmentError("API key not found. Please check your .env file.")

# Set up the environment for OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key

# Importing necessary modules from langchain-community instead of langchain
from langchain_community.utilities import WikipediaAPIWrapper, PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import OpenAI
from langchain.agents import Tool, create_react_agent

# Instantiate the Wikipedia API Wrapper
wikipedia = WikipediaAPIWrapper()

# Instantiate the Python REPL for executing Python code
python_repl = PythonREPL()

# Instantiate the DuckDuckGo Search tool for internet searches
search = DuckDuckGoSearchRun()

# Instantiate the OpenAI model with no temperature setting for deterministic outputs
llm = OpenAI(temperature=0)

# List of tools to be used with the agent
tools = [
    Tool(name="python repl", func=python_repl.run, description="Execute Python code"),
    Tool(name="DuckDuckGo Search", func=search.run, description="Internet search for information"),
    Tool(name='wikipedia', func=wikipedia.run, description="Look up Wikipedia articles")
]

# Initialize the agent with the specified tools and language model
agent = create_react_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

# Execute various queries with the agent
agent.run("When was Barack Obama born?")
agent.run("What is 17*6?")
agent.run("Tell me about LangChain")
agent.run("Tell me about Singapore")
agent.run("What is the current price of BTC?")
agent.run("Is 11 a prime number?")
agent.run("Write a function to check if 11 is a prime number and test it")

# Print the template used by the agent for its operations (optional)
print(agent.agent.llm_chain.prompt.template)
