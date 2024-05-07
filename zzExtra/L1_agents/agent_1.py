# langchain[docarray]==0.0.284
import os
from langchain.agents import initialize_agent
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import Ollama
from langchain.agents import Tool
from langchain import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access the OPENAI API key securely
api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is set in environment (should be done securely, not hardcoded)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise EnvironmentError("API key not found. Please check your .env file.")

# # openai
# llm = OpenAI(temperature=0)

# # define LLM
llm = Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)

# define tools
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name='wikipedia',
    func= wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
)

# setting tools
tools = [
    search_tool,
    wikipedia_tool
]

# define agent
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
)

zero_shot_agent.run("write 5 pointers about llama3 model.")



