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


os.environ["OPENAI_API_KEY"] = api_key

from langchain.utilities import WikipediaAPIWrapper
wikipedia = WikipediaAPIWrapper()

from langchain.utilities import PythonREPL
python_repl = PythonREPL()

python_repl.run("print(17*2)")

from langchain.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

from langchain import OpenAI

llm = OpenAI(temperature=0)

from langchain.agents import Tool

tools = [
    Tool(
        name = "python repl",
        func=python_repl.run,
        description="useful for when you need to use python to answer a question. You should input python code"
    )
]

wikipedia_tool = Tool(
    name='wikipedia',
    func= wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
)

duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

tools.append(duckduckgo_tool)
tools.append(wikipedia_tool)

from langchain.agents import initialize_agent

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
)

zero_shot_agent.run("When was Barak Obama born?")

zero_shot_agent.run("What is 17*6?")

print(zero_shot_agent.agent.llm_chain.prompt.template)


zero_shot_agent.run("Tell me about LangChain")
zero_shot_agent.run("Tell me about Singapore")
zero_shot_agent.run('what is the current price of btc')
zero_shot_agent.run('Is 11 a prime number?')
zero_shot_agent.run('Write a function to check if 11 a prime number and test it')


