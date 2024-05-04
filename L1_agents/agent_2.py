import os
from langchain.agents import initialize_agent
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
# from langchain_experimental.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import Ollama
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.globals import set_debug, set_verbose
from langchain import OpenAI
from dotenv import load_dotenv

# set_debug(True)
# set_verbose(True)

# Load environment variables from .env file
load_dotenv()

# Access the OPENAI API key securely
api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is set in environment (should be done securely, not hardcoded)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise EnvironmentError("API key not found. Please check your .env file.")

# openai
# llm = OpenAI(temperature=0)

# # # # define LLM
llm = Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)


# define tools
wikipedia = WikipediaAPIWrapper()
python_repl = PythonREPL()
search = DuckDuckGoSearchRun()

python_repl_tool = Tool(
        name = "python_repl",
        func=python_repl.run,
        description="useful for when you strictly need to use python code to answer a question. You should input python code. You should take care of the syntax of the python code, it should not have (`) symbol which can break the code. code should have proper paranthesis."
    )
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
tools = [duckduckgo_tool, wikipedia_tool, python_repl_tool]

# define agent
zero_shot_agent = initialize_agent(
    # agent="zero-shot-react-description",
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
)


# test
# zero_shot_agent.run("write 5 pointers about llama3 model.")
zero_shot_agent.run("What is olympics, explain in one sentence.")
# result = zero_shot_agent.run("What is 1+1")
# print(zero_shot_agent.agent.llm_chain.prompt.template)
# python_repl.run("what is 17324 multiply by 232423")
# zero_shot_agent.run("what is 17324 multiply by 232423")
# zero_shot_agent.run("Tell me April 2024 news of Singapore")
# zero_shot_agent.run('what is the current price of btc')
# zero_shot_agent.run('Is 11 a prime number?')
# zero_shot_agent.run('Write a function to check if 11 a prime number and test it')
# print(result)


