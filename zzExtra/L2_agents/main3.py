from langchain_openai import ChatOpenAI
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

# Initialize the LLM model with an API key and specific model configuration.
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)
print("LLM Initialized.")

from langchain.agents import tool
from langchain.agents import AgentExecutor

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    print(f"Calculating the length of the word: {word}")
    return len(word)

# Testing the get_word_length function
length = get_word_length.invoke("abc")
print(f"Length of 'abc': {length}")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Creating a prompt template for use with the langchain LLM.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are very powerful assistant, but don't know current events"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
print(f"Prompt Template Created: {prompt}")

tools = [get_word_length]

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# Binding tools to the LLM and creating a chained agent with a prompt and output parser.
llm_with_tools = llm.bind_tools(tools)
print("Tools bound to LLM.")

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
print("Agent pipeline constructed.")

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print("Agent Executor ready.")

query = "what is the length of the word 'example'?"

# Invoking the agent with a query.
result = agent_executor.invoke({"input": query})
print(f"Query Result: {result}")
