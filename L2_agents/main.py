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

llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)

from langchain.agents import tool
from langchain.agents import AgentExecutor


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return (len(word) + 2)


print(get_word_length.invoke("abc"))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
print(prompt)

tools = [get_word_length]

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
llm_with_tools = llm.bind_tools(tools)
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = "what is the length of the word 'elephant'?"


result = agent_executor.invoke({"input": query})
print(result)