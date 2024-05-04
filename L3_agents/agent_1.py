from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from pydantic import BaseModel, Field
from langchain.llms import Ollama
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# Load environment variables from .env file
load_dotenv()

# Access the OPENAI API key securely
try:
    api_key = os.getenv("OPENAI_API_KEY")
except:
    pass
# Ensure API key is set in environment (should be done securely, not hardcoded)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    pass
    # raise EnvironmentError("API key not found. Please check your .env file.")

MEMORY_KEY = "chat_history"
chat_history = []

# Initialize the LLM model with an API key and specific model configuration.
# llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)
llm = Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)
print("LLM Initialized.")


class CustomToolInput(BaseModel):
    query: str = Field(..., description="The word to calculate the length for")

# Verify instantiation
print("CustomToolInput is a BaseModel:", issubclass(CustomToolInput, BaseModel))

def custom_tool_func(query: CustomToolInput) -> str:
    """Returns the length of a word."""
    print(f"Calculating the length of the word: {query.query}")
    return len(query.query)

from langchain.tools import Tool

# Attempt to create the tool with explicit typing
try:
    custom_tool = Tool.from_function(
        func=custom_tool_func,
        name="GetWordLength",
        description="Calculates the length of a given word",
        args_schema=CustomToolInput
    )
    print("Tool created successfully.")
except Exception as e:
    print("Failed to create tool:", e)
    
    
# Creating a prompt template for use with the langchain LLM.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are very powerful assistant, but don't know current events"),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
print(f"Prompt Template Created: {prompt}")

tools = [custom_tool]



# Binding tools to the LLM and creating a chained agent with a prompt and output parser.
# llm_with_tools = llm.bind_tools(tools)
print("Tools bound to LLM.")

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm
    | OpenAIToolsAgentOutputParser()
)
print("Agent pipeline constructed.")

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    llm = llm,
    verbose=True, 
    return_intermediate_steps=True
)
print("Agent Executor ready.")

input1 = "how many letters in the word banana?"
result1 = agent_executor.invoke({"input": input1, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result1["output"]),
    ]
)
input2 = "how many letters in the word car?"
result2 = agent_executor.invoke({"input": input2, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input2),
        AIMessage(content=result2["output"]),
    ]
)
input3 = "now how many letters are there in the above two words combined?"
result3 = agent_executor.invoke({"input": input3, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input3),
        AIMessage(content=result3["output"]),
    ]
)
final_answer = result3["output"]
print(final_answer)
