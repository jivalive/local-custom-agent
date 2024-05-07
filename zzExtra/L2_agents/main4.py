from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
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

MEMORY_KEY = "chat_history"
chat_history = []

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

@tool
def lower_case_function(input_text: str) -> str:
    """Returns the lower case version of a string."""
    print(f"Converting the following string to lower case: {input_text}")
    return input_text.lower()

# # Testing the get_word_length function
# length = get_word_length.invoke("abc")
# print(f"Length of 'abc': {length}")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

tools = [get_word_length, lower_case_function]

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
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
print("Agent pipeline constructed.")

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
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
print(result1["output"])
# input2 = "how many letters in the word car?"
# result2 = agent_executor.invoke({"input": input2, "chat_history": chat_history})
# chat_history.extend(
#     [
#         HumanMessage(content=input2),
#         AIMessage(content=result2["output"]),
#     ]
# )
# input3 = "now how many letters are there in the above two words combined?"
# result3 = agent_executor.invoke({"input": input3, "chat_history": chat_history})
# chat_history.extend(
#     [
#         HumanMessage(content=input3),
#         AIMessage(content=result3["output"]),
#     ]
# )
# final_answer = result3["output"]
# print(final_answer)