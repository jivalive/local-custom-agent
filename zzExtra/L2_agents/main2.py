import logging
import textwrap
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
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


class TextWrapperHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    def emit(self, record):
        width = 80  # Set the desired width of the output
        msg = self.format(record)
        wrapped_text = textwrap.fill(msg, width=width)
        separation_line = "-" * width  # Define a separation line with the same width
        with open('debug.log', 'a') as file:
            file.write(wrapped_text + '\n' + separation_line + '\n')

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = TextWrapperHandler()
logger.addHandler(handler)

# Initialize the ChatGPT model with OpenAI's API
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)
logging.debug("LLM initialized with API key and model configuration.")

# Define a tool function to get the length of a word
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    length = len(word)
    logging.debug(f"Computed length of word '{word}': {length}")
    return length

word_length = get_word_length.invoke("abc")
print(f"Length of 'abc': {word_length}")
logging.debug(f"Length of 'abc': {word_length}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are very powerful assistant, but don't know current events"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
print("Prompt Template:", prompt)
logging.debug(f"Prompt Template initialized: {prompt}")

tools = [get_word_length]
print("Tools defined:", tools)
logging.debug(f"Tools defined: {tools}")

llm_with_tools = llm.bind_tools(tools)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

print("Agent configured.")
logging.debug("Agent configured with tools and templates.")
