# from langchain.tools import tool, Tool
# from langchain.agents import initialize_agent, AgentType
# from langchain.llms import Ollama
# import pandas as pd
# from typing import Tuple
# import os
# import time

# # Define a function to find Excel or CSV files in the download folder
# @tool
# def list_files(directory: str) -> str:
#     """Lists all Excel or CSV files in the given directory."""
#     files = [f for f in os.listdir(directory) if f.endswith('.xlsx') or f.endswith('.csv')]
#     if files:
#         return "Available files: " + ", ".join(files)
#     else:
#         return "No Excel or CSV files found."

# # Define a function to get column names from a given file
# @tool
# def get_column_names(filename: str, directory: str) -> str:
#     """Returns column names from the specified Excel or CSV file in the given directory."""
#     try:
#         file_path = os.path.join(directory, filename)
#         if filename.endswith('.csv'):
#             df = pd.read_csv(file_path)
#         elif filename.endswith('.xlsx'):
#             df = pd.read_excel(file_path)
#         columns = ", ".join(df.columns)
#         return f"Column names in {filename}: {columns}"
#     except Exception as e:
#         return f"Error processing file: {str(e)}"

# # Initialize the chat model
# llm = Ollama(model="openhermes", base_url="http://localhost:11434", verbose=True)


# list_files_tool = Tool(
#     func=list_files,
#     name="list_files",
#     description="this tool list files in the directory"
# )
# get_column_names_tool = Tool(
#     func=get_column_names,
#     name="get_column_names",
#     description="this tool gets the column names from the excel/csv file"
# )

# # Initialize the agent with the list of tools
# agent = initialize_agent(
#     tools=[list_files_tool, get_column_names_tool],
#     llm=llm,
#     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     max_iterations=4
# )

# # Function to format time
# def format_time(seconds: float) -> str:
#     minutes = int(seconds // 60)
#     seconds = int(seconds % 60)
#     milliseconds = int((seconds - int(seconds)) * 1000)
#     return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

# # Function to run query with timing
# def run_query(query: str) -> Tuple[str, str]:
#     start_time = time.time()
#     output = agent.run(query)
#     elapsed_time = time.time() - start_time
#     formatted_time = format_time(elapsed_time)
#     return output, formatted_time

# # Example usage
# query = "List the names of the columns in the student file in the download directory."
# output, time_taken = run_query(query)
# print(f"Output: {output}, Time taken: {time_taken}")

##############################################################################

# from langchain.tools import tool, Tool
# from langchain.agents import initialize_agent, AgentType
# from langchain.llms import Ollama
# import pandas as pd
# from typing import Tuple
# import os
# import time

# # Define a function to find Excel or CSV files in the download folder
# @tool
# def list_files(directory: str) -> str:
#     """Lists all Excel or CSV files in the given directory."""
#     try:
#         files = [f for f in os.listdir(directory) if f.endswith('.xlsx') or f.endswith('.csv')]
#         if files:
#             return "Available files: " + ", ".join(files)
#         else:
#             return "No Excel or CSV files found."
#     except Exception as e:
#         return f"Error accessing directory: {str(e)}"

# # Define a function to get column names from a given file
# @tool
# def get_column_names(filename: str, directory: str) -> str:
#     """Returns column names from the specified Excel or CSV file in the given directory."""
#     try:
#         file_path = os.path.join(directory, filename)
#         if not os.path.exists(file_path):
#             return f"File {filename} not found."
#         if filename.endswith('.csv'):
#             df = pd.read_csv(file_path)
#         elif filename.endswith('.xlsx'):
#             df = pd.read_excel(file_path)
#         columns = ", ".join(df.columns)
#         return f"Column names in {filename}: {columns}"
#     except Exception as e:
#         return f"Error reading file {filename}: {str(e)}"

# # Initialize the chat model
# llm = Ollama(model="openhermes", base_url="http://localhost:11434", verbose=True)

# # Define tools using the Tool constructor
# list_files_tool = Tool(
#     func=list_files,
#     name="list_files",
#     description="List files in the directory"
# )
# get_column_names_tool = Tool(
#     func=get_column_names,
#     name="get_column_names",
#     description="Get column names from the excel/csv file"
# )

# # Initialize the agent with the list of tools
# agent = initialize_agent(
#     tools=[list_files_tool, get_column_names_tool],
#     llm=llm,
#     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     max_iterations=4,
#     handle_parsing_errors=True
# )

# # Function to format time
# def format_time(seconds: float) -> str:
#     minutes = int(seconds // 60)
#     seconds = int(seconds % 60)
#     milliseconds = int((seconds - int(seconds)) * 1000)
#     return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

# # Function to run query with timing
# def run_query(query: str) -> Tuple[str, str]:
#     start_time = time.time()
#     output = agent.run(query)
#     elapsed_time = time.time() - start_time
#     formatted_time = format_time(elapsed_time)
#     return output, formatted_time

# # Example usage
# def generate_promt(query):
#     prompt = f"""
#     TASK: According to the below user query, 
#     if user asks for some information about the file or csv or excel, 
#     you must look for folder with name 'download' and then call the list_files tool to return 
#     the names of all files present in the folder and then act accordingly.

#     USER QUERY: {query}
#     """
#     return prompt

# query = "List the names of the columns in the student file"
# full_prompt = generate_promt(query=query)

# output, time_taken = run_query(full_prompt)
# print(f"Output: {output}, Time taken: {time_taken}")

################################################################################

from langchain.tools import tool, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
import pandas as pd
from typing import Tuple
import os
import time

# Define a function to find Excel or CSV files in the download folder
@tool
def list_files(directory: str) -> str:
    """Lists all Excel or CSV files in the given directory."""
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.xlsx') or f.endswith('.csv')]
        if files:
            return "Available files: " + ", ".join(files)
        else:
            return "No Excel or CSV files found."
    except Exception as e:
        return f"Error accessing directory: {str(e)}"

# Define a function to get column names from a given file
@tool
def get_column_names(file_path: str) -> str:
    """Returns column names from the specified Excel or CSV file."""
    try:
        if not os.path.exists(file_path):
            return f"File not found at {file_path}."
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        columns = ", ".join(df.columns)
        return f"Column names in {os.path.basename(file_path)}: {columns}"
    except Exception as e:
        return f"Error reading file {os.path.basename(file_path)}: {str(e)}"

# Initialize the chat model
llm = Ollama(model="openhermes", base_url="http://localhost:11434", verbose=True)

# Define tools using the Tool constructor
list_files_tool = Tool(
    func=list_files,
    name="list_files",
    description="List files in the directory"
)
get_column_names_tool = Tool(
    func=get_column_names,
    name="get_column_names",
    description="Get column names from the excel/csv file"
)

# Initialize the agent with the list of tools
agent = initialize_agent(
    tools=[list_files_tool, get_column_names_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=4
)

# Function to format time
def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

# Function to run query with timing
def run_query(query: str) -> Tuple[str, str]:
    start_time = time.time()
    output = agent.run(query)
    elapsed_time = time.time() - start_time
    formatted_time = format_time(elapsed_time)
    return output, formatted_time

# Example usage
def generate_prompt(query):
    prompt = f"""
    TASK: According to the below user query, 
    if user asks for some information about the file or csv or excel, 
    you must look for folder with name 'download' and then call the list_files tool to return 
    the names of all files present in the folder and then act accordingly.

    USER QUERY: {query}
    """
    return prompt

query = "List the names of the columns in the student file"
full_prompt = generate_prompt(query=query)

output, time_taken = run_query(full_prompt)
print(f"Output: {output}, Time taken: {time_taken}")
