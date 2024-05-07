from langchain.tools import tool, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
import pandas as pd
from typing import Tuple
import os
import time

@tool
def list_files(directory: str) -> str:
    """
    Scans the specified directory and returns a formatted string listing all Excel (.xlsx) and CSV (.csv) files found.
    If no such files are found, it returns a message indicating that no files were found.
    If an error occurs (e.g., directory not found), it returns an error message specifying the issue.
    """
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.xlsx') or f.endswith('.csv')]
        if files:
            return "I found the following files: " + ", ".join(files)
        else:
            return "I could not find any Excel or CSV files in the specified directory."
    except Exception as e:
        return f"Error while accessing the directory: {str(e)}"

@tool
def get_column_names(file_path: str) -> str:
    """
    Opens the specified Excel or CSV file and returns a string listing all column names.
    If the file does not exist, it returns a message stating that the file was not found.
    If an error occurs during file reading (e.g., file is corrupted or unreadable), it provides an error message detailing the problem.
    """
    try:
        if not os.path.exists(file_path):
            return f"The file at {file_path} could not be found."
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        columns = ", ".join(df.columns)
        return f"The columns in {os.path.basename(file_path)} are: {columns}"
    except Exception as e:
        return f"An error occurred while reading {os.path.basename(file_path)}: {str(e)}"

# Initialize the chat model
llm = Ollama(model="openhermes", base_url="http://localhost:11434", verbose=True)

# Define tools with more descriptive names and descriptions
list_files_tool = Tool(
    func=list_files,
    name="Find_Excel_and_CSV_Files",
    description="Lists all Excel and CSV files in a specified directory to help users locate their data files."
)
get_column_names_tool = Tool(
    func=get_column_names,
    name="Extract_Column_Names",
    description="Extracts and lists column names from a specified Excel or CSV file to aid users in understanding the structure of their data."
)

# Initialize the agent with the list of tools
agent = initialize_agent(
    tools=[list_files_tool, get_column_names_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=4,
    handle_parsing_errors=True
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

# Generate detailed prompts for better understanding
def generate_prompt(query):
    prompt = f"""
    TASK: Based on the query provided, analyze if the user is requesting data from an Excel or CSV file.
    If so, confirm the presence of the 'download' folder, then use the 'Find_Excel_and_CSV_Files' tool to list all relevant files.
    Depending on the files found, use the 'Extract_Column_Names' tool to detail the structure of the specified data file, making sure to include the directory in the file path.

    USER QUERY: {query}
    ACTION: Verify 'download' directory, list files, then construct full path for file to extract column names.
    """
    return prompt

query = "List the names of the columns in the student file"
full_prompt = generate_prompt(query=query)

output, time_taken = run_query(full_prompt)
print(f"Output: {output}, Time taken: {time_taken}")
