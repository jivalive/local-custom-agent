from langchain.tools import tool, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import os
import uuid
import time

@tool
def create_chart(file_path: str, column_names: list) -> str:
    """
    Reads the specified Excel or CSV file, generates a chart for the provided column names,
    saves the chart as an image file in the current directory, and returns the file name.
    The type of chart depends on the number of columns provided:
    - One column: Pie chart.
    - Two columns: Bar chart.
    - More than two columns: Not currently supported.
    """
    try:
        # Load data from file
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Check if all columns exist
        if not all(col in df.columns for col in column_names):
            missing_cols = [col for col in column_names if col not in df.columns]
            return f"Columns {', '.join(missing_cols)} do not exist in the file."

        # Plotting based on the number of columns
        if len(column_names) == 1:
            # Generate pie chart for a single categorical column
            plt.figure(figsize=(8, 6))
            df[column_names[0]].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
            plt.title(f'Pie Chart for {column_names[0]}')
            plt.ylabel('')
        elif len(column_names) == 2:
            # Generate bar chart for two columns
            plt.figure(figsize=(10, 8))
            df.groupby(column_names[0])[column_names[1]].sum().plot(kind='bar')
            plt.title(f'Bar Chart showing {column_names[1]} for each {column_names[0]}')
            plt.xlabel(column_names[0])
            plt.ylabel(column_names[1])
        else:
            return "More than two columns are not currently supported for chart generation."

        # Save the plot to a file
        file_name = f"chart_{uuid.uuid4()}.png"
        plt.savefig(file_name)
        plt.close()
        return file_name

    except Exception as e:
        return f"Failed to create chart: {str(e)}"

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
create_chart_tool = Tool(
    func=create_chart,
    name="Create_Chart",
    description="Creates a chart from specified columns in an Excel or CSV file and saves it as an image."
)

# Initialize the agent with the list of tools
agent = initialize_agent(
    tools=[list_files_tool, get_column_names_tool, create_chart_tool],
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
def generate_prompt(user_query):
    """
    Generate a detailed prompt for the LLM based on the user's query.
    The prompt directs the LLM to analyze the query, check for file and data details,
    potentially list files and column names, and prepare to create a chart if requested.
    """
    if 'chart' in user_query.lower():
        # If the query includes a request for a chart
        return f"""
        TASK: The user has requested a chart. First, verify the presence of the 'download' folder.
        Use the 'Find_Excel_and_CSV_Files' tool to list all relevant files. 
        Once the file is identified, use the 'Extract_Column_Names' tool to retrieve and list all column names from the specified file.
        Analyze the columns to determine which can be used to create the chart effectively.
        Based on the columns found and the data within, determine if a chart can be created and then use the 'Create_Chart' tool to generate the chart.
        Ensure the columns selected are appropriate for the type of chart requested by the user.

        USER QUERY: {user_query}
        """
    else:
        # For other types of queries
        return f"""
        TASK: Analyze the user's query to understand what specific information is needed.
        If it relates to file details, confirm the presence of the 'download' folder and use the 'Find_Excel_and_CSV_Files' tool to list all relevant files.
        If the query involves specific data within the files, use the 'Extract_Column_Names' tool to detail the structure of the specified data file,
        making sure to include the directory in the file path. 

        USER QUERY: {user_query}
        """


# query = "List the names of the columns in the student file"
# full_prompt = generate_prompt(query=query)
# output, time_taken = run_query(full_prompt)
# print(f"Output: {output}, Time taken: {time_taken}")


# Example usage of the function
query = "Create a sales pie chart showing in sales data file"
full_prompt = generate_prompt(user_query=query)
output, time_taken = run_query(full_prompt)
print(f"Output: {output}, Time taken: {time_taken}")