from langchain.tools import tool, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import os
import uuid
import time

# Base directory for files
BASE_DIR = os.getcwd()  # Modify this path to the directory you are using

@tool
def summarize_data(input_string: str) -> str:
    """
    This tool takes a string data  and summarize the data given in a crisp manner."
    """
    prompt = f"Now we have the input data as {input_string}. \n we should summarize it as per the user request and return the output in crisp manner."
    return prompt

@tool
def create_chart(input_string: str) -> str:
    """
    Parses the input string to get the file path and column names, reads the specified Excel or CSV file,
    generates a chart for the provided column names, saves the chart as an image file in the current directory,
    and returns the file name.
    Input format: 'file_path;column1,column2'
    The type of chart depends on the number of columns provided:
    - One column: Pie chart.
    - Two columns: Bar chart.
    - More than two columns: Not currently supported.
    """
    try:
        parts = input_string.split(';')
        # Strip any leading/trailing whitespace from the file path
        file_path = os.path.normpath(os.path.join(BASE_DIR, parts[0].strip()))  # Normalize and construct the file path

        # Create a list of column names, stripping whitespace from each
        column_names = [name.strip() for name in parts[1].split(',')]

        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        if not all(col in df.columns for col in column_names):
            missing_cols = [col for col in column_names if col not in df.columns]
            return f"Columns {', '.join(missing_cols)} do not exist in the file."

        if len(column_names) == 1:
            plt.figure(figsize=(8, 6))
            df[column_names[0]].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
            plt.title(f'Pie Chart for {column_names[0]}')
            plt.ylabel('')
            chart_type = 'Pie Chart'
        elif len(column_names) == 2:
            plt.figure(figsize=(10, 8))
            df.groupby(column_names[0])[column_names[1]].sum().plot(kind='bar')
            plt.title(f'Bar Chart showing {column_names[1]} for each {column_names[0]}')
            plt.xlabel(column_names[0])
            plt.ylabel(column_names[1])
            chart_type = 'Bar Chart'
        else:
            return "More than two columns are not currently supported for chart generation."

        # Save the plot with a dynamically generated name based on the chart type
        file_name = f"{chart_type.lower().replace(' ', '_')}_{uuid.uuid4()}.png"
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
    full_directory_path = os.path.normpath(os.path.join(BASE_DIR, directory))  # Ensure full path is constructed with BASE_DIR
    try:
        if not os.path.isdir(full_directory_path):  # Check if the path is a directory
            return f"Error: '{full_directory_path}' is not a directory. If you are trying to get Column details of the file then use 'Extract_Column_Names' tool instead with the filepath, example:- download/filename.file_extension"

        files = [f for f in os.listdir(full_directory_path) if f.endswith('.xlsx') or f.endswith('.csv')]
        if files:
            return "I found the following files: " + ", ".join(files) + " . Now, if you are trying to get Column details of the suitable file related to user query then use 'Extract_Column_Names' tool using the filepath, example:- download/filename.file_extension"
        else:
            return "No Excel or CSV files found in the directory."
    except Exception as e:
        return f"Error while accessing the directory: {str(e)}"

@tool
def get_column_names(file_path: str) -> str:
    """
    Opens the specified Excel or CSV file and returns a string listing all column names.
    If the file does not exist, it returns a message stating that the file was not found.
    If an error occurs during file reading (e.g., file is corrupted or unreadable), it provides an error message detailing the problem.
    """
    full_file_path = os.path.normpath(os.path.join(BASE_DIR, file_path))
    try:
        if not os.path.exists(full_file_path):
            return f"The file at {full_file_path} could not be found."

        df = pd.read_csv(full_file_path) if full_file_path.endswith('.csv') else pd.read_excel(full_file_path)
        return f"The columns in {os.path.basename(file_path)} are: {', '.join(df.columns)}. Now you have the column names, if user have asked to create chart then you can call Create_Chart tool to create a chart and save it as a file."
    except Exception as e:
        return f"An error occurred while reading {os.path.basename(file_path)}: {str(e)}"

# Initialize the chat model
llm = Ollama(model="openhermes", base_url="http://localhost:11434", verbose=True)

# Define tools with more descriptive names and descriptions
list_files_tool = Tool(
    func=list_files,
    name="Check_Folder_And_List_Files",
    description="Lists all Excel and CSV files in a specified directory to help users locate their data files. It will be given the filepath such download/filename.xlsx remember that it will always have download directory attached to filepath."
)
get_column_names_tool = Tool(
    func=get_column_names,
    name="Extract_Column_Names",
    description="Extracts and lists column names from a specified Excel or CSV file to aid users in understanding the structure of their data. It will be given the filepath such download/filename.xlsx remember that it will always have download directory attached to filepath."
)
create_chart_tool = Tool(
    func=create_chart,
    name="Create_Chart",
    description="Creates a chart from specified columns in an Excel or CSV file and saves it as an image. If you have the columns ready, then you can create a chart using this tool."
)
summarize_data_tool = Tool(
    func=summarize_data,
    name="Summarize_Data",
    description="This tool take a string input data  and summarizes the given data in a crisp manner."
)

# Initialize the agent with the list of tools
agent = initialize_agent(
    tools=[list_files_tool, get_column_names_tool, create_chart_tool, summarize_data_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=9,
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

# Function to generate prompts
def generate_prompt(user_query):
    """
    Generate a detailed prompt for the LLM based on the user's query.
    The prompt instructs the LLM to analyze the query, check for file and data details,
    potentially list files and column names, and prepare to create a chart if requested.
    Instructs that the create_chart tool takes an input string in the format 'file_path;column1,column2'.
    """
    if 'chart' in user_query.lower():
        return f"""
        TASK: The user has requested a chart. First, verify the presence of the download folder with name download.
        Use the 'Check_Folder_And_List_Files' tool to list all relevant files, files will be in directory named download. Once the file is identified,
        use the 'Extract_Column_Names' tool to retrieve and list all column names from the specified file.
        Analyze the columns to determine which can be used to create the chart effectively.
        Based on the columns found and the data within, construct the input for the 'Create_Chart' tool in the format 'file_path;column1,column2'
        and use it to generate the chart. Ensure the columns selected are appropriate for the type of chart requested by the user.
        If user has asked to summarize then you should have some data to give the Summarize_Data tool, so make sure that you have some data, if you don't have the data then you should follow a path to get the data first.
        Remember that Summarize_Data tool will only take a string input  and returns a string output, so use the tool accordingly.

        USER QUERY: {user_query}
        """
    else:
        return f"""
        TASK: Analyze the user's query to understand what specific information is needed.
        If it relates to file details, confirm the presence of the download folder with name download and use the 'Check_Folder_And_List_Files' tool to list all relevant files.
        If the query involves specific data within the files, use the 'Extract_Column_Names' tool to detail the structure of the specified data file,
        making sure to include the directory in the file path for example file path will be download/filename.xlsx 
        If user has asked to summarize then you should have some data to give the Summarize_Data tool, so make sure that you have some data, if you don't have the data then you should follow a path to get the data first.
        Remember that Summarize_Data tool will only take a string input  and returns a string output, so use the tool accordingly.
        
        USER QUERY: {user_query}
        """

def execute_custom_df_agent_query(user_query):
    full_prompt = generate_prompt(user_query=user_query)
    output, time_taken = run_query(full_prompt)
    return {
        "Output": {output}, 
        "Time taken": {time_taken}
    }

# Example usage of the function
# user_query = "Create a chart using marks data file"
# user_query = "Create a sales bar chart using sales data file"
# user_query = "summarize the sales data"
# output = execute_custom_df_agent_query(user_query=user_query)
# print(output)

df_data = """
camera_id,monotainer_id,lane_name,sorting_timestamp,key_str,ifstaged,staged_timestamp,ifprocessed,processed_timestamp,ifuntagged
C001,511759,chatham,2023-06-13T08:30:00,511759#2023-06-13T08:30:00,TRUE,2023-06-13T21:45:00,TRUE,2023-06-13T21:52:00,FALSE
C001,101FA0,chatham,2023-06-13T09:30:00,101FA0#2023-06-13T09:30:00,TRUE,2023-06-13T21:46:01,TRUE,2023-06-13T21:56:01,FALSE
C001,23040B,chatham,2023-06-13T10:30:00,23040B#2023-06-13T10:30:00,TRUE,2023-06-13T21:47:02,TRUE,2023-06-13T21:57:02,FALSE
C001,24A712,chatham,2023-06-13T11:30:00,24A712#2023-06-13T11:30:00,TRUE,2023-06-13T22:15:03,TRUE,2023-06-13T22:20:03,FALSE
C001,32AFB8,chatham,2023-06-13T12:30:00,32AFB8#2023-06-13T12:30:00,TRUE,2023-06-13T22:20:04,TRUE,2023-06-13T22:30:04,FALSE
C001,46B101,chatham,2023-06-13T13:30:00,46B101#2023-06-13T13:30:00,TRUE,2023-06-13T22:22:05,TRUE,2023-06-13T22:32:05,FALSE
C001,55287F,chatham,2023-06-13T14:30:00,55287F#2023-06-13T14:30:00,TRUE,2023-06-13T22:25:06,TRUE,2023-06-13T22:35:06,FALSE
C001,5D3D5F,chatham,2023-06-13T15:30:00,5D3D5F#2023-06-13T15:30:00,TRUE,2023-06-13T22:30:07,TRUE,2023-06-13T22:40:07,FALSE
C001,711D5A,chatham,2023-06-13T16:30:00,711D5A#2023-06-13T16:30:00,TRUE,2023-06-13T22:32:08,TRUE,2023-06-13T22:42:08,FALSE
C001,99BC75,chatham,2023-06-13T17:30:00,99BC75#2023-06-13T17:30:00,TRUE,2023-06-13T22:35:09,TRUE,2023-06-13T22:45:09,FALSE
C001,9D4C80,chatham,2023-06-13T18:30:00,9D4C80#2023-06-13T18:30:00,TRUE,2023-06-13T22:40:10,TRUE,2023-06-13T22:48:10,FALSE
C001,A19AA6,chatham,2023-06-13T19:30:00,A19AA6#2023-06-13T19:30:00,TRUE,2023-06-13T22:45:11,TRUE,2023-06-13T22:53:11,FALSE
C001,A807AF,chatham,2023-06-13T20:30:00,A807AF#2023-06-13T20:30:00,TRUE,2023-06-13T22:47:12,TRUE,2023-06-13T22:55:12,FALSE
C001,AA7F82,chatham,2023-06-13T21:30:00,AA7F82#2023-06-13T21:30:00,TRUE,2023-06-13T22:50:13,TRUE,2023-06-13T22:58:13,FALSE
C001,DA9726,chatham,2023-06-13T21:35:00,DA9726#2023-06-13T21:35:00,TRUE,2023-06-13T22:55:14,TRUE,2023-06-13T23:05:14,FALSE
C001,521759,chatham,2023-06-13T08:30:03,521759#2023-06-13T08:30:03,TRUE,2023-06-13T21:45:00,TRUE,2023-06-13T21:52:00,FALSE
C001,202FA0,chatham,2023-06-13T09:30:04,202FA0#2023-06-13T09:30:04,TRUE,2023-06-13T21:46:01,TRUE,2023-06-13T21:56:01,FALSE
C001,23040B,chatham,2023-06-13T10:30:02,23040B#2023-06-13T10:30:02,TRUE,2023-06-13T21:47:02,TRUE,2023-06-13T21:57:02,FALSE
C001,24A712,chatham,2023-06-13T11:30:05,24A712#2023-06-13T11:30:05,TRUE,2023-06-13T22:15:03,TRUE,2023-06-13T22:20:03,FALSE
C001,32AFC9,chatham,2023-06-13T12:30:04,32AFC9#2023-06-13T12:30:04,TRUE,2023-06-13T22:20:04,TRUE,2023-06-13T22:30:04,FALSE
C001,46B202,chatham,2023-06-13T13:30:06,46B202#2023-06-13T13:30:06,TRUE,2023-06-13T22:22:05,TRUE,2023-06-13T22:32:05,FALSE
C001,55288B,chatham,2023-06-13T14:30:07,55288B#2023-06-13T14:30:07,TRUE,2023-06-13T22:25:06,TRUE,2023-06-13T22:35:06,FALSE
C001,5D3D55,chatham,2023-06-13T15:30:08,5D3D55#2023-06-13T15:30:08,TRUE,2023-06-13T22:30:07,TRUE,2023-06-13T22:40:07,FALSE
C001,711D7B,chatham,2023-06-13T16:30:09,711D7B#2023-06-13T16:30:09,TRUE,2023-06-13T22:32:08,TRUE,2023-06-13T22:42:08,FALSE
C001,99BC96,chatham,2023-06-13T17:30:05,99BC96#2023-06-13T17:30:05,TRUE,2023-06-13T22:35:09,TRUE,2023-06-13T22:45:09,FALSE
C001,9D4C90,chatham,2023-06-13T18:30:06,9D4C90#2023-06-13T18:30:06,TRUE,2023-06-13T22:40:10,TRUE,2023-06-13T22:48:10,FALSE
C001,A19AB9,chatham,2023-06-13T19:30:07,A19AB9#2023-06-13T19:30:07,TRUE,2023-06-13T22:45:11,TRUE,2023-06-13T22:53:11,FALSE
C001,A807CD,chatham,2023-06-13T20:30:02,A807CD#2023-06-13T20:30:02,TRUE,2023-06-13T22:47:12,TRUE,2023-06-13T22:55:12,FALSE
C001,AA7F90,chatham,2023-06-13T21:30:01,AA7F90#2023-06-13T21:30:01,TRUE,2023-06-13T22:50:13,TRUE,2023-06-13T22:58:13,FALSE
C001,DA9756,chatham,2023-06-13T21:35:06,DA9756#2023-06-13T21:35:06,TRUE,2023-06-13T22:55:14,TRUE,2023-06-13T23:05:14,FALSE
C002,511759,Toronto,2023-06-13T08:30:00,511759#2023-06-13T08:30:00,TRUE,2023-06-13T21:45:00,TRUE,2023-06-13T21:52:00,FALSE
C002,101FA0,Toronto,2023-06-13T09:30:00,101FA0#2023-06-13T09:30:00,TRUE,2023-06-13T21:46:01,TRUE,2023-06-13T21:56:01,FALSE
C002,23040B,Toronto,2023-06-13T10:30:00,23040B#2023-06-13T10:30:00,TRUE,2023-06-13T21:47:02,TRUE,2023-06-13T21:57:02,FALSE
C002,24A712,Toronto,2023-06-13T11:30:00,24A712#2023-06-13T11:30:00,TRUE,2023-06-13T22:15:03,TRUE,2023-06-13T22:20:03,FALSE
C002,32AFB8,Toronto,2023-06-13T12:30:00,32AFB8#2023-06-13T12:30:00,TRUE,2023-06-13T22:20:04,TRUE,2023-06-13T22:30:04,FALSE
C002,46B101,Toronto,2023-06-13T13:30:00,46B101#2023-06-13T13:30:00,TRUE,2023-06-13T22:22:05,TRUE,2023-06-13T22:32:05,FALSE
C002,55287F,Toronto,2023-06-13T14:30:00,55287F#2023-06-13T14:30:00,TRUE,2023-06-13T22:25:06,TRUE,2023-06-13T22:35:06,FALSE
C002,5D3D5F,Toronto,2023-06-13T15:30:00,5D3D5F#2023-06-13T15:30:00,TRUE,2023-06-13T22:30:07,TRUE,2023-06-13T22:40:07,FALSE
C002,711D5A,Toronto,2023-06-13T16:30:00,711D5A#2023-06-13T16:30:00,TRUE,2023-06-13T22:32:08,TRUE,2023-06-13T22:42:08,FALSE
C002,99BC75,Toronto,2023-06-13T17:30:00,99BC75#2023-06-13T17:30:00,TRUE,2023-06-13T22:35:09,TRUE,2023-06-13T22:45:09,FALSE
C002,9D4C80,Toronto,2023-06-13T18:30:00,9D4C80#2023-06-13T18:30:00,TRUE,2023-06-13T22:40:10,TRUE,2023-06-13T22:48:10,FALSE
C002,A19AA6,Toronto,2023-06-13T19:30:00,A19AA6#2023-06-13T19:30:00,TRUE,2023-06-13T22:45:11,TRUE,2023-06-13T22:53:11,FALSE
C002,A807AF,Toronto,2023-06-13T20:30:00,A807AF#2023-06-13T20:30:00,TRUE,2023-06-13T22:47:12,TRUE,2023-06-13T22:55:12,FALSE
C002,AA7F82,Toronto,2023-06-13T21:30:00,AA7F82#2023-06-13T21:30:00,TRUE,2023-06-13T22:50:13,TRUE,2023-06-13T22:58:13,FALSE
C002,DA9726,Toronto,2023-06-13T21:35:00,DA9726#2023-06-13T21:35:00,TRUE,2023-06-13T22:55:14,TRUE,2023-06-13T23:05:14,FALSE
C002,521759,Toronto,2023-06-13T08:30:03,521759#2023-06-13T08:30:03,TRUE,2023-06-13T21:45:00,TRUE,2023-06-13T21:52:00,FALSE
C002,202FA0,Toronto,2023-06-13T09:30:04,202FA0#2023-06-13T09:30:04,TRUE,2023-06-13T21:46:01,TRUE,2023-06-13T21:56:01,FALSE
C002,23040B,Toronto,2023-06-13T10:30:02,23040B#2023-06-13T10:30:02,TRUE,2023-06-13T21:47:02,TRUE,2023-06-13T21:57:02,FALSE
C002,24A712,Toronto,2023-06-13T11:30:05,24A712#2023-06-13T11:30:05,TRUE,2023-06-13T22:15:03,TRUE,2023-06-13T22:20:03,FALSE
C002,32AFC9,Toronto,2023-06-13T12:30:04,32AFC9#2023-06-13T12:30:04,TRUE,2023-06-13T22:20:04,TRUE,2023-06-13T22:30:04,FALSE
C002,46B202,Toronto,2023-06-13T13:30:06,46B202#2023-06-13T13:30:06,TRUE,2023-06-13T22:22:05,TRUE,2023-06-13T22:32:05,FALSE
C002,55288B,Toronto,2023-06-13T14:30:07,55288B#2023-06-13T14:30:07,TRUE,2023-06-13T22:25:06,TRUE,2023-06-13T22:35:06,FALSE
C002,5D3D55,Toronto,2023-06-13T15:30:08,5D3D55#2023-06-13T15:30:08,TRUE,2023-06-13T22:30:07,TRUE,2023-06-13T22:40:07,FALSE
C002,711D7B,Toronto,2023-06-13T16:30:09,711D7B#2023-06-13T16:30:09,TRUE,2023-06-13T22:32:08,TRUE,2023-06-13T22:42:08,FALSE
C002,99BC96,Toronto,2023-06-13T17:30:05,99BC96#2023-06-13T17:30:05,TRUE,2023-06-13T22:35:09,TRUE,2023-06-13T22:45:09,FALSE
C002,9D4C90,Toronto,2023-06-13T18:30:06,9D4C90#2023-06-13T18:30:06,TRUE,2023-06-13T22:40:10,TRUE,2023-06-13T22:48:10,FALSE
C002,A19AB9,Toronto,2023-06-13T19:30:07,A19AB9#2023-06-13T19:30:07,TRUE,2023-06-13T22:45:11,TRUE,2023-06-13T22:53:11,FALSE
C002,A807CD,Toronto,2023-06-13T20:30:02,A807CD#2023-06-13T20:30:02,TRUE,2023-06-13T22:47:12,TRUE,2023-06-13T22:55:12,FALSE
C002,AA7F90,Toronto,2023-06-13T21:30:01,AA7F90#2023-06-13T21:30:01,TRUE,2023-06-13T22:50:13,TRUE,2023-06-13T22:58:13,FALSE
"""
user_query = f"please summarize the below data: \n\n\n\n {df_data}"
output = execute_custom_df_agent_query(user_query=user_query)
print(output)