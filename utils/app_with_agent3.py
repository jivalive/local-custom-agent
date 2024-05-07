import os
import streamlit as st
from io import StringIO
from contextlib import contextmanager, redirect_stdout
import pandas as pd
from langchain.llms import Ollama
from langchain.agents.agent_types import AgentType
import re
from langchain.tools import tool, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from PIL import Image
import os
import uuid
import time
import sys


######################################################################################################
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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

# Initialize the LLM model with an API key and specific model configuration.
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)
######################################################################################################

# Initialize the chat model
# llm = Ollama(model="openhermes", base_url="http://localhost:11434", verbose=True)

# Base directory for files
BASE_DIR = os.getcwd()  # Modify this path to the directory you are using

# Function to remove ANSI escape codes
def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Context manager to capture stdout and process it for Streamlit
@contextmanager
def st_capture(output_func):
    class DualOutput:
        def __init__(self, std_out):
            self.terminal = std_out
            self.buffer = StringIO()

        def write(self, message):
            self.terminal.write(message)
            self.buffer.write(message)

        def flush(self):
            self.terminal.flush()

    old_stdout = sys.stdout
    dual_output = DualOutput(old_stdout)
    sys.stdout = dual_output
    try:
        yield
    finally:
        sys.stdout = old_stdout
        output = dual_output.buffer.getvalue()
        clean_output = remove_ansi_codes(output)
        if clean_output:
            output_func(clean_output)

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
            print(f"creating a pie chart using {column_names}")
            plt.figure(figsize=(8, 6))
            df[column_names[0]].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
            plt.title(f'Pie Chart for {column_names[0]}')
            plt.ylabel('')
            chart_type = 'Pie Chart'
        elif len(column_names) == 2:
            print(f"creating a bar chart using {column_names}")
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
        plt.savefig("generated_charts/"+file_name)
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
    description="Creates a chart from specified columns in an Excel or CSV file and saves it as an image. If you have the columns ready, then you can create a chart using this tool. If user ask for pie chart then send any one column to 'Create_Chart' tool, if user ask for bar chart then send any two columns to 'Create_Chart' tool. If 'Create_Chart' tool returns a filename with extension png or jpg then just show the filename in final output."
)

# Initialize the agent with the list of tools
agent = initialize_agent(
    tools=[list_files_tool, get_column_names_tool, create_chart_tool],
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
        If user ask for pie chart then send any one column to 'Create_Chart' tool, if user ask for bar chart then send any two columns to 'Create_Chart' tool.
        Always construct the input for the 'Create_Chart' tool in the format 'file_path;column1,column2' nothing else no other type of format should be given.
        Remember you need to Always follow one action at a time.
        
        USER QUERY: {user_query}
        """
    else:
        return f"""
        TASK: Analyze the user's query to understand what specific information is needed.
        If it relates to file details, confirm the presence of the download folder with name download and use the 'Check_Folder_And_List_Files' tool to list all relevant files.
        If the query involves specific data within the files, use the 'Extract_Column_Names' tool to detail the structure of the specified data file,
        making sure to include the directory in the file path for example file path will be download/filename.xlsx 
        Remember you need to Always follow one action at a time.
        
        USER QUERY: {user_query}
        """

def execute_custom_df_agent_query(user_query):
    full_prompt = generate_prompt(user_query=user_query)
    output, time_taken = run_query(full_prompt)
    return {
        "Output": {output}, 
        "Time taken": {time_taken}
    }

def save_uploaded_file(uploaded_file):
    # Ensure the download directory exists
    save_path = os.path.join("download", uploaded_file.name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display success message
    st.sidebar.success("File uploaded successfully.")
    
    # Return the path to the saved file for further use
    return save_path

def extract_filename(text):
    # Regular expression to find filenames ending with .png or .jpg
    match = re.search(r'\b\w+[-\w]*(?:\.png|\.jpg)\b', text)
    return match.group(0) if match else None

# def main():
#     st.title("Chat with CSV/Excel")
    
#     uploaded_file = st.sidebar.file_uploader("Choose a file to upload", type=['csv', 'xlsx'])
    
#     if uploaded_file is not None:
#         save_uploaded_file(uploaded_file)
#         user_question = st.text_input(f"Ask any question:")

#         # # with terminal output
#         # if user_question:
#         #     full_prompt = generate_prompt(user_query=user_question)
#         #     output_placeholder = st.empty()  # Place to display the output
#         #     with st_capture(lambda x: output_placeholder.code(x)):
#         #         agent.run(full_prompt)
                
#         # only final output
#         if user_question:
#             # Display spinner and capture start time
#             with st.spinner('Processing your query...'):
#                 start_time2 = time.time()
#                 output = execute_custom_df_agent_query(user_query=user_question)
#                 end_time2 = time.time()
#                 elapsed_time2 = end_time2 - start_time2
                
#             print(f"final_result:-\n{output}\n")
            
#             output_text = next(iter(output.get('Output', [])), "")
#             print(f"output_text_result:-\n{output_text}\n")
            
#             filename = extract_filename(output_text)
#             if filename:
#                 if os.path.exists("generated_charts/"+filename):
#                     image = Image.open("generated_charts/"+filename)
#                     st.image(image, caption='Output Image')
#                 else:
#                     st.error(f"File not found: {filename}")
#             else:
#                 st.write("No image file name found in the output.")  # Fallback if no filename is extracted
#             st.info(f"Got result in {elapsed_time2:.2f} seconds")

# def main():
#     # Custom CSS to style the input bar at the bottom
#     st.markdown("""
#         <style>
#             .fixed-bottom {
#                 position: fixed;
#                 bottom: 0;
#                 left: 0;
#                 width: 100%;
#                 padding: 10px 20px;
#                 background-color: #f1f1f1;
#                 border-top: 2px solid #ccc;
#             }
#             /* Additional CSS for other elements can go here */
#         </style>
#         """, unsafe_allow_html=True)

#     st.title("Chat with CSV/Excel")

#     # Upper part of the page for outputs
#     output_container = st.container()

#     # Sidebar for uploading files
#     uploaded_file = st.sidebar.file_uploader("Choose a file to upload", type=['csv', 'xlsx'])
#     if uploaded_file is not None:
#         save_uploaded_file(uploaded_file)
    
#     # Bottom part of the page for user input
#     with st.container():
#         with st.form(key='query_form', clear_on_submit=False):
#             st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
#             user_question = st.text_input("Ask any question:", "")
#             submit_button = st.form_submit_button("Submit Query")
#             st.markdown('</div>', unsafe_allow_html=True)
    
#     if submit_button and user_question:
#         with output_container:
#             with st.spinner('Processing your query...'):
#                 start_time = time.time()
#                 output = execute_custom_df_agent_query(user_query=user_question)
#                 end_time = time.time()
#                 elapsed_time = end_time - start_time
            
#             print(f"final_result:-\n{output}\n")
            
#             output_text = next(iter(output.get('Output', [])), "")
#             print(f"output_text_result:-\n{output_text}\n")
            
#             filename = extract_filename(output_text)
#             if filename:
#                 if os.path.exists("generated_charts/"+filename):
#                     image = Image.open("generated_charts/"+filename)
#                     st.image(image, caption='Output Image')
#                 else:
#                     st.error(f"File not found: {filename}")
#             else:
#                 st.write("No image file name found in the output.")  # Fallback if no filename is extracted
#             st.info(f"Got result in {elapsed_time:.2f} seconds")


def display_message(msg):
    """Display a message in the chat, handling text and image files separately."""
    role, content = msg['role'], msg['content']
    if content.endswith('.png'):
        filepath = os.path.join("generated_charts", content)
        if os.path.exists(filepath):
            image = Image.open(filepath)
            st.image(image, caption=f'{role.capitalize()}: Output Image')
        else:
            st.error(f"File not found: {filepath}")
    else:
        st.write(f"{role.capitalize()}: {content}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with CSV/Excel", page_icon=":file_folder:")
    st.title("Chat with CSV/Excel")

    uploaded_file = st.sidebar.file_uploader("Choose a file to upload", type=['csv', 'xlsx'])

    # Initialize or retrieve chat history from session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "Welcome! Upload a file and ask any question related to the data."}]

    user_question = st.chat_input("Ask any question about your data")
    if user_question:
        st.session_state["messages"].append({"role": "user", "content": user_question})

        if uploaded_file:
            with st.spinner('Processing your query...'):
                output = execute_custom_df_agent_query(user_question)
                output_text = next(iter(output.get('Output', [])), "")
                filename = extract_filename(output_text)

                # Add message for filename or output text
                st.session_state["messages"].append({"role": "assistant", "content": filename or "No output generated."})

        else:
            st.session_state["messages"].append({"role": "assistant", "content": "Please upload a file to proceed."})

    # Always display the chat history
    for msg in st.session_state["messages"]:
        display_message(msg)
                

    
if __name__ == "__main__":
    main()
    # Example usage of the function
    # user_query = "Create a chart using marks data file"
    # user_query = "Create a sales chart using sales data file"
    # output = execute_custom_df_agent_query(user_query=user_query)
    # print(output)
