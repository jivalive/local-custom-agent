import os
import streamlit as st
from io import StringIO
from contextlib import contextmanager, redirect_stdout
import pandas as pd
from langchain.llms import Ollama
from langchain_experimental.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import re

# Function to remove ANSI escape codes
def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Context manager to capture stdout and process it for Streamlit
@contextmanager
def st_capture(output_func):
    with StringIO() as buffer, redirect_stdout(buffer):
        yield
        output = buffer.getvalue()
        clean_output = remove_ansi_codes(output)
        if clean_output:
            output_func(clean_output)

def main():
    st.set_page_config(layout="wide", page_title="Jivalive ChatBot", page_icon='')
    st.title("Jivalive ChatBot")
    st.header("Ask your CSV 📈")

    llm = Ollama(model="openhermes", base_url="http://localhost:11434", verbose=True)
    
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Determine the file type and read it accordingly
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if file_extension.lower() == '.xlsx':
            # sheet_name = st.sidebar.text_input("Enter Excel sheet name (default is the first sheet):", value="")
            sheet_name = 0
            if not sheet_name:
                sheet_name = 0  # default to the first sheet if no name provided
            data = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
        elif file_extension.lower() == '.csv':
            data = pd.read_csv(uploaded_file)

        # Common agent creation from the read data
        agent = create_pandas_dataframe_agent(
            llm,
            data,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )

        user_question = st.text_input(f"Ask a question about your {file_extension.upper()} file:", key=file_extension)

        if user_question:
            output_placeholder = st.empty()  # Place to display the output
            with st_capture(lambda x: output_placeholder.code(x)):
                agent.run(user_question)

if __name__ == "__main__":
    main()
