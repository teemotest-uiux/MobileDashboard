import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import requests
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_debug.log"),  # Save logs to a file
        logging.StreamHandler(sys.stdout)        # Print logs in CMD
    ]
)

# Function to validate OpenAI API key
def validate_openai_key(api_key):
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        # If the response status code is 200, the key is valid
        if response.status_code == 200:
            return True
        else:
            logging.warning(f"API Key validation failed with status code {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        logging.error(f"Error while validating API key: {e}")
        return False

# Page configuration
st.set_page_config(page_title="School Management Dashboard", page_icon=":school:", layout="wide")
st.title(":school: School Management Dashboard")

# Data description for better query
data_description = """
Check the dataset if it's PG or UG, PG for postgraduates and UG for undergraduates. There is no combined dataset of PG and UG
First check the first row of the dataset to determine if it's PG or UG data.

The PG Dataset contains multiple student information with the following columns:
- hash_id: the hashed identity number of a student.
- admission_yr: the year that the student was admitted to school.
- gender: the student’s gender, F for female M for Male.
- nationality_group: the student’s nationalities where SC is Singaporean student, IS is international student and PR is a permanent Resident of Singapore student.
- pg_prog_type_enrol: the course the student was pursuing when he was admitted, C for Coursework, R for Research.
- degree_type_enrol: the type of degree he is pursing when he was admitted, PhD for doctoral degrees, MbC for MbC, MbR for Mbr.
- cdbcourse_finance_type_enrol: Whether the student was SUBSIDISED or SELF-FINANCING when paying his/her school fees.
- cdbcourse_term_per_yr_enrol: how many terms are there per year. 2 for 2 semesters in a year, 3 for 3 semesters in a year.
- pg_fullpart_enrol: FT is full time student, PT is part time student. 
- pg_rs_start_dt: the date and time of the start of their studies.
- candidature_end_dt: the estimated date and time of their end of studies.
- pg_min_candidature_dt: the minimum date and time they would need to graduate.
- current_student_status: DO is Dropout, CF is Graduated, AC is Active.
- pg_course_code: The Program that the student is pursuing.
- school_name_short: the school name in the university.
- degree code: the code for the student's degree.

The UG Dataset contains multiple student information with the following columns:
- admission: The year the student was admitted into the school.
- gender: the gender of the student (M/F)
- nationality: SC is Singapore citizen, IS is International student, SPR is Singapore permanent resident.
- ug_student_type: FT is Full time, PT is Part time.
- ug_total_candidature: is the amount of candidature time the student took.
- ug_normal_candidature: is the normal amount of candidature time for that course
- drop_out_dt: the drop-out date and time for that student, Null would mean that the student did not drop out.
- gr_graduated_yr: the year the student graduated, Null would mean the student is still actively studying or have dropped out of school
- current_student_status: DO is Dropout, CF is Graduated, AC is Active.
- ug_course_code: the program the student is studying in.
- school_name_short: the school that the student is in.

you are to help with data retrieval and data analytics, you do not need to answer if it's a UG or PG dataset, You are expected to give a value if it is a calculation question. check if the condition is missing value first, if it is use another column for estimation. 
"""

# Sidebar for API key and file upload
with st.sidebar:
    st.title("Menu:")
    
    # API Key Input
    openai_key_input = st.text_input("Input your OpenAI KEY here", type="password")  # Hide input for security
    if st.button("Set Key"):
        if openai_key_input:
            openai_key_input = openai_key_input.strip()  # Remove any leading/trailing spaces
            
            # Validate the OpenAI API key
            if validate_openai_key(openai_key_input):
                st.session_state.openai_api_key = openai_key_input
                st.success("OpenAI API key set.")
                logging.info("OpenAI API key set.")
            else:
                st.error("Invalid API key. Please check and try again.")
                logging.warning("Invalid API key entered.")
        else:
            st.error("Please enter a valid API key.")
            logging.warning("No API key entered.")
    
    # Chat Input
    input_text = st.text_area("Chatbot", value="")
    submit_button = st.button("Submit")

    # Container for chat messages    
    container = st.container(border=True, height=300)
    if "messages" not in st.session_state:
        st.session_state.messages = []          


col1, col2 = st.columns([2, 3]) 

with col1:
    # File Upload Section
    file = st.file_uploader("Upload Student Data", type=["csv", "xlsx"])
    # After file is uploaded, display filters and data processing
    if file:
        # Read file after upload
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        
        logging.info("File uploaded successfully.")
        logging.info(f"Dataset columns: {list(df.columns)}")
        
        # Check if the dataset is UG or PG
        if 'ug_student_type' in df.columns:  # UG dataset
            dataset_type = "UG"
        elif 'degree_type_enrol' in df.columns:  # PG dataset
            dataset_type = "PG"
        else:
            dataset_type = "Unknown"
            st.warning("The dataset type is unknown. Please upload a valid UG or PG dataset.")

        # Filters Section
        if dataset_type != "Unknown":
            st.header("Filters")

            if dataset_type == "PG":
                selected_degree = st.selectbox("Select Degree Type", ["All", "MbC", "PhD", "MbR"])
            else:
                selected_degree = "All"  
            
            selected_gender = st.selectbox("Select Gender", ["All", "M", "F"])
            
           
            if dataset_type == "PG":
                selected_Finance = st.selectbox("Select Finance Type", ["All", "SELF-FINANCING", "SUBSIDISED"])
            else:
                selected_Finance = "All"  

            # Apply filters to dataframe
            if selected_degree != "All":
                if dataset_type == "PG":
                    df = df[df['degree_type_enrol'] == selected_degree]
                elif dataset_type == "UG":
                    df = df[df['ug_course_code'] == selected_degree]
            
            if selected_gender != "All":
                df = df[df['gender'] == selected_gender]
            
            if selected_Finance != "All" and dataset_type == "PG":
                df = df[df['cdbcourse_finance_type_enrol'] == selected_Finance]

            logging.info(f"Filters Applied - Degree: {selected_degree}, Gender: {selected_gender}, Finance: {selected_Finance}")
        
# Right column: Data preview and charts
with col2:
    # Display data preview
    if file and dataset_type != "Unknown":
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Plotting charts based on dataset type
        if dataset_type == "PG":
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))  
            sns.countplot(data=df, x="gender", ax=axs[0])
            axs[0].set_title("Gender Distribution (PG)")

            sns.countplot(data=df, x="degree_type_enrol", ax=axs[1])
            axs[1].set_title("Degree Type Distribution (PG)")

            sns.countplot(data=df, x="cdbcourse_finance_type_enrol", ax=axs[2])
            axs[2].set_title("Finance Type Distribution (PG)")
            st.pyplot(fig)

        elif dataset_type == "UG":
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))  
            sns.countplot(data=df, x="gender", ax=axs[0])
            axs[0].set_title("Gender Distribution (UG)")

            sns.countplot(data=df, x="ug_student_type", ax=axs[1])
            axs[1].set_title("Student Type Distribution (UG)")

            sns.countplot(data=df, x="current_student_status", ax=axs[2])
            axs[2].set_title("Current Status Distribution (UG)")
            st.pyplot(fig)

# Ensure both OpenAI key and file are uploaded before using the AI agent
if "openai_api_key" in st.session_state and file is not None:
    llm = ChatOpenAI(api_key=st.session_state.openai_api_key, model="gpt-3.5-turbo", temperature=0.5)
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS, allow_dangerous_code=True
    )
    
    # Callback handler for better tracking
    callback_handler = StdOutCallbackHandler()
    
    if submit_button and input_text:
        enhanced_input = data_description + " " + input_text + """Please reason through the problem step by step., if there's missing data reuse the original dataset and estimate with other data in the dataset"""
        logging.info(f"Sending query to AI: {enhanced_input}")
        result = agent.invoke(enhanced_input)
    
        st.session_state.messages.append({"role": "user", "content": input_text})
    
        response = result.get("output", "No response returned.")
        st.session_state.messages.append({"role": "assistant", "content": response})
        logging.info(f"AI Response: {response}")

        with container:
            for message in st.session_state.messages:        
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
elif submit_button:
    with container:
        if "openai_api_key" not in st.session_state:
            st.warning("Please set your OpenAI API key before submitting a query.")
            logging.warning("User attempted to submit a query without setting the OpenAI API key.")
        elif file is None:
            st.warning("Please upload a CSV or Excel file before submitting a query.")
            logging.warning("User attempted to submit a query without uploading a file.")
