import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="School Management Dashboard", page_icon=":school:", layout="wide")
st.title(":school: School Management Dashboard")

# Data description for better query
data_description = "This dataset contains student information such as age and gender. DO = Dropout, CF = Graduate, AC = Active."

# Sidebar for API key and file upload
with st.sidebar:
    st.title("Menu:")
    
    # API Key Input
    openai_key_input = st.text_input("Input your OpenAI KEY here", type="password")  # Hide input for security
    if st.button("Set Key"):
        if openai_key_input:
            st.session_state.openai_api_key = openai_key_input
            st.success("OpenAI API key set.")
        else:
            st.error("Please enter a valid API key.")
    # Chat Input
    input_text = st.text_area("Chatbot", value="")
    submit_button = st.button("Submit")

    # Container for chat messages    
    container = st.container(border=True, height=300)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with container:
            st.markdown(f"**{message['role']}**: {message['content']}")
            

# Create two columns: one for filters and one for data
col1, col2 = st.columns(2)

# Filters in the left column
with col1:
    st.header("Filters")
    file = st.file_uploader("Upload Student Data", type=["csv"])
    selected_degree = st.selectbox("Select Degree Type", ["All", "MbC", "PhD", "MbR"])
    selected_gender = st.selectbox("Select Gender", ["All", "M", "F"])
    selected_Finance = st.selectbox("Select Finance Type", ["All", "SELF-FINANCING", "SUBSIDISED"])

# Main area for processing in the right column
with col2:
    # Apply filters if selected
    if file:
        df = pd.read_csv(file)
        if selected_degree != "All":
            df = df[df['degree_type_enrol'] == selected_degree]
        if selected_gender != "All":
            df = df[df['gender'] == selected_gender]
        if selected_Finance != "All":
            df = df[df['cdbcourse_finance_type_enrol'] == selected_Finance]

        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Visualizations
        st.subheader("Visualizations")
        vis_col1, vis_col2, vis_col3 = st.columns(3)

        # Gender Distribution
        with vis_col1:
            st.subheader("Gender Distribution")
            gender_counts = df['gender'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(3, 2))
            sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax1)
            ax1.set_title("Gender Distribution", color='black')
            ax1.set_xlabel("Gender", color='black')
            ax1.set_ylabel("Count", color='black')
            plt.setp(ax1.xaxis.get_majorticklabels(), color='black')
            plt.setp(ax1.yaxis.get_majorticklabels(), color='black')
            st.pyplot(fig1)

        # Degree Type Distribution
        with vis_col2:
            st.subheader("Degree Type Distribution")
            degree_counts = df['degree_type_enrol'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(3, 2))
            sns.barplot(x=degree_counts.index, y=degree_counts.values, ax=ax2)
            ax2.set_title("Degree Type Distribution", color='black')
            ax2.set_xlabel("Degree Type", color='black')
            ax2.set_ylabel("Count", color='black')
            plt.setp(ax2.xaxis.get_majorticklabels(), color='black')
            plt.setp(ax2.yaxis.get_majorticklabels(), color='black')
            st.pyplot(fig2)

        # Financial Status Distribution
        with vis_col3:
            st.subheader("Financial Status Distribution")
            financial_counts = df['cdbcourse_finance_type_enrol'].value_counts()
            fig3, ax3 = plt.subplots(figsize=(3, 2))
            sns.barplot(x=financial_counts.index, y=financial_counts.values, ax=ax3)
            ax3.set_title("Financial Status Distribution", color='black')
            ax3.set_xlabel("Financial Status", color='black')
            ax3.set_ylabel("Count", color='black')
            plt.setp(ax3.xaxis.get_majorticklabels(), color='black')
            plt.setp(ax3.yaxis.get_majorticklabels(), color='black')
            st.pyplot(fig3)

if "openai_api_key" in st.session_state and file:
        
    # Chat functionality
    llm = ChatOpenAI(api_key=st.session_state.openai_api_key, model="gpt-3.5-turbo", temperature=0.5)
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=False, agent_type=AgentType.OPENAI_FUNCTIONS, allow_dangerous_code=True
    )
    
    
    if submit_button and input_text:
        enhanced_input = data_description + " " + input_text
        result = agent.invoke(enhanced_input)
    
        # Display user message
        st.session_state.messages.append({"role": "user", "content": input_text})
    
        response = result.get("output", "No response returned.")
        st.session_state.messages.append({"role": "assistant", "content": response})
    
        for message in st.session_state.messages:
            with container:
                st.markdown(f"**{message['role']}**: {message['content']}")

            
elif submit_button:
    with container:
        st.warning("Please upload a CSV file.")
