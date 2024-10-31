import streamlit as st
import pandas as pd
import warnings
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')

# Data description for better query
data_description = "This dataset contains student information such as age and gender. DO = Dropout, CF = Graduate, AC = Active."

st.set_page_config(page_title="Mobile Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title(":bar_chart: School Dashboard")
st.markdown('<style>div.block-container{padding-top: 2rem;padding-bottom: 0rem;padding-left: 5rem;padding-right: 5rem;}</style>', unsafe_allow_html=True)

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

    # File Upload
    file = st.file_uploader("Select your file", type=["csv"])

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

# Main area for processing
if "openai_api_key" in st.session_state and file:
    try:
        df = pd.read_csv(file)

        # Pass the API key to the ChatOpenAI instance
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

    except Exception as e:
        st.error(f"Error processing the file: {e}")

elif submit_button:
    st.warning("Please upload a CSV file.")
