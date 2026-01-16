import streamlit as st
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. SETUP MATPLOTLIB (Prevents crashing) ---
matplotlib.use('Agg')

st.set_page_config(page_title="Data Analyst (OpenRouter)", page_icon="🦄")
st.title("🦄 Data Analyst (via OpenRouter)")

# Sidebar
with st.sidebar:
    st.header("🔑 Setup")
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.markdown("[Get Key Here](https://openrouter.ai/keys)")
    
    # Allow changing models easily if one is down
    model_name = st.text_input("Model Name", value="deepseek/deepseek-r1-0528:free")
    st.caption("Alternatives: `meta-llama/llama-3.1-8b-instruct:free`, `google/gemini-2.0-flash-lite-preview-02-05:free`")

# --- 2. ERROR HANDLER ---
def validation_error_handler(error: Exception) -> str:
    response_str = str(error)
    if "Final Answer:" in response_str:
        return response_str.split("Final Answer:")[-1].strip()
    return f"**Raw Answer:** {response_str}"

# --- 3. THE SOLVER ---
def query_with_openrouter(df, query, key, model):
    # Clear old plots
    if os.path.exists("plot.png"):
        os.remove("plot.png")
    
    try:
        # CONNECT TO OPENROUTER
        llm = ChatOpenAI(
            model=model,
            openai_api_key=key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0
        )

        prefix = (
            "You are a Python Data Analyst. \n"
            "Given a dataframe `df`, write code to answer the user's question.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. NEVER answer based on the `df.head()` sample. Use the entire dataset.\n"
            "2. If the user asks to PLOT/CHART:\n"
            "   - Create a plot using matplotlib.\n"
            "   - Save it as 'plot.png'.\n"
            "   - In the Final Answer say: 'I have saved the plot.'\n"
            "3. FORMATTING:\n" 
            "   - If the answer is a DataFrame, use `print(df.to_markdown())`.\n"
            "   - Do NOT output internal <thinking> tags or reasoning logs in the final answer.\n"
            "   - Do NOT include 'Thought:', 'Action:', or 'Observation:' in your final response.\n"
            "   - ONLY provide the final answer with 'Final Answer:' prefix.\n"
            "4. Always start your final response with 'Final Answer:'"
        )

        # We pass "handle_parsing_errors=True" to let the agent try to recover automatically
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True, 
            prefix=prefix
        )

        response = agent.run(query)
        
        # Check for plot
        if os.path.exists("plot.png"):
            st.image("plot.png", caption="Generated Visualization")
            return f"✅ Visualization Generated.\n\n{response}"
            
        return response

    except Exception as e:
        # --- THE FIX: RESCUE THE ANSWER FROM THE ERROR ---
        error_msg = str(e)
        if "Final Answer:" in error_msg:
            # The agent actually finished but the parser complained.
            # We extract the text AFTER "Final Answer:" manually.
            recovered_answer = error_msg.split("Final Answer:")[-1].strip()
            
            # Sometimes the error includes trailing weirdness, so we clean it up
            if "For troubleshooting" in recovered_answer:
                recovered_answer = recovered_answer.split("For troubleshooting")[0].strip()
            
            return recovered_answer
            
        return f"❌ Error: {error_msg}"

# --- 4. MAIN LOGIC ---
if "last_result" not in st.session_state:
    st.session_state.last_result = None

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file and api_key:
    # Smart Loader
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        
    st.write("### Data Preview")
    st.dataframe(df.head(3))

    with st.form("analysis_form"):
        user_query = st.text_input("Ask a question:")
        submitted = st.form_submit_button("Analyze Data")

        if submitted and user_query:
            with st.spinner(f"Thinking with {model_name}..."):
                result = query_with_openrouter(df, user_query, api_key, model_name)
                st.session_state.last_result = result

    if st.session_state.last_result:
        st.success("Result:")
        st.markdown(st.session_state.last_result)

elif not api_key:
    st.warning("👈 Please paste your OpenRouter API Key in the sidebar.")
