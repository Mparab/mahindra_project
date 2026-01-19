import streamlit as st
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import tempfile
import re
from io import BytesIO

# Try to import PDF and Word libraries
try:
    import PyPDF2
    import python_docx
    PDF_AVAILABLE = True
    WORD_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    WORD_AVAILABLE = False

# --- CACHE LOGIC ---
@st.cache_data
def load_data(uploaded_file):
    """
    Universal File Loader. Supports CSV, Excel, and uses Docling for PDF/Docx.
    """
    try:
        file_name = uploaded_file.name.lower()
        
        # 1. HANDLE CSV FILES (with encoding fallbacks)
        if file_name.endswith('.csv'):
            try:
                return pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # If UTF-8 fails, try ISO-8859-1 (Common in Excel CSVs)
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        
        # 2. HANDLE EXCEL FILES
        elif file_name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
            
        # 3. HANDLE PDF / DOCX / PPTX (via Docling)
        elif file_name.endswith(('.pdf', '.docx', '.pptx', '.html')):
            with st.spinner("📄 Docling is reading document tables..."):
                # Create a temporary file because Docling needs a path
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Convert
                converter = DocumentConverter()
                result = converter.convert(tmp_path)
                
                # Extract first table found
                # Docling returns a list of tables. We take the largest one by row count.
                tables = []
                for table in result.document.tables:
                    # Convert Docling table to Pandas
                    df_table = table.export_to_dataframe()
                    tables.append(df_table)
                
                if not tables:
                    st.error("❌ No tables found in this document.")
                    return None
                
                # Return table with most rows (assumed to be the main dataset)
                largest_table = max(tables, key=lambda df: len(df))
                return largest_table

        else:
            st.error(f"❌ Unsupported file type: {file_name}")
            return None

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

# --- PAGE CONFIG ---
st.set_page_config(page_title="Data Analyst (Control Panel)", page_icon="🎛️")
st.title("🎛️ Data Analyst (Control Panel)")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("🔑 Setup")
    api_key = st.text_input("OpenRouter API Key", type="password")
    
    st.divider()
    
    st.header("⚙️ Settings")
    model_name = st.text_input("Model Name", value="meta-llama/llama-3.1-8b-instruct")
    
    # NEW: Control how many results to show!
    max_rows = st.number_input("Max Rows in Result", min_value=1, max_value=100, value=10)
    st.caption(f"The AI will only show the top {max_rows} rows to keep things fast.")
    
    # NEW: Fast Mode for sampling
    use_sample = st.checkbox("⚡ Fast Mode (Sample Data)", value=False)
    if use_sample:
        st.caption("⚡ Fast Mode: Using 20% sample for instant analysis")
    else:
        st.caption("📊 Full Mode: Using complete dataset")

# --- ERROR HANDLER ---
def validation_error_handler(error: Exception) -> str:
    """
    If AI generates a good answer but formats it weirdly, 
    this function rescues text so app doesn't crash.
    """
    error_str = str(error)
    if "Final Answer:" in error_str:
        return error_str.split("Final Answer:")[-1].strip()
    return str(error)

# --- MAIN AGENT LOGIC ---
def query_with_openrouter(df, query, key, model, limit):
    try:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0
        )

        # STRICT TEXT-ONLY PROMPT WITH ROW LIMIT
        # Create a "Cheat Sheet" with column info
        columns_info = "\n".join([f"- {col}: {dtype}" for col, dtype in df.dtypes.items()])
        
        prefix = (
            "You are a Python Data Analyst. \n"
            f"DATA SUMMARY:\n{columns_info}\n\n"  # <--- The AI now knows the schema instantly
            "Given a dataframe `df`, write code to answer the user's question.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You are a TEXT-ONLY analyst. You CANNOT generate images or plots.\n"
            "2. If the user asks for a chart, politely refuse and offer a Markdown summary table instead.\n"
            f"3. ROW LIMIT: When displaying tables, ONLY show the top {limit} rows.\n"
            "   - If the user explicitly asks for 'ALL' rows, you can ignore this limit.\n"
            "4. TABLE FORMATTING: ALWAYS format tables using proper Markdown:\n"
            "   - Use: `print(df.head(limit).to_markdown(index=False, tablefmt='pipe'))`\n"
            "   - This ensures proper table formatting with pipes (|) as column separators.\n"
            "   - NEVER use `print(df)` or `print(df.head())` - these create raw text.\n"
            "5. Always start your final response with 'Final Answer:'"
        )

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=validation_error_handler, # <--- Uses rescue logic
            prefix=prefix
        )

        return agent.run(query)

    except Exception as e:
        # Fallback rescue if the agent crashes completely
        error_msg = str(e)
        if "Final Answer:" in error_msg:
             return error_msg.split("Final Answer:")[-1].strip()
        return f"❌ Error: {error_msg}"

# --- MAIN APP UI ---
if "last_result" not in st.session_state:
    st.session_state.last_result = None

uploaded_file = st.file_uploader("Upload Document", type=['csv', 'xlsx', 'pdf', 'docx', 'pptx', 'md', 'html', 'txt'], help="Upload your document for analysis (CSV, Excel, PDF, Word, PowerPoint, etc.)")

if uploaded_file and api_key:
    # Load Data with caching
    df = load_data(uploaded_file)
    
    if df is None:
        st.error("❌ No tables found in this document. Please upload a document that contains data tables.")
        st.info("💡 Supported formats: CSV, Excel files with tables, PDFs with tables, Word documents with tables.")
        df = None
    else:
        st.write("### Data Preview")
        st.dataframe(df.head(3))

    # Apply sampling if Fast Mode is enabled
    if use_sample and df is not None:
        # Use 20% of data for speed, or max 1000 rows
        df_analysis = df.sample(frac=0.2, random_state=42)
        if len(df_analysis) > 1000:
            df_analysis = df_analysis.head(1000)
        st.info(f"⚡ Fast Mode: Analyzing {len(df_analysis)} rows (sampled from {len(df)})")
    elif df is not None:
        df_analysis = df  # Use full data
        st.info(f"📊 Full Mode: Analyzing all {len(df)} rows")

    # Query Form
    with st.form("analysis_form"):
        user_query = st.text_input("Ask a question:")
        submitted = st.form_submit_button("Analyze Data")

        if submitted and user_query:
            with st.spinner(f"Thinking (Limit: Top {max_rows} rows)..."):
                # Pass df_analysis (sampled or full) to function
                result = query_with_openrouter(df_analysis, user_query, api_key, model_name, max_rows)
                st.session_state.last_result = result

    # Display Result
    if st.session_state.last_result:
        st.success("Result:")
        st.markdown(st.session_state.last_result)

elif not api_key:
    st.warning("👈 Please enter your API Key in the sidebar to start.")