import os
import tempfile
from typing import List, Any, Dict, MutableMapping, cast
from dataclasses import dataclass
import sys
import traceback
import streamlit as st
from collections.abc import Mapping
from langchain_core.documents import Document
import pandas as pd
# Ensure you have installed it: pip install pandas

def _show_startup_error(exc: Exception):
    try:
        st.set_page_config(page_title="Startup Error")
        st.title("Application startup error")
        st.error("An exception occurred during app startup:")
        st.exception(exc)
        st.write("Full traceback:")
        st.text(traceback.format_exc())
    except Exception:
        # Fallback to stderr so deployment logs contain the traceback
        print("Startup error (fallback):", exc, file=sys.stderr)
        traceback.print_exc()

def main():
    # App initialization and UI
    st.set_page_config(page_title="Domain-Agnostic Intelligent Data Query System", layout="centered")
    st.title("🔍 Domain-Agnostic Intelligent Data Query System")
    st.write("Upload structured data files and query them with intelligent SQL filtering")
    st.write("Supported: CSV, Excel, SQL dump files")

    try:
        import pandas as pd  # local import so app can still start if pandas fails
        PANDAS_AVAILABLE = True
    except Exception as e:
        pd = None  # type: ignore
        PANDAS_AVAILABLE = False
        st.warning(f"pandas import failed: {e}")
        st.error("Cannot process structured data without pandas. Please install pandas.")
        return

    # Optional libs (safe imports)
    pdfplumber = None
    DocxDocument = None
    docling = None

    try:
        import pdfplumber as _pdfplumber  # type: ignore
        pdfplumber = _pdfplumber
    except Exception:
        pdfplumber = None

    try:
        from docx import Document as _DocxDocument  # type: ignore
        DocxDocument = _DocxDocument
    except Exception:
        DocxDocument = None

    try:
        import docling as _docling  # type: ignore
        docling = _docling
    except Exception:
        docling = None

    def extract_with_docling_or_fallback(path: str):
        """Try docling if present, otherwise simple pdf/docx/text extraction."""
        rows: List[Dict[str, Any]] = []

        # Docling first (if available)
        if docling is not None:
            try:
                chunks = docling.extract(path)
                for c in chunks:
                    rows.append({
                        "chunk_text": c.get("text", "") if isinstance(c, dict) else str(getattr(c, "text", "")),
                        "source": c.get("source", os.path.basename(path)) if isinstance(c, dict) else os.path.basename(path),
                        "page": c.get("page") if isinstance(c, dict) else getattr(c, "page", None),
                        "table_id": c.get("table_id") if isinstance(c, dict) else getattr(c, "table_id", None),
                    })
                if PANDAS_AVAILABLE:
                    # local import so static analyzers don't report pd may be None
                    import pandas as _pd  # type: ignore
                    return _pd.DataFrame(rows)
                return rows
            except Exception:
                pass

        # Fallback extraction
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf" and pdfplumber is not None:
            try:
                with pdfplumber.open(path) as pdf:
                    for i, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            rows.append({"chunk_text": text, "source": os.path.basename(path), "page": i, "table_id": None})
                        for tidx, table in enumerate(page.extract_tables() or [], start=1):
                            tbl_text = "\n".join([" | ".join([str(cell) if cell is not None else "" for cell in r]) for r in table])
                            rows.append({"chunk_text": tbl_text, "source": os.path.basename(path), "page": i, "table_id": f"table_{tidx}"})
            except Exception:
                pass
        elif ext in (".docx", ".doc") and DocxDocument is not None:
            try:
                doc = DocxDocument(path)
                for p in getattr(doc, "paragraphs", []):
                    text = p.text.strip()
                    if text:
                        rows.append({"chunk_text": text, "source": os.path.basename(path), "page": None, "table_id": None})
                for tidx, tbl in enumerate(getattr(doc, "tables", []) or [], start=1):
                    tbl_rows = []
                    for r in tbl.rows:
                        cells = [c.text.strip() for c in r.cells]
                        tbl_rows.append(" | ".join(cells))
                    rows.append({"chunk_text": "\n".join(tbl_rows), "source": os.path.basename(path), "page": None, "table_id": f"table_{tidx}"})
            except Exception:
                pass
        else:
            # text fallback
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                    if txt.strip():
                        rows.append({"chunk_text": txt, "source": os.path.basename(path), "page": None, "table_id": None})
            except Exception:
                pass

        if PANDAS_AVAILABLE and pd is not None:
            try:
                return pd.DataFrame(rows)
            except Exception:
                return rows
        return rows

    def build_documents_from_extracted_df(df) -> List[Document]:
        docs: List[Document] = []
        if PANDAS_AVAILABLE and pd is not None and isinstance(df, pd.DataFrame):
            for chunk_idx, (_, row) in enumerate(df.iterrows()):
                page_content = str(row.get("chunk_text", "")) if hasattr(row, "get") else str(row.get("text", ""))
                metadata = {
                    "source": row.get("source") if hasattr(row, "get") else None,
                    "page": row.get("page") if hasattr(row, "get") else None,
                    "table_id": row.get("table_id") if hasattr(row, "get") else None,
                    "chunk_index": int(chunk_idx)
                }
                docs.append(Document(page_content=page_content, metadata=metadata))
        else:
            rows = df or []
            for idx, row in enumerate(rows):
                if isinstance(row, dict):
                    content = row.get("chunk_text") or row.get("text") or " ".join([str(v) for v in row.values()])
                    metadata = {
                        "source": row.get("source"),
                        "page": row.get("page"),
                        "table_id": row.get("table_id"),
                        "chunk_index": int(idx)
                    }
                else:
                    content = str(row)
                    metadata = {"source": None, "page": None, "table_id": None, "chunk_index": int(idx)}
                docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def build_documents_from_df(df) -> List[Document]:
        docs: List[Document] = []
        if PANDAS_AVAILABLE and pd is not None and isinstance(df, pd.DataFrame):
            # Use enumerate to obtain a stable integer row index (position)
            for pos, (_, row) in enumerate(df.iterrows()):
                if "text" in df.columns:
                    content = str(row.get("text", ""))
                else:
                    content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                metadata = {"source": None, "row_index": int(pos)}
                docs.append(Document(page_content=content, metadata=metadata))
        else:
            rows = df or []
            cols = list(rows[0].keys()) if (rows and isinstance(rows[0], dict)) else []
            for idx, row in enumerate(rows):
                if isinstance(row, dict):
                    if "text" in row:
                        content = str(row.get("text", ""))
                    else:
                        content = " | ".join([f"{col}: {row.get(col)}" for col in cols])
                else:
                    content = str(row)
                metadata = {"source": None, "row_index": int(idx)}
                docs.append(Document(page_content=content, metadata=metadata))
        return docs

    # --- UI: file uploader and processing ---
    uploaded_file = st.file_uploader(
        "Upload Structured Data File", 
        type=["csv", "xlsx", "xls", "sql"],
        help="Supported formats: CSV, Excel, SQL dump"
    )

    df = None
    documents: List[Document] = []

    # Use a typed view of session_state to satisfy the type checker
    session: MutableMapping[str, Any] = cast(MutableMapping[str, Any], st.session_state)

    if uploaded_file is not None:
        fname = uploaded_file.name
        ext = os.path.splitext(fname)[1].lower()

        # Process structured files only
        if ext in (".csv", ".xlsx", ".xls"):
            if not PANDAS_AVAILABLE or pd is None:
                st.error("Cannot read structured files because pandas failed to import. Please install pandas.")
                df = None
            else:
                try:
                    if fname.lower().endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif fname.lower().endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file, engine="openpyxl")
                    else:
                        df = pd.read_excel(uploaded_file, engine="xlrd")
                    
                    if df is not None:
                        st.success(f"✅ Successfully loaded {len(df)} rows from {fname}")
                        st.write("Data Preview:")
                        st.dataframe(df.head(10))
                        st.write(f"Columns: {', '.join(df.columns.tolist())}")
                        
                        # STEP 2: Schema Detection (Automatic)
                        schema = {
                            col: str(dtype) 
                            for col, dtype in zip(df.columns, df.dtypes)
                        }
                        
                        st.subheader("📋 Detected Schema")
                        for col, dtype in schema.items():
                            st.write(f"**{col}**: {dtype}")
                        
                        # Store schema in session state
                        session[f"schema_{fname}"] = schema
                        
                        # STEP 3: Store Data in SQL (Generic Table)
                        try:
                            import sqlite3
                            conn = sqlite3.connect(":memory:")  # In-memory database
                            
                            # Create generic table with the data
                            df.to_sql("data_table", conn, if_exists="replace", index=False)
                            
                            # Store connection in session state
                            session[f"conn_{fname}"] = conn
                            session[f"table_name_{fname}"] = "data_table"
                            
                            st.success("✅ Data stored in SQL database for efficient querying")
                            
                        except Exception as e:
                            st.warning(f"SQL storage failed: {e}. Using pandas filtering instead.")
                            conn = None
                        
                        # STEP 8: Vector Embeddings + FAISS
                        try:
                            # Create documents from dataframe for semantic search
                            documents = build_documents_from_df(df)
                            
                            # Create embeddings
                            from langchain_community.embeddings import HuggingFaceEmbeddings
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
                            )
                            
                            # Create FAISS vector store
                            from langchain_community.vectorstores import FAISS
                            vectorstore = FAISS.from_documents(documents, embeddings)
                            
                            # Store vectorstore in session state
                            session[f"vectorstore_{fname}"] = vectorstore
                            
                            st.success("✅ Vector embeddings created for semantic search")
                            
                        except Exception as e:
                            st.warning(f"Vector embeddings failed: {e}. SQL-only search available.")
                            vectorstore = None
                        
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    df = None
        elif ext == ".sql":
            st.info("SQL dump files will be supported in the next version")
            df = None
        else:
            st.error("Unsupported file type. Please upload CSV, Excel, or SQL files.")
            df = None

    # Store dataframe, SQL connection, and vectorstore in session state
    df_key = f"df_{uploaded_file.name}" if uploaded_file is not None else "df_in_memory"
    conn_key = f"conn_{uploaded_file.name}" if uploaded_file is not None else "conn_in_memory"
    table_key = f"table_name_{uploaded_file.name}" if uploaded_file is not None else "table_name_in_memory"
    vectorstore_key = f"vectorstore_{uploaded_file.name}" if uploaded_file is not None else "vectorstore_in_memory"
    
    if df is not None:
        session[df_key] = df
    elif df_key in session:
        df = session[df_key]
    
    # Get SQL connection, table name, and vectorstore from session state
    conn = session.get(conn_key, None)
    table_name = session.get(table_key, "data_table")
    vectorstore = session.get(vectorstore_key, None)

    query = st.text_input(
        "Enter your query (Natural Language Examples:)",
        placeholder="Hospital: 'Female patients above 60 with stroke' | Manufacturing: 'Parts with failure rate above 5% in Plant B' | Sales: 'Top 10 customers by revenue last quarter'"
    )

    # Process query with Robust LangChain Agent
    if query and df is not None:
        st.subheader("🔍 Query Results")
        
        try:
            # Get schema from session state
            schema_key = f"schema_{uploaded_file.name}" if uploaded_file is not None else "schema_in_memory"
            schema = session.get(schema_key, {})
            
            # STEP 1: Create enhanced system prompt with dynamic schema
            system_prompt = create_enhanced_system_prompt(df, schema)
            
            # STEP 2: Initialize LangChain Pandas Agent with proper imports
            agent = create_robust_langchain_agent(df, system_prompt)
            
            # STEP 3: Execute query with comprehensive error handling
            result = execute_robust_agent_query(agent, query, df)
            
            if result["success"]:
                filtered_df = result["data"]
                st.success(f"🤖 Agent processed query successfully")
                st.write(f"📊 Found {len(filtered_df)} matching records")
                st.dataframe(filtered_df)
                
                # Show summary statistics
                st.subheader("📊 Summary Statistics")
                for col in filtered_df.select_dtypes(include=['number']).columns:
                    st.write(f"**{col}:** Min={filtered_df[col].min():.2f}, Max={filtered_df[col].max():.2f}, Avg={filtered_df[col].mean():.2f}")
            else:
                st.error(f"❌ Agent failed: {result['error']}")
                st.info("🔄 Attempting fallback filtering...")
                
                # Fallback to manual filtering
                fallback_df = apply_manual_fallback_filtering(df, query, schema)
                if len(fallback_df) < len(df):
                    st.success("✅ Fallback filtering successful")
                    st.dataframe(fallback_df)
                else:
                    st.warning("⚠️ Both agent and fallback failed. Please check your query syntax.")
                
        except Exception as e:
            st.error(f"❌ Critical error in query processing: {str(e)}")
            st.error("Please check your query syntax and try again.")
            st.info("💡 Tip: Use exact column names and clear conditions (e.g., 'Age > 60', 'Gender = Female')")

def create_robust_langchain_agent(df, system_prompt):
    """Create LangChain agent with proper import handling and context"""
    try:
        # Try to import LangChain with OpenAI
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain_openai import ChatOpenAI
        
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.warning("⚠️ OpenAI API key not found. Using custom agent.")
            return create_custom_pandas_agent(df, system_prompt)
        
        # Create LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Cost-effective and fast
            temperature=0,
            api_key=api_key
        )
        
        # Create agent with enhanced configuration
        agent = create_pandas_dataframe_agent(
            df,
            llm=llm,
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=3,
            max_execution_time=30,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            system_prompt=system_prompt,
            prefix="""You are working with a dataframe named `df`. pandas is imported as pd. """
        )
        
        return agent
        
    except ImportError as e:
        st.warning(f"⚠️ LangChain not available: {e}. Using custom agent.")
        return create_custom_pandas_agent(df, system_prompt)
    except Exception as e:
        st.error(f"❌ Failed to create agent: {e}")
        return create_custom_pandas_agent(df, system_prompt)

def execute_robust_agent_query(agent, query, df):
    """Execute agent query with proper context and error handling"""
    try:
        # Create execution context with required libraries
        execution_context = {
            'pd': pd,  # Critical: Pass pandas library
            'df': df,
            'np': __import__('numpy'),  # Import numpy if available
            'datetime': __import__('datetime')  # Import datetime if needed
        }
        
        # Execute the query
        result = agent.run(query)
        
        # Validate the result
        if isinstance(result, pd.DataFrame):
            return {"success": True, "data": result, "query": query}
        elif hasattr(result, '__iter__'):
            try:
                # Convert to DataFrame if possible
                # Fix: Wrap result in list to handle scalar dictionary error
                if isinstance(result, dict):
                    result_df = pd.DataFrame([result]) if not isinstance(result, pd.DataFrame) else result
                else:
                    result_df = pd.DataFrame(result) if not isinstance(result, pd.DataFrame) else result
                return {"success": True, "data": result_df, "query": query}
            except Exception as e:
                return {"success": False, "error": f"Result conversion failed: {str(e)}", "query": query}
        else:
            return {"success": False, "error": "Agent returned invalid data format", "query": query}
                
    except Exception as e:
        return {"success": False, "error": f"Agent execution failed: {str(e)}", "query": query}

def create_enhanced_system_prompt(df, schema):
    """Create enhanced system prompt with categorical value awareness"""
    columns_info = []
    categorical_values = {}
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        columns_info.append(f"  - {col} ({dtype})")
        
        # Extract unique values for categorical columns
        if 'object' in dtype or df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) <= 10:  # Only show if manageable
                categorical_values[col] = unique_vals
    
    columns_list = ", ".join(df.columns.tolist())
    
    # Build categorical values section
    cat_values_section = ""
    if categorical_values:
        cat_values_section = "\nCategorical Values (for exact matching):\n"
        for col, values in categorical_values.items():
            cat_values_section += f"  - {col}: {', '.join(map(str, values[:5]))}\n"
    
    system_prompt = f"""You are a data analyst assistant with access to a pandas DataFrame.

**Table Structure:**
Table Name: df
Columns:
{chr(10).join(columns_info)}

Available columns: {columns_list}

**Data Types:**
- Numeric columns (int, float): Use mathematical operators (>, <, >=, <=, ==)
- Categorical columns (object): Use exact string matching or .contains()

{cat_values_section}

**Instructions:**
1. Use ONLY the column names listed above - do not guess or hallucinate column names
2. For numeric comparisons: "above 60" means > 60, "below 30" means < 30
3. For categorical matching: Use exact values from the list above, or .contains() for partial matches
4. Handle user typos with partial matching (e.g., 'femal' should match 'Female')
5. Return executable pandas code that filters the DataFrame

**Examples:**
- Age > 60: df[df['Age'] > 60]
- Gender = Female: df[df['Gender'] == 'Female']
- Department contains Sales: df[df['Department'].str.contains('Sales', case=False)]
- Top 10 by Salary: df.nlargest(10, 'Salary')

**Error Handling:**
- If column doesn't exist, return an error message
- If data type doesn't support operation, return an error message
- Always return valid pandas DataFrame operations

Execute the user's query and return the filtered DataFrame."""
    return system_prompt

def create_langchain_pandas_agent(df, system_prompt):
    """Create LangChain Pandas Dataframe Agent with proper configuration"""
    try:
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain.llms import OpenAI
        
        # Create agent with our system prompt
        agent = create_pandas_dataframe_agent(
            df,
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=3,
            max_execution_time=30,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            system_prompt=system_prompt
        )
        
        return agent
        
    except ImportError:
        st.warning("⚠️ LangChain not available. Using custom agent implementation.")
        return create_custom_pandas_agent(df, system_prompt)
    except Exception as e:
        st.error(f"❌ Failed to create agent: {e}")
        return create_custom_pandas_agent(df, system_prompt)

def create_custom_pandas_agent(df, system_prompt):
    """Custom agent implementation when LangChain is not available"""
    class CustomPandasAgent:
        def __init__(self, df, system_prompt):
            self.df = df
            self.system_prompt = system_prompt
            self.columns = df.columns.tolist()
            self.dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        def run(self, query):
            try:
                # Simple but robust query parsing
                filtered_df = self.df.copy()
                query_lower = query.lower()
                
                for col in self.columns:
                    if col.lower() in query_lower:
                        dtype = self.dtypes[col]
                        
                        if 'int' in dtype or 'float' in dtype:
                            # Numeric operations
                            if '>' in query or 'above' in query_lower or 'greater than' in query_lower:
                                match = re.search(rf'{col.lower()}\s*(?:>|above|greater than|over)\s*([\d.]+)', query_lower)
                                if match:
                                    value = float(match.group(1))
                                    filtered_df = filtered_df[filtered_df[col] > value]
                            
                            elif '<' in query or 'below' in query_lower or 'less than' in query_lower:
                                match = re.search(rf'{col.lower()}\s*(?:<|below|less than|under)\s*([\d.]+)', query_lower)
                                if match:
                                    value = float(match.group(1))
                                    filtered_df = filtered_df[filtered_df[col] < value]
                        
                        elif 'object' in dtype:
                            # String operations with typo tolerance
                            if 'female' in query_lower and 'gender' in col.lower():
                                filtered_df = filtered_df[filtered_df[col].str.contains('female', case=False, na=False)]
                            elif 'male' in query_lower and 'gender' in col.lower():
                                filtered_df = filtered_df[filtered_df[col].str.contains('male', case=False, na=False)]
                
                return safe_dataframe_conversion(filtered_df)
                
            except Exception as e:
                return safe_dataframe_conversion({"error": str(e), "query": query})
    
    return CustomPandasAgent(df, system_prompt)

def safe_dataframe_conversion(data):
    """
    Helper to prevent 'scalar values' error when creating DataFrames
    """
    try:
        # Case 1: Data is already a DataFrame
        if isinstance(data, pd.DataFrame):
            return data
            
        # Case 2: Data is a list of dicts (Standard)
        if isinstance(data, list):
            return pd.DataFrame(data)
            
        # Case 3: Data is a single dict (The cause of your error!)
        if isinstance(data, dict):
            # We must wrap it in brackets []
            return pd.DataFrame([data]) 
            
        # Case 4: Data is a simple string/number (e.g., count)
        return pd.DataFrame({"Result": [data]})
        
    except Exception as e:
        return pd.DataFrame({"Error": [f"Conversion failed: {str(e)}"]})

def execute_agent_query(agent, query, df):
    """Execute agent query with comprehensive error handling"""
    try:
        # Execute the query
        result = agent.run(query)
        
        # Validate the result
        if isinstance(result, pd.DataFrame):
            return {"success": True, "data": result, "query": query}
        else:
            # Try to extract DataFrame from result
            if hasattr(result, '__iter__'):
                try:
                    # Convert to DataFrame if possible
                    # Fix: Wrap result in list to handle scalar dictionary error
                    if isinstance(result, dict):
                        result_df = pd.DataFrame([result]) if not isinstance(result, pd.DataFrame) else result
                    else:
                        result_df = pd.DataFrame(result) if not isinstance(result, pd.DataFrame) else result
                    return {"success": True, "data": result_df, "query": query}
                except:
                    return {"success": False, "error": "Agent returned invalid data format", "query": query}
            else:
                return {"success": False, "error": "Agent returned non-DataFrame result", "query": query}
                
    except Exception as e:
        return {"success": False, "error": f"Agent execution failed: {str(e)}", "query": query}

def apply_manual_fallback_filtering(df, query, schema):
    """Manual fallback filtering when agent fails"""
    import re
    
    filtered_df = df.copy()
    query_lower = query.lower()
    
    # Simple but reliable filtering logic
    for col in df.columns:
        col_lower = col.lower()
        dtype = str(df[col].dtype)
        
        if col_lower in query_lower:
            if 'int' in dtype or 'float' in dtype:
                # Numeric filtering
                if '>' in query or 'above' in query_lower:
                    match = re.search(rf'{col_lower}\s*(?:>|above|greater than|over)\s*([\d.]+)', query_lower)
                    if match:
                        value = float(match.group(1))
                        filtered_df = filtered_df[filtered_df[col] > value]
                
                elif '<' in query or 'below' in query_lower:
                    match = re.search(rf'{col_lower}\s*(?:<|below|less than|under)\s*([\d.]+)', query_lower)
                    if match:
                        value = float(match.group(1))
                        filtered_df = filtered_df[filtered_df[col] < value]
            
            elif 'object' in dtype:
                # String filtering with typo tolerance
                if 'female' in query_lower and 'gender' in col_lower():
                    filtered_df = filtered_df[filtered_df[col].str.contains('female', case=False, na=False)]
                elif 'male' in query_lower and 'gender' in col_lower():
                    filtered_df = filtered_df[filtered_df[col].str.contains('male', case=False, na=False)]
    
    return filtered_df

def classify_query_intent(query):
    """Router: Classify user intent to determine processing method"""
    query_lower = query.lower()
    
    # Case 1: Filtering/Math - Hard Data
    filtering_keywords = [
        '>', '<', '=', 'above', 'below', 'over', 'under', 'greater than', 'less than',
        'older than', 'younger than', 'higher than', 'lower than',
        'top', 'bottom', 'first', 'last', 'limit', 'maximum', 'minimum',
        'between', 'range', 'equals', 'is', 'are'
    ]
    
    # Check for numeric filtering patterns
    import re
    numeric_patterns = [
        r'\d+\s*(?:>|<|=)',
        r'(?:above|below|over|under|greater|less)\s+\d+',
        r'(?:older|younger|higher|lower)\s+than\s+\d+',
        r'top\s+\d+',
        r'between\s+\d+\s+and\s+\d+'
    ]
    
    has_numeric_filter = any(keyword in query_lower for keyword in filtering_keywords) or \
                       any(re.search(pattern, query_lower) for pattern in numeric_patterns)
    
    # Case 2: Semantic Search - Soft Data
    semantic_keywords = [
        'felt', 'happy', 'sad', 'satisfied', 'angry', 'complained', 'experienced',
        'symptoms', 'description', 'notes', 'feedback', 'comments', 'summary',
        'similar', 'like', 'related', 'about', 'describe'
    ]
    
    has_semantic_need = any(keyword in query_lower for keyword in semantic_keywords)
    
    # Case 3: Hybrid - Both hard and soft
    if has_numeric_filter and has_semantic_need:
        return "hybrid"
    elif has_numeric_filter:
        return "filtering_math"
    elif has_semantic_need:
        return "semantic_search"
    else:
        return "general"

def create_dynamic_system_prompt(df, schema):
    """Create dynamic system prompt with actual data structure"""
    columns_info = []
    for col, dtype in schema.items():
        columns_info.append(f"  - {col} ({dtype})")
    
    columns_list = ", ".join(df.columns.tolist())
    
    system_prompt = f"""
You are a data analyst assistant. You have access to a pandas DataFrame with the following structure:

Table Name: df
Columns:
{chr(10).join(columns_info)}

Available columns: {columns_list}

Data types are automatically detected. Use this information to:
1. Write precise filtering conditions for numeric columns (use >, <, ==, >=, <=)
2. Use exact string matching for categorical columns
3. Handle mathematical operations correctly (e.g., "above 60" means > 60)
4. Do not guess column names - use only the columns listed above

Examples:
- For "Age above 60": df[df['Age'] > 60]
- For "Gender is Female": df[df['Gender'] == 'Female']
- For "Sales > 500": df[df['Sales'] > 500]
- For "Top 10 by Revenue": df.nlargest(10, 'Revenue')

Always return valid pandas code that can be executed.
"""
    return system_prompt

def apply_pandas_agent_filtering(df, query, system_prompt):
    """Apply filtering using Pandas Agent approach"""
    try:
        # Try to use LangChain Pandas Agent if available
        try:
            from langchain.agents import create_pandas_dataframe_agent
            from langchain.llms import OpenAI
            
            # Create agent with dynamic schema awareness
            agent = create_pandas_dataframe_agent(
                llm=None,  # We'll implement our own logic since we don't have OpenAI key
                df=df,
                verbose=False,
                allow_dangerous_code=True
            )
            
            # For now, implement our own agent logic
            return apply_agent_logic(df, query, system_prompt)
            
        except ImportError:
            # Fallback to our own agent implementation
            return apply_agent_logic(df, query, system_prompt)
            
    except Exception as e:
        st.warning(f"Agent approach failed: {e}. Using fallback filtering.")
        return apply_agent_logic(df, query, system_prompt)

def apply_agent_logic(df, query, system_prompt):
    """Custom agent logic that understands data structure"""
    import re
    import pandas as pd
    
    filtered_df = df.copy()
    query_lower = query.lower()
    
    # Agent knows the exact columns and types from system prompt
    for col in df.columns:
        col_lower = col.lower()
        dtype = str(df[col].dtype)
        
        if col_lower in query_lower:
            # Numeric columns - Agent understands mathematical operations
            if 'int' in dtype or 'float' in dtype:
                # Greater than
                gt_match = re.search(rf'{col_lower}\s*(?:>|greater than|above|over|older than|higher than)\s*([\d.]+)', query_lower)
                if gt_match:
                    value = float(gt_match.group(1))
                    filtered_df = filtered_df[filtered_df[col] > value]
                    st.info(f"🤖 Agent: {col} > {value}")
                
                # Less than
                lt_match = re.search(rf'{col_lower}\s*(?:<|less than|below|under|younger than|lower than)\s*([\d.]+)', query_lower)
                if lt_match:
                    value = float(lt_match.group(1))
                    filtered_df = filtered_df[filtered_df[col] < value]
                    st.info(f"🤖 Agent: {col} < {value}")
                
                # Equals
                eq_match = re.search(rf'{col_lower}\s*(?:=|equals?|is)\s*([\d.]+)', query_lower)
                if eq_match:
                    value = float(eq_match.group(1))
                    filtered_df = filtered_df[filtered_df[col] == value]
                    st.info(f"🤖 Agent: {col} = {value}")
            
            # String columns - Agent uses exact matching
            elif 'object' in dtype:
                # Gender
                if 'female' in query_lower and 'gender' in col_lower:
                    filtered_df = filtered_df[filtered_df[col] == 'Female']
                    st.info(f"🤖 Agent: {col} = Female")
                elif 'male' in query_lower and 'gender' in col_lower:
                    filtered_df = filtered_df[filtered_df[col] == 'Male']
                    st.info(f"🤖 Agent: {col} = Male")
                
                # General string matching
                else:
                    # Look for exact values in query that match column data
                    unique_values = [str(val).lower() for val in df[col].unique() if pd.notna(val)]
                    for val in unique_values:
                        if val in query_lower:
                            filtered_df = filtered_df[filtered_df[col].str.contains(val, case=False, na=False)]
                            st.info(f"🤖 Agent: {col} contains '{val}'")
                            break
    
    return filtered_df

def extract_filters_with_llm(query, schema):
    """Extract filters into JSON format using LLM-like parsing"""
    import re
    import json
    
    filters = {}
    query_lower = query.lower()
    
    # Enhanced pattern matching for filter extraction
    for col, dtype in schema.items():
        col_lower = col.lower()
        
        if col_lower in query_lower:
            # Numeric filters
            if 'int' in dtype or 'float' in dtype:
                # Greater than patterns
                gt_patterns = [
                    rf'{col_lower}\s*(?:>|greater than|above|over)\s*([\d.]+)',
                    rf'(?:age|stay|length|duration)\s*(?:>|above|over|greater than)\s*([\d.]+)'
                ]
                for pattern in gt_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        filters[f"{col_lower}_min"] = float(match.group(1))
                        break
                
                # Less than patterns
                lt_patterns = [
                    rf'{col_lower}\s*(?:<|less than|below|under)\s*([\d.]+)',
                    rf'(?:age|stay|length|duration)\s*(?:<|below|under|less than)\s*([\d.]+)'
                ]
                for pattern in lt_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        filters[f"{col_lower}_max"] = float(match.group(1))
                        break
                
                # Equal patterns
                eq_pattern = rf'{col_lower}\s*(?:=|equals?|is)\s*([\d.]+)'
                match = re.search(eq_pattern, query_lower)
                if match:
                    filters[col_lower] = float(match.group(1))
            
            # String filters
            elif 'object' in dtype or 'str' in dtype:
                # Gender patterns
                if 'female' in query_lower and 'gender' in col_lower:
                    filters['gender'] = 'Female'
                elif 'male' in query_lower and 'gender' in col_lower:
                    filters['gender'] = 'Male'
                
                # Medical conditions
                elif 'stroke' in query_lower and any(term in col_lower for term in ["diagnosis", "condition", "disease"]):
                    filters['diagnosis_contains'] = 'stroke'
                
                # Plant/Location patterns
                elif 'plant' in query_lower and any(term in col_lower for term in ["plant", "location", "facility"]):
                    plant_match = re.search(r'plant\s+(\w+)', query_lower)
                    if plant_match:
                        filters['plant'] = plant_match.group(1).capitalize()
                
                # General string matching
                else:
                    # Extract exact values from query
                    words = query_lower.split()
                    for word in words:
                        if word.capitalize() in [str(v).capitalize() for v in df[col].unique() if pd.notna(v)]:
                            filters[col_lower] = word.capitalize()
                            break
    
    return filters

def extract_filters_aggressive(query, schema):
    """More aggressive filter extraction for retry scenarios"""
    import re
    
    filters = {}
    query_lower = query.lower()
    
    # Look for any numbers in the query and associate with nearest column
    numbers = re.findall(r'\d+\.?\d*', query)
    
    # Age patterns
    if 'age' in query_lower:
        for num in numbers:
            if 'above' in query_lower or 'over' in query_lower or '>' in query:
                filters['age_min'] = float(num)
            elif 'below' in query_lower or 'under' in query_lower or '<' in query:
                filters['age_max'] = float(num)
            else:
                filters['age'] = float(num)
    
    # Gender patterns
    if 'female' in query_lower:
        filters['gender'] = 'Female'
    elif 'male' in query_lower:
        filters['gender'] = 'Male'
    
    # Plant patterns
    plant_match = re.search(r'plant\s+(\w+)', query_lower)
    if plant_match:
        filters['plant'] = plant_match.group(1).capitalize()
    
    return filters

def has_specific_conditions(query):
    """Check if query contains specific conditions that should generate WHERE clauses"""
    specific_keywords = [
        '>', '<', '=', 'above', 'below', 'over', 'under', 'greater than', 'less than',
        'female', 'male', 'plant', 'stroke', 'age', 'stay', 'top', 'limit'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in specific_keywords)

def needs_semantic_search(query):
    """Determine if query needs semantic search for qualitative matching"""
    semantic_keywords = [
        'felt', 'happy', 'sad', 'satisfied', 'complained', 'experienced',
        'symptoms', 'description', 'notes', 'feedback', 'comments'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in semantic_keywords)

def apply_hard_filters_sql(conn, table_name, filters, schema):
    """Apply extracted filters using SQL WHERE clauses"""
    import pandas as pd
    
    conditions = []
    
    for filter_key, filter_value in filters.items():
        # Handle min/max filters
        if filter_key.endswith('_min'):
            col = filter_key.replace('_min', '')
            if col in schema:
                conditions.append(f'"{col}" > {filter_value}')
        elif filter_key.endswith('_max'):
            col = filter_key.replace('_max', '')
            if col in schema:
                conditions.append(f'"{col}" < {filter_value}')
        # Handle contains filters
        elif filter_key.endswith('_contains'):
            col = filter_key.replace('_contains', '')
            if col in schema:
                conditions.append(f'"{col}" LIKE \'%{filter_value}%\'')
        # Handle exact match filters
        elif filter_key in schema:
            if isinstance(filter_value, str):
                conditions.append(f'"{filter_key}" = \'{filter_value}\'')
            else:
                conditions.append(f'"{filter_key}" = {filter_value}')
    
    # Build SQL query
    sql_query = f"SELECT * FROM {table_name}"
    if conditions:
        sql_query += " WHERE " + " AND ".join(conditions)
        st.info(f"🛡️ SQL WHERE clause: " + " AND ".join(conditions))
    
    return pd.read_sql(sql_query, conn)

def apply_hard_filters_pandas(df, filters, schema):
    """Apply extracted filters using Pandas operations"""
    filtered_df = df.copy()
    
    for filter_key, filter_value in filters.items():
        # Handle min/max filters
        if filter_key.endswith('_min'):
            col = filter_key.replace('_min', '')
            if col in df.columns:
                filtered_df = filtered_df[filtered_df[col] > filter_value]
        elif filter_key.endswith('_max'):
            col = filter_key.replace('_max', '')
            if col in df.columns:
                filtered_df = filtered_df[filtered_df[col] < filter_value]
        # Handle contains filters
        elif filter_key.endswith('_contains'):
            col = filter_key.replace('_contains', '')
            if col in df.columns:
                filtered_df = filtered_df[filtered_df[col].str.contains(str(filter_value), case=False, na=False)]
        # Handle exact match filters
        elif filter_key in df.columns:
            if isinstance(filter_value, str):
                filtered_df = filtered_df[filtered_df[filter_key] == filter_value]
            else:
                filtered_df = filtered_df[filtered_df[filter_key] == filter_value]
    
    return filtered_df

def apply_sql_filter(conn, table_name, query, schema):
    """Apply enhanced SQL filtering with proper Text-to-SQL conversion"""
    import pandas as pd
    import re
    
    query_lower = query.lower()
    
    # Enhanced Text-to-SQL prompt engineering
    def build_sql_from_natural_language(query, schema):
        """Convert natural language to valid SQL WHERE clauses"""
        conditions = []
        order_by = None
        limit = None
        
        # Enhanced pattern matching with column type awareness
        for col, dtype in schema.items():
            col_lower = col.lower()
            
            if col_lower in query_lower:
                # Numeric columns - support mathematical operators
                if 'int' in dtype or 'float' in dtype:
                    # Greater than patterns
                    gt_match = re.search(rf'{col_lower}\s*(?:>|greater than|above|over)\s*([\d.]+)', query_lower)
                    if gt_match:
                        value = float(gt_match.group(1))
                        conditions.append(f'"{col}" > {value}')
                        st.info(f"� Applied filter: {col} > {value}")
                    
                    # Less than patterns
                    lt_match = re.search(rf'{col_lower}\s*(?:<|less than|below|under)\s*([\d.]+)', query_lower)
                    if lt_match:
                        value = float(lt_match.group(1))
                        conditions.append(f'"{col}" < {value}')
                        st.info(f"� Applied filter: {col} < {value}")
                    
                    # Equal patterns
                    eq_match = re.search(rf'{col_lower}\s*(?:=|equals?|is)\s*([\d.]+)', query_lower)
                    if eq_match:
                        value = float(eq_match.group(1))
                        conditions.append(f'"{col}" = {value}')
                        st.info(f"� Applied filter: {col} = {value}")
                    
                    # Age-specific patterns
                    if 'age' in col_lower:
                        age_gt = re.search(r'(?:age|ages?)\s*(?:>|above|over|greater than)\s*(\d+)', query_lower)
                        if age_gt:
                            value = float(age_gt.group(1))
                            conditions.append(f'"{col}" > {value}')
                            st.info(f"� Applied filter: {col} > {value}")
                        
                        age_lt = re.search(r'(?:age|ages?)\s*(?:<|below|under|less than)\s*(\d+)', query_lower)
                        if age_lt:
                            value = float(age_lt.group(1))
                            conditions.append(f'"{col}" < {value}')
                            st.info(f"� Applied filter: {col} < {value}")
                
                # String columns - support exact matching and LIKE
                elif 'object' in dtype or 'str' in dtype:
                    # Gender patterns
                    if 'female' in query_lower and 'gender' in col_lower:
                        conditions.append(f'"{col}" = \'Female\'')
                        st.info(f"� Applied filter: {col} = Female")
                    elif 'male' in query_lower and 'gender' in col_lower:
                        conditions.append(f'"{col}" = \'Male\'')
                        st.info(f"� Applied filter: {col} = Male")
                    
                    # Medical conditions
                    elif 'stroke' in query_lower and any(term in col_lower for term in ["diagnosis", "condition", "disease"]):
                        conditions.append(f'"{col}" LIKE \'%stroke%\'')
                        st.info(f"� Applied filter: {col} contains 'stroke'")
                    
                    # Plant/Location patterns
                    elif 'plant' in query_lower and any(term in col_lower for term in ["plant", "location", "facility"]):
                        plant_match = re.search(r'plant\s+(\w+)', query_lower)
                        if plant_match:
                            plant = plant_match.group(1).capitalize()
                            conditions.append(f'"{col}" = \'{plant}\'')
                            st.info(f"� Applied filter: {col} = {plant}")
                    
                    # General exact matching
                    else:
                        # Get distinct values for exact matching
                        try:
                            distinct_vals = pd.read_sql(f'SELECT DISTINCT "{col}" FROM {table_name}', conn)[col].unique()
                            for val in distinct_vals:
                                if str(val).lower() in query_lower:
                                    conditions.append(f'"{col}" = \'{val}\'')
                                    st.info(f"� Applied filter: {col} = {val}")
                                    break
                        except:
                            pass
        
        # Handle ranking and ordering
        if any(keyword in query_lower for keyword in ["top", "highest", "largest", "best"]):
            numeric_cols = [col for col, dtype in schema.items() if 'int' in dtype or 'float' in dtype]
            
            # Smart column detection for ranking
            revenue_col = next((col for col in numeric_cols if any(term in col.lower() for term in ["revenue", "sales", "amount", "value", "cost", "price"])), None)
            if revenue_col:
                order_by = f'"{revenue_col}" DESC'
                st.info(f"� Ordering by: {revenue_col} (descending)")
            elif numeric_cols:
                order_by = f'"{numeric_cols[0]}" DESC'
                st.info(f"� Ordering by: {numeric_cols[0]} (descending)")
            
            # Extract limit
            top_match = re.search(r'top\s+(\d+)', query_lower)
            if top_match:
                limit = int(top_match.group(1))
                st.info(f"🔧 Limiting to: Top {limit} results")
        
        return conditions, order_by, limit
    
    # Build SQL query
    conditions, order_by, limit = build_sql_from_natural_language(query, schema)
    
    sql_query = f"SELECT * FROM {table_name}"
    
    # Add WHERE conditions (HARD FILTERING)
    if conditions:
        sql_query += " WHERE " + " AND ".join(conditions)
        st.success(f"🛡️ Applied {len(conditions)} hard filters for precise results")
    
    # Add ordering
    if order_by:
        sql_query += f" ORDER BY {order_by}"
    
    # Add limit
    if limit:
        sql_query += f" LIMIT {limit}"
    
    # Execute SQL query
    st.info(f"🔍 Executing SQL: {sql_query}")
    result_df = pd.read_sql(sql_query, conn)
    return result_df

def apply_intelligent_filter(df, query, schema):
    """Apply intelligent filtering based on natural language query and detected schema"""
    query_lower = query.lower()
    filtered_df = df.copy()
    
    # Enhanced filtering using schema information
    for col, dtype in schema.items():
        col_lower = col.lower()
        
        # Check if column is mentioned in query
        if col_lower in query_lower:
            # Handle numeric columns
            if 'int' in dtype or 'float' in dtype:
                if ">" in query:
                    parts = query.split(">")
                    if len(parts) == 2 and col_lower in parts[0].lower():
                        try:
                            value = float(parts[1].strip())
                            filtered_df = filtered_df[filtered_df[col] > value]
                            st.info(f"🔍 Applied filter: {col} > {value}")
                        except:
                            pass
                
                elif "<" in query:
                    parts = query.split("<")
                    if len(parts) == 2 and col_lower in parts[0].lower():
                        try:
                            value = float(parts[1].strip())
                            filtered_df = filtered_df[filtered_df[col] < value]
                            st.info(f"🔍 Applied filter: {col} < {value}")
                        except:
                            pass
            
            # Handle categorical/string columns
            elif 'object' in dtype or 'str' in dtype:
                if "=" in query or "is" in query_lower or "==" in query:
                    # Look for exact matches in the query
                    for val in filtered_df[col].unique():
                        if str(val).lower() in query_lower:
                            filtered_df = filtered_df[filtered_df[col] == val]
                            st.info(f"🔍 Applied filter: {col} = {val}")
                            break
    
    return filtered_df

def is_reasoning_query(query):
    """Detect if query requires reasoning beyond simple filtering"""
    reasoning_keywords = [
        "why", "what", "how", "compare", "which", "reason", "cause", 
        "contribute", "trend", "pattern", "analysis", "insight",
        "relationship", "correlation", "difference", "summary"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in reasoning_keywords)

def apply_llm_reasoning(filtered_df, query, semantic_results=None):
    """Apply LLM reasoning over the data and embeddings"""
    # This is a template for LLM integration
    # You can replace this with actual LLM calls (OpenAI, Anthropic, etc.)
    
    reasoning_prompt = f"""
    Analyze the following data and query to provide insights:
    
    Query: {query}
    Data Summary: {len(filtered_df)} records found
    Key Statistics: {filtered_df.describe().to_string() if not filtered_df.empty else 'No data'}
    
    Please provide:
    1. Direct answer to the query
    2. Key insights from the data
    3. Notable patterns or trends
    4. Recommendations based on the analysis
    """
    
    # Placeholder for actual LLM integration
    reasoning_response = generate_mock_reasoning(query, filtered_df)
    
    return reasoning_response

def generate_mock_reasoning(query, filtered_df):
    """Generate mock reasoning responses (replace with actual LLM)"""
    query_lower = query.lower()
    
    if "why" in query_lower and ("failure" in query_lower or "plant" in query_lower):
        return """
        📊 **Analysis of Plant Performance**
        
        Based on the data, Plant B shows higher failure rates due to:
        - Equipment age: Older machinery in Plant B (avg 8.2 years vs 4.1 years)
        - Production volume: 23% higher output increases stress on equipment
        - Maintenance schedule: Less frequent preventive maintenance
        
        💡 **Recommendations:**
        1. Upgrade critical equipment in Plant B
        2. Implement predictive maintenance program
        3. Consider load balancing across plants
        """
    
    elif "which" in query_lower and ("department" in query_lower or "cost" in query_lower):
        return """
        💰 **Cost Analysis by Department**
        
        The Manufacturing department contributes the highest costs:
        - 42% of total operational costs
        - Primary drivers: Raw materials (58%), Labor (32%), Overhead (10%)
        
        📈 **Cost Optimization Opportunities:**
        1. Bulk purchasing agreements for raw materials
        2. Process automation to reduce labor costs
        3. Energy efficiency improvements
        """
    
    elif "compare" in query_lower and ("quarter" in query_lower or "period" in query_lower):
        return """
        📅 **Quarter-over-Quarter Comparison**
        
        Key trends identified:
        - Revenue growth: +15.3% vs previous quarter
        - Customer acquisition: +8.7% increase
        - Operating costs: +3.2% (controlled growth)
        
        ⚠️ **Areas of Concern:**
        - Customer churn rate increased by 0.8%
        - Average order value decreased by 2.1%
        """
    
    else:
        return f"""
        🔍 **Query Analysis: {query}**
        
        Found {len(filtered_df)} matching records.
        
        Key insights from the data:
        • Dataset contains {len(filtered_df.columns)} variables
        • Time period covers the last 12 months
        • Most significant patterns identified in the data
        
        📊 **Recommendations:**
        • Further analysis recommended for trend identification
        • Consider additional data sources for comprehensive view
        • Implement monitoring for key metrics
        """

def generate_followup_questions(query, filtered_df, schema):
    """Generate intelligent follow-up questions based on context"""
    query_lower = query.lower()
    followups = []
    
    if "plant" in query_lower or "failure" in query_lower:
        followups.extend([
            "What is the maintenance schedule for Plant B?",
            "How does Plant A compare to Plant B in productivity?",
            "What are the top failure types across all plants?"
        ])
    
    elif "department" in query_lower or "cost" in query_lower:
        followups.extend([
            "Which department has the highest ROI?",
            "How have costs changed over the last 6 months?",
            "What is the cost per employee by department?"
        ])
    
    elif "quarter" in query_lower or "compare" in query_lower:
        followups.extend([
            "What are the year-to-date trends?",
            "How does this compare to last year?",
            "Which products performed best this quarter?"
        ])
    
    else:
        # Generic follow-ups based on available columns
        numeric_cols = [col for col, dtype in schema.items() if 'int' in dtype or 'float' in dtype]
        if numeric_cols:
            followups.append(f"What are the trends in {numeric_cols[0]}?")
        
        categorical_cols = [col for col, dtype in schema.items() if 'object' in dtype]
        if categorical_cols:
            followups.append(f"How does {categorical_cols[0]} affect performance?")
        
        followups.append("What are the key insights from this dataset?")
    
    return followups[:3]  # Return top 3 follow-up questions

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _show_startup_error(e)
        raise
