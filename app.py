import os
import tempfile
from typing import List, Any, Dict, MutableMapping, cast
from dataclasses import dataclass
import sys
import traceback
import streamlit as st
from collections.abc import Mapping
from langchain_core.documents import Document

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
        st.write("pandas:", pd.__version__)
        PANDAS_AVAILABLE = True
    except Exception as e:
        pd = None  # type: ignore
        PANDAS_AVAILABLE = False
        st.warning(f"pandas import failed: {e}")
        try:
            st.error(
                "Pandas failed to import (likely numpy/pandas binary mismatch). "
                "Tabular (CSV/Excel) support is disabled. To enable, reinstall/upgrade numpy and pandas, for example:\n\n"
                "pip install --upgrade numpy pandas\n\n"
                f"Import error: {e}"
            )
        except Exception:
            pass

    st.success("Minimal debug UI rendered — environment OK")
    st.write("Now revert to app.py and run streamlit run app.py to capture actual app errors.")

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

    # Process query with hybrid SQL + Vector filtering
    if query and df is not None:
        st.subheader("🔍 Query Results")
        
        try:
            # Get schema from session state
            schema_key = f"schema_{uploaded_file.name}" if uploaded_file is not None else "schema_in_memory"
            schema = session.get(schema_key, {})
            
            # Hybrid Search: Try SQL first, then Vector for semantic understanding
            if conn:
                filtered_df = apply_sql_filter(conn, table_name, query, schema)
                st.success("🔧 Used SQL filtering for precise results")
            else:
                filtered_df = apply_intelligent_filter(df, query, schema)
                st.info("📊 Used pandas filtering")
            
            # Optional: Add vector search for semantic similarity if available
            if vectorstore and len(filtered_df) > 0:
                st.subheader("🧠 Semantic Similarity Results")
                try:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    semantic_results = retriever.invoke(query)
                    
                    if semantic_results:
                        st.write("Semantically similar records:")
                        for i, doc in enumerate(semantic_results, 1):
                            st.markdown(f"**Result {i}**")
                            st.write(doc.page_content)
                            st.write("---")
                except Exception as e:
                    st.warning(f"Semantic search failed: {e}")
            
            # STEP 9: Reasoning & Follow-up Questions
            if vectorstore and is_reasoning_query(query):
                st.subheader("🤖 AI Reasoning & Analysis")
                try:
                    reasoning_result = apply_llm_reasoning(filtered_df, query, semantic_results if 'semantic_results' in locals() else None)
                    st.write(reasoning_result)
                    
                    # Suggest follow-up questions
                    st.subheader("💡 Suggested Follow-up Questions")
                    followups = generate_followup_questions(query, filtered_df, schema)
                    for i, followup in enumerate(followups, 1):
                        st.write(f"{i}. {followup}")
                        
                except Exception as e:
                    st.warning(f"AI reasoning failed: {e}")
                    st.info("This feature requires LLM integration for advanced reasoning.")
            
            # Display main filtered results
            if len(filtered_df) > 0:
                st.success(f"Found {len(filtered_df)} matching records")
                st.dataframe(filtered_df)
                
                # Show summary statistics
                st.subheader("📊 Summary Statistics")
                for col in filtered_df.select_dtypes(include=['number']).columns:
                    st.write(f"**{col}:** Min={filtered_df[col].min():.2f}, Max={filtered_df[col].max():.2f}, Avg={filtered_df[col].mean():.2f}")
            else:
                st.info("No records found matching your query")
                
        except Exception as e:
            st.error(f"Error processing query: {e}")
            st.info("Try simpler queries like: 'show all', 'filter by column_name', 'column_name > value'")

def apply_sql_filter(conn, table_name, query, schema):
    """Apply advanced SQL filtering using natural language processing"""
    import pandas as pd
    import re
    
    query_lower = query.lower()
    
    # Build SQL query based on natural language
    sql_query = f"SELECT * FROM {table_name}"
    conditions = []
    order_by = None
    limit = None
    
    # Enhanced natural language processing
    for col, dtype in schema.items():
        col_lower = col.lower()
        
        if col_lower in query_lower:
            # Handle numeric comparisons
            if 'int' in dtype or 'float' in dtype:
                # Greater than
                if ">" in query:
                    parts = query.split(">")
                    if len(parts) == 2 and col_lower in parts[0].lower():
                        try:
                            value = float(parts[1].strip())
                            conditions.append(f'"{col}" > {value}')
                            st.info(f"🔍 Applied filter: {col} > {value}")
                        except:
                            pass
                
                # Less than
                elif "<" in query:
                    parts = query.split("<")
                    if len(parts) == 2 and col_lower in parts[0].lower():
                        try:
                            value = float(parts[1].strip())
                            conditions.append(f'"{col}" < {value}')
                            st.info(f"🔍 Applied filter: {col} < {value}")
                        except:
                            pass
                
                # Above/Below patterns
                elif "above" in query_lower or "over" in query_lower:
                    match = re.search(rf'{col_lower}\s+(?:above|over)\s+([\d.]+)', query_lower)
                    if match:
                        value = float(match.group(1))
                        conditions.append(f'"{col}" > {value}')
                        st.info(f"🔍 Applied filter: {col} > {value}")
                
                elif "below" in query_lower or "under" in query_lower:
                    match = re.search(rf'{col_lower}\s+(?:below|under)\s+([\d.]+)', query_lower)
                    if match:
                        value = float(match.group(1))
                        conditions.append(f'"{col}" < {value}')
                        st.info(f"🔍 Applied filter: {col} < {value}")
                
                # Age patterns (e.g., "above 60")
                elif "age" in col_lower and ("above" in query_lower or "over" in query_lower):
                    match = re.search(r'(?:above|over)\s+(\d+)', query_lower)
                    if match:
                        value = float(match.group(1))
                        conditions.append(f'"{col}" > {value}')
                        st.info(f"🔍 Applied filter: {col} > {value}")
            
            # Handle categorical/string columns
            elif 'object' in dtype or 'str' in dtype:
                # Gender patterns
                if "female" in query_lower and "gender" in col_lower:
                    conditions.append(f'"{col}" = \'Female\'')
                    st.info(f"🔍 Applied filter: {col} = Female")
                elif "male" in query_lower and "gender" in col_lower:
                    conditions.append(f'"{col}" = \'Male\'')
                    st.info(f"🔍 Applied filter: {col} = Male")
                
                # Medical conditions
                elif "stroke" in query_lower and any(term in col_lower for term in ["diagnosis", "condition", "disease"]):
                    conditions.append(f'"{col}" LIKE \'%stroke%\'')
                    st.info(f"🔍 Applied filter: {col} contains 'stroke'")
                
                # Plant/Location patterns
                elif "plant" in query_lower and any(term in col_lower for term in ["plant", "location", "facility"]):
                    match = re.search(r'plant\s+(\w+)', query_lower)
                    if match:
                        plant = match.group(1).capitalize()
                        conditions.append(f'"{col}" = \'{plant}\'')
                        st.info(f"🔍 Applied filter: {col} = {plant}")
                
                # General equality patterns
                elif "=" in query or "is" in query_lower or "==" in query:
                    # Get distinct values to match against
                    try:
                        distinct_vals = pd.read_sql(f'SELECT DISTINCT "{col}" FROM {table_name}', conn)[col].unique()
                        for val in distinct_vals:
                            if str(val).lower() in query_lower:
                                conditions.append(f'"{col}" = \'{val}\'')
                                st.info(f"🔍 Applied filter: {col} = {val}")
                                break
                    except:
                        pass
    
    # Handle ranking/sorting patterns
    if "top" in query_lower or "highest" in query_lower or "largest" in query_lower:
        # Find numeric columns for ranking
        numeric_cols = [col for col, dtype in schema.items() if 'int' in dtype or 'float' in dtype]
        
        # Revenue/sales patterns
        revenue_col = next((col for col in numeric_cols if any(term in col.lower() for term in ["revenue", "sales", "amount", "value"])), None)
        if revenue_col and ("revenue" in query_lower or "sales" in query_lower):
            order_by = f'"{revenue_col}" DESC'
            st.info(f"🔍 Ordering by: {revenue_col} (descending)")
        elif numeric_cols:
            # Default to first numeric column
            order_by = f'"{numeric_cols[0]}" DESC'
            st.info(f"🔍 Ordering by: {numeric_cols[0]} (descending)")
        
        # Extract limit for top N
        match = re.search(r'top\s+(\d+)', query_lower)
        if match:
            limit = int(match.group(1))
            st.info(f"🔍 Limiting to: Top {limit} results")
    
    # Add conditions to SQL query
    if conditions:
        sql_query += " WHERE " + " AND ".join(conditions)
    
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
