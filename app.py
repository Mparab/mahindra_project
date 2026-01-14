import os
import tempfile
import re
from typing import List, Any, Dict
from dataclasses import dataclass

import streamlit as st

# Basic UI header
st.title("Streamlit smoke test")
st.write("Python:", __import__("sys").version.splitlines()[0])

# Try to import pandas; gracefully fall back if unavailable
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception as e:
    pd = None
    PANDAS_AVAILABLE = False
    try:
        st.error(
            "Pandas failed to import (likely numpy/pandas binary mismatch). "
            "Tabular (CSV/Excel) support is disabled. To enable, reinstall/upgrade numpy and pandas, for example:\n\n"
            "pip install --upgrade numpy pandas\n\n"
            f"Import error: {e}"
        )
    except Exception:
        pass

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

# Minimal Document dataclass
@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

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
            return pd.DataFrame(rows) if PANDAS_AVAILABLE else rows
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

    if PANDAS_AVAILABLE:
        try:
            return pd.DataFrame(rows)
        except Exception:
            return rows
    return rows

def build_documents_from_extracted_df(df) -> List[Document]:
    docs: List[Document] = []
    if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
        for idx, row in df.iterrows():
            page_content = str(row.get("chunk_text", "")) if hasattr(row, "get") else str(row.get("text", ""))
            metadata = {
                "source": row.get("source") if hasattr(row, "get") else None,
                "page": row.get("page") if hasattr(row, "get") else None,
                "table_id": row.get("table_id") if hasattr(row, "get") else None,
                "chunk_index": int(idx)
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
    if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
        for idx, row in df.iterrows():
            if "text" in df.columns:
                content = str(row.get("text", ""))
            else:
                content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            metadata = {"source": None, "row_index": int(idx)}
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
uploaded_file = st.file_uploader("Upload file", type=["pdf", "docx", "doc", "txt", "csv", "xlsx", "xls"])

df = None
documents: List[Document] = []

if uploaded_file is not None:
    fname = uploaded_file.name
    ext = os.path.splitext(fname)[1].lower()

    # Tabular files
    if ext in (".csv", ".xlsx", ".xls"):
        if not PANDAS_AVAILABLE:
            st.error("Cannot read CSV/Excel because pandas failed to import; repair pandas to enable tabular uploads.")
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
                    st.write("Preview of uploaded tabular file:")
                    st.dataframe(df.head(10))
            except Exception as e:
                st.error(f"Failed to read tabular file: {e}")
                df = None
    else:
        # Non-tabular -> extract
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            ext_df = extract_with_docling_or_fallback(tmp_path)
            has_content = False
            if PANDAS_AVAILABLE and isinstance(ext_df, pd.DataFrame):
                has_content = not ext_df.empty
            else:
                has_content = bool(ext_df)

            if has_content:
                if PANDAS_AVAILABLE and isinstance(ext_df, pd.DataFrame):
                    st.write(f"Extracted {len(ext_df)} chunks from {fname}:")
                    st.dataframe(ext_df.head(10))
                    df = ext_df.rename(columns={"chunk_text": "text"}).reset_index(drop=True)
                else:
                    st.write(f"Extracted {len(ext_df)} chunks from {fname}:")
                    preview = ext_df[:10]
                    st.write(preview)
                    df = []
                    for r in ext_df:
                        newr = dict(r)
                        if "chunk_text" in newr:
                            newr["text"] = newr.pop("chunk_text")
                        df.append(newr)

                # store Document objects in session
                key = f"docs_{fname}"
                st.session_state[key] = build_documents_from_extracted_df(ext_df)
                documents = st.session_state[key]
            else:
                st.error("No content extracted from the document.")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# Make sure documents exist (either from session or from df)
if uploaded_file is not None:
    key = f"docs_{uploaded_file.name}"
else:
    key = None

if key and key in st.session_state:
    documents = st.session_state[key]
else:
    if df is not None:
        key = f"docs_{uploaded_file.name}" if uploaded_file is not None else "docs_in_memory"
        if key not in st.session_state:
            st.session_state[key] = build_documents_from_df(df)
        documents = st.session_state.get(key, [])
    else:
        documents = []

# Show count of documents (simple feedback)
st.write(f"Documents prepared: {len(documents)}")
if len(documents) > 0:
    st.write(documents[0].page_content[:500])
