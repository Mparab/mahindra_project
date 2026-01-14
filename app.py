import os
import tempfile
import re
from typing import List, Any, Dict, MutableMapping, cast
from dataclasses import dataclass
import sys
import traceback
import streamlit as st

def _show_startup_error(exc: Exception):
    """Display startup exception in the Streamlit app and also print to stderr."""
    try:
        st.set_page_config(page_title="Startup Error")
        st.title("Application startup error")
        st.error("An exception occurred during app startup:")
        st.exception(exc)
        st.write("Full traceback:")
        st.text(traceback.format_exc())
    except Exception:
        # Fallback to stderr so server logs show the error
        print("Startup error (fallback):", exc, file=sys.stderr)
        traceback.print_exc()

def main():
    # App initialization and UI
    st.set_page_config(page_title="Debug", layout="centered")
    st.title("Streamlit debug smoke test")
    st.write("Python:", sys.version.splitlines()[0])

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
    uploaded_file = st.file_uploader("Upload file", type=["pdf", "docx", "doc", "txt", "csv", "xlsx", "xls"])

    df = None
    documents: List[Document] = []

    # Use a typed view of session_state to satisfy the type checker
    session: MutableMapping[str, Any] = cast(MutableMapping[str, Any], st.session_state)

    if uploaded_file is not None:
        fname = uploaded_file.name
        ext = os.path.splitext(fname)[1].lower()

        # Tabular files
        if ext in (".csv", ".xlsx", ".xls"):
            if not PANDAS_AVAILABLE or pd is None:
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
                if PANDAS_AVAILABLE and pd is not None and isinstance(ext_df, pd.DataFrame):
                    has_content = not ext_df.empty
                else:
                    has_content = bool(ext_df)

                if has_content:
                    if PANDAS_AVAILABLE and pd is not None and isinstance(ext_df, pd.DataFrame):
                        st.write(f"Extracted {len(ext_df)} chunks from {fname}:")
                        st.dataframe(ext_df.head(10))
                        df = ext_df.rename(columns={"chunk_text": "text"}).reset_index(drop=True)
                    else:
                        st.write(f"Extracted {len(ext_df)} chunks from {fname}:")
                        preview = ext_df[:10]
                        st.write(preview)
                        df = []
                        for r in ext_df:
                            # Normalize each extracted row into a plain dict safely.
                            if isinstance(r, dict):
                                newr = dict(r)
                            else:
                                # Try mapping protocol (objects that behave like dicts)
                                try:
                                    newr = dict(r)  # may work for mappings or iterable of pairs
                                except Exception:
                                    # Fallback: try extracting common attributes, otherwise use string repr
                                    try:
                                        text_val = getattr(r, "text", None) or getattr(r, "chunk_text", None)
                                        source_val = getattr(r, "source", None)
                                        page_val = getattr(r, "page", None)
                                        table_id_val = getattr(r, "table_id", None)
                                        if text_val is None and not any(v is not None for v in (source_val, page_val, table_id_val)):
                                            # Nothing meaningful found on the object, use its string form
                                            newr = {"text": str(r)}
                                        else:
                                            newr = {
                                                "text": text_val if text_val is not None else "",
                                                "source": source_val,
                                                "page": page_val,
                                                "table_id": table_id_val,
                                            }
                                    except Exception:
                                        newr = {"text": str(r)}

                            # Normalize legacy key name
                            if "chunk_text" in newr and "text" not in newr:
                                newr["text"] = newr.pop("chunk_text")

                            # Ensure there's at least a text field
                            if "text" not in newr:
                                # try commonly used keys, otherwise empty string
                                newr["text"] = newr.get("text") or newr.get("chunk_text") or ""

                            df.append(newr)

                    # store Document objects in session
                    key = f"docs_{fname}"
                    docs = build_documents_from_extracted_df(ext_df)
                    session[key] = docs
                    documents = cast(List[Document], session[key])
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

    if key and key in session:
        documents = cast(List[Document], session[key])
    else:
        if df is not None:
            key = f"docs_{uploaded_file.name}" if uploaded_file is not None else "docs_in_memory"
            if key not in session:
                session[key] = build_documents_from_df(df)
            documents = cast(List[Document], session.get(key, []))
        else:
            documents = []

    # Show count of documents (simple feedback)
    st.write(f"Documents prepared: {len(documents)}")
    if len(documents) > 0:
        st.write(documents[0].page_content[:500])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _show_startup_error(e)
        raise
