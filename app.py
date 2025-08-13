"""
Streamlit application demonstrating ConTextTab‑powered missing value imputation.

This app allows users to upload a table (Excel/CSV) or select from a set of
pre‑generated examples.  It computes basic KPIs about the data, applies
ConTextTab to infer missing values and presents the completed table with
imputed cells highlighted.  The application **requires** the ConTextTab
library (and thus Python 3.11 or newer).  If the ConTextTab library cannot
be imported, the app will display an error and refuse to run.  There is
**no fallback** to mean/mode imputation.
"""

from __future__ import annotations

import os
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

# Attempt to import ConTextTab.  This library requires Python 3.11 or newer.
try:
    from contexttab import ConTextTabRegressor, ConTextTabClassifier
    _CONTEXTTAB_AVAILABLE = True
    _CONTEXTTAB_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:
    _CONTEXTTAB_AVAILABLE = False
    _CONTEXTTAB_IMPORT_ERROR = e


def load_table(uploaded_file: Optional[bytes] = None, example_name: Optional[str] = None) -> pd.DataFrame:
    """Load a table either from an uploaded file or from an example."""
    if example_name:
        examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
        filename = f"{example_name}_example.xlsx"
        path = os.path.join(examples_dir, filename)
        return pd.read_excel(path)

    if uploaded_file is None:
        raise ValueError("Either an uploaded file or an example name must be provided.")

    name = uploaded_file.name
    if name.lower().endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif name.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {name}")


def compute_kpis(df: pd.DataFrame) -> Tuple[int, int, int, float]:
    """Compute KPIs for the given DataFrame."""
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    total_cells = rows * cols
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0
    return rows, cols, missing_cells, missing_pct


def _prepare_features_for_contexttab(X: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Preprocess feature matrix for ConTextTab."""
    processed = X.copy()
    for col in processed.columns:
        if pd.api.types.is_numeric_dtype(processed[col]):
            mean_val = reference[col].dropna().mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            processed[col] = processed[col].fillna(mean_val)
        else:
            mode_val = reference[col].dropna().mode().iloc[0] if not reference[col].dropna().empty else 'missing'
            processed[col] = processed[col].fillna(mode_val)
            processed[col] = processed[col].astype(str)
    return processed


def impute_with_contexttab(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values using ConTextTab.  Raises RuntimeError if unavailable."""
    if not _CONTEXTTAB_AVAILABLE:
        raise RuntimeError(
            "ConTextTab could not be imported in this environment. "
            "Please ensure that the `contexttab` package is installed and you "
            "are running Python 3.11 or newer. Import error: "
            f"{_CONTEXTTAB_IMPORT_ERROR}"
        )

    imputed_df = df.copy()
    imputed_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        if not df[col].isna().any():
            continue

        Model = ConTextTabRegressor if pd.api.types.is_numeric_dtype(df[col]) else ConTextTabClassifier
        train_df = df.dropna(subset=[col])
        test_idx = df[df[col].isna()].index
        if train_df.empty or len(test_idx) == 0:
            continue

        X_train_raw = train_df.drop(columns=[col])
        y_train = train_df[col]
        X_train = _prepare_features_for_contexttab(X_train_raw, train_df)

        X_test_raw = df.loc[test_idx].drop(columns=[col])
        X_test = _prepare_features_for_contexttab(X_test_raw, train_df)

        model = Model(bagging=1, max_context_size=1024, test_chunk_size=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        flat_preds = np.asarray(preds).squeeze().tolist()
        imputed_df.loc[test_idx, col] = flat_preds
        imputed_mask.loc[test_idx, col] = True

    return imputed_df, imputed_mask


def render_table_with_highlight(df: pd.DataFrame, mask: pd.DataFrame) -> None:
    """Render a DataFrame in Streamlit with imputed cells highlighted in green."""
    def highlight(c: pd.Series) -> List[str]:
        return [
            'background-color: #107E3E; color: #FFFFFF;' if m else ''
            for m in mask[c.name]
        ]
    styled = (
        df.style
        .set_properties(**{'color': '#FFFFFF', 'font-size': '0.95rem'})
        .apply(highlight, axis=0)
    )
    st.markdown(styled.to_html(index=False), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="ConTextTab Imputation Demo",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark-theme CSS with white text and coherent sizing
    st.markdown(
        f"""
        <style>
          :root {{
            --sap-brand-blue: #0A6ED1;
            --sap-positive-green: #107E3E;
            --bg-main: #0B1F33;
            --bg-card: #0F2A4C;
            --txt-color: #FFFFFF;
          }}
          html, body, [class*="css"] {{
            background-color: var(--bg-main) !important;
            color: var(--txt-color) !important;
            font-size: 16px !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
          }}
          section[data-testid="stSidebar"] {{
            background: var(--bg-card) !important;
            color: var(--txt-color) !important;
          }}
          h1, h2, h3, h4, h5, h6 {{
            color: var(--txt-color) !important;
            font-weight: 700 !important;
          }}
          p, label, span, div, li {{
            color: var(--txt-color) !important;
            font-size: 1rem !important;
          }}
          h2 {{ font-size: 1.5rem !important; }}
          .stButton>button {{
            background: var(--sap-brand-blue) !important;
            color: var(--txt-color) !important;
            border-radius: 12px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: .45rem .9rem !important;
          }}
          .stButton>button:hover {{
            background: #0854A1 !important;
          }}
          .stDownloadButton>button {{
            background: transparent !important;
            color: var(--txt-color) !important;
            border: 1px solid var(--sap-brand-blue) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: .45rem .9rem !important;
          }}
          .stDownloadButton>button:hover {{
            background: var(--sap-brand-blue) !important;
            color: var(--txt-color) !important;
          }}
          div[data-testid="stFileUploaderDropzone"] {{
            background: var(--bg-card) !important;
            border: 1px dashed rgba(255,255,255,0.3) !important;
          }}
          div[data-testid="stFileUploaderDropzone"] * {{
            color: var(--txt-color) !important;
          }}
          [data-testid="stMetricValue"] {{
            color: var(--txt-color) !important;
            font-size: 1.35rem !important;
            font-weight: 700 !important;
          }}
          [data-testid="stMetricLabel"] {{
            color: var(--txt-color) !important;
            opacity: .9 !important;
            font-size: .95rem !important;
          }}
          div[data-testid="stDataFrame"] * {{
            color: var(--txt-color) !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("ConTextTab Imputation Demo")
    st.markdown(
        """
        This app demonstrates how the SAP **ConTextTab** model can be used to fill in
        missing values in tabular data.  You can either upload your own Excel or
        CSV file, or select from one of the preloaded examples.  After loading a
        table, click **Impute Missing Data** to apply ConTextTab.  Imputed cells are
        highlighted with a translucent green background.  Note: this app requires
        the `contexttab` package and Python 3.11 or newer.
        """
    )

    # Early error if ConTextTab isn't installed
    if not _CONTEXTTAB_AVAILABLE:
        st.error(
            "ConTextTab could not be imported in this environment.\n"
            "This application requires the `contexttab` Python package and Python 3.11 or newer.\n"
            f"Import error: {_CONTEXTTAB_IMPORT_ERROR}"
        )
        st.stop()

    # Sidebar: choose example or upload file
    st.sidebar.header("Data Input")
    examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
    available_examples = [f.split('_example.xlsx')[0] for f in os.listdir(examples_dir) if f.endswith('.xlsx')]
    example_choice = st.sidebar.selectbox(
        "Choose an example dataset",
        options=[None] + available_examples,
        index=0,
        format_func=lambda x: "Select an example..." if x is None else x.title(),
    )
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Or upload your own table", type=["csv", "xlsx", "xls"])

    df: Optional[pd.DataFrame] = None
    if example_choice:
        df = load_table(example_name=example_choice)
    elif uploaded_file is not None:
        df = load_table(uploaded_file=uploaded_file)
    else:
        st.info("Select an example dataset from the sidebar or upload a file to begin.")

    if df is not None:
        rows, cols, missing, missing_pct = compute_kpis(df)
        st.subheader("Data Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{rows}")
        c2.metric("Columns", f"{cols}")
        c3.metric("Missing cells", f"{missing}")
        c4.metric("Missing %", f"{missing_pct:.1f}%")

        # Preview original data with subtle dark highlight on missing cells
        with st.expander("Show original data", expanded=False):
            missing_mask = df.isna()
            def highlight_missing(c: pd.Series) -> List[str]:
                return [
                    'background-color: #3b4c63;' if m else ''
                    for m in missing_mask[c.name]
                ]
            styled = df.style.apply(highlight_missing, axis=0)
            st.markdown(styled.to_html(index=False), unsafe_allow_html=True)

        # Imputation
        if st.button("Impute Missing Data"):
            with st.spinner("Running ConTextTab imputation.  This may take a while..."):
                imputed_df, mask = impute_with_contexttab(df)
            st.success("Imputation complete!")

            st.subheader("Imputed Data")
            render_table_with_highlight(imputed_df, mask)

            # Download completed table
            from io import BytesIO
            output = BytesIO()
            imputed_df.to_excel(output, index=False)
            st.download_button(
                label="Download completed table",
                data=output.getvalue(),
                file_name="imputed_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            # Additional safeguard in case ConTextTab becomes unavailable mid-run
            if not _CONTEXTTAB_AVAILABLE:
                st.error(
                    "ConTextTab became unavailable during runtime.\n"
                    "Please restart the app after installing the `contexttab` package on Python 3.11 or newer."
                )
                st.stop()


if __name__ == '__main__':
    main()