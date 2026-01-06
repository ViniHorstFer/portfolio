"""
═══════════════════════════════════════════════════════════════════════════════
DATA STORAGE MODULE - File Management for Fund Analytics Platform
═══════════════════════════════════════════════════════════════════════════════

This module handles data file storage and retrieval.

STORAGE OPTIONS:
1. GitHub LFS (Recommended) - Files stored in Git repo, loaded directly
2. Local Upload - User uploads files through the app interface
3. Session Cache - Files cached in session state after first load

IMPORTANT: For files > 50MB (like funds_info.pkl at 70MB), use GitHub LFS.
Supabase free tier only supports files up to 50MB.

GITHUB LFS SETUP:
1. Install Git LFS: git lfs install
2. Track large files: git lfs track "*.pkl" "*.xlsx"
3. Commit .gitattributes
4. Push to GitHub
5. Streamlit Cloud automatically handles LFS files
"""

import streamlit as st
import pandas as pd
import joblib
import io
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import json


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default file paths (relative to app root - works with GitHub LFS)
DEFAULT_PATHS = {
    'fund_metrics': 'data/fund_metrics.pkl',      # Changed to pkl for speed
    'fund_details': 'data/funds_info.pkl',
    'benchmarks': 'data/benchmarks_data.pkl',     # Changed to pkl for speed
}

# Alternative xlsx paths (for backwards compatibility)
XLSX_PATHS = {
    'fund_metrics': 'Sheets/fund_metrics.xlsx',
    'benchmarks': 'Sheets/benchmarks_data.xlsx',
}

# Session state keys for caching
CACHE_KEYS = {
    'fund_metrics': '_cached_fund_metrics',
    'fund_details': '_cached_fund_details',
    'benchmarks': '_cached_benchmarks',
    'last_update': '_cached_last_update',
}


# ═══════════════════════════════════════════════════════════════════════════════
# FILE LOADING FUNCTIONS (with caching)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_pickle_file(file_path: str) -> Any:
    """Load a pickle file with caching."""
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_excel_file(file_path: str) -> pd.DataFrame:
    """Load an Excel file with caching."""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None


def load_fund_metrics(
    file_path: Optional[str] = None,
    uploaded_file: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """
    Load fund metrics data.
    Priority: uploaded_file > session_state > file_path > default_path
    """
    # Check session state first
    if CACHE_KEYS['fund_metrics'] in st.session_state:
        return st.session_state[CACHE_KEYS['fund_metrics']]
    
    # Try uploaded file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.pkl'):
                data = joblib.load(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.session_state[CACHE_KEYS['fund_metrics']] = data
            return data
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    
    # Try provided path
    if file_path:
        if file_path.endswith('.pkl') and os.path.exists(file_path):
            data = load_pickle_file(file_path)
        elif os.path.exists(file_path):
            data = load_excel_file(file_path)
        else:
            data = None
        
        if data is not None:
            st.session_state[CACHE_KEYS['fund_metrics']] = data
            return data
    
    # Try default paths
    for path in [DEFAULT_PATHS['fund_metrics'], XLSX_PATHS.get('fund_metrics', '')]:
        if path and os.path.exists(path):
            if path.endswith('.pkl'):
                data = load_pickle_file(path)
            else:
                data = load_excel_file(path)
            if data is not None:
                st.session_state[CACHE_KEYS['fund_metrics']] = data
                return data
    
    return None


def load_fund_details(
    file_path: Optional[str] = None,
    uploaded_file: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """
    Load fund details (quota history) data.
    Priority: uploaded_file > session_state > file_path > default_path
    """
    # Check session state first
    if CACHE_KEYS['fund_details'] in st.session_state:
        return st.session_state[CACHE_KEYS['fund_details']]
    
    # Try uploaded file
    if uploaded_file is not None:
        try:
            data = joblib.load(uploaded_file)
            st.session_state[CACHE_KEYS['fund_details']] = data
            return data
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    
    # Try provided path
    if file_path and os.path.exists(file_path):
        data = load_pickle_file(file_path)
        if data is not None:
            st.session_state[CACHE_KEYS['fund_details']] = data
            return data
    
    # Try default path
    if os.path.exists(DEFAULT_PATHS['fund_details']):
        data = load_pickle_file(DEFAULT_PATHS['fund_details'])
        if data is not None:
            st.session_state[CACHE_KEYS['fund_details']] = data
            return data
    
    return None


def load_benchmarks(
    file_path: Optional[str] = None,
    uploaded_file: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """
    Load benchmarks data.
    Priority: uploaded_file > session_state > file_path > default_path
    """
    # Check session state first
    if CACHE_KEYS['benchmarks'] in st.session_state:
        return st.session_state[CACHE_KEYS['benchmarks']]
    
    # Try uploaded file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.pkl'):
                data = joblib.load(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file, index_col=0)
            st.session_state[CACHE_KEYS['benchmarks']] = data
            return data
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    
    # Try provided path
    if file_path:
        if file_path.endswith('.pkl') and os.path.exists(file_path):
            data = load_pickle_file(file_path)
        elif os.path.exists(file_path):
            data = load_excel_file(file_path)
            if data is not None and data.columns[0] not in ['CDI', 'Date']:
                data = data.set_index(data.columns[0])
        else:
            data = None
        
        if data is not None:
            st.session_state[CACHE_KEYS['benchmarks']] = data
            return data
    
    # Try default paths
    for path in [DEFAULT_PATHS['benchmarks'], XLSX_PATHS.get('benchmarks', '')]:
        if path and os.path.exists(path):
            if path.endswith('.pkl'):
                data = load_pickle_file(path)
            else:
                data = load_excel_file(path)
                if data is not None and data.columns[0] not in ['CDI', 'Date']:
                    data = data.set_index(data.columns[0])
            if data is not None:
                st.session_state[CACHE_KEYS['benchmarks']] = data
                return data
    
    return None


def load_all_data(
    metrics_path: Optional[str] = None,
    details_path: Optional[str] = None,
    benchmarks_path: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load all three data files with async-style progress indication.
    Returns: (fund_metrics, fund_details, benchmarks)
    """
    fund_metrics = None
    fund_details = None
    benchmarks = None
    
    with st.spinner("Loading fund metrics..."):
        fund_metrics = load_fund_metrics(metrics_path)
    
    with st.spinner("Loading fund details..."):
        fund_details = load_fund_details(details_path)
    
    with st.spinner("Loading benchmarks..."):
        benchmarks = load_benchmarks(benchmarks_path)
    
    return fund_metrics, fund_details, benchmarks


# ═══════════════════════════════════════════════════════════════════════════════
# DATA UPDATE UI COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

def render_data_management_panel() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Render the data management panel in the sidebar or main area.
    Allows users to upload new data files.
    Returns: (fund_metrics, fund_details, benchmarks)
    """
    st.markdown("### 📊 Data Management")
    
    # Show current data status
    has_metrics = CACHE_KEYS['fund_metrics'] in st.session_state
    has_details = CACHE_KEYS['fund_details'] in st.session_state
    has_benchmarks = CACHE_KEYS['benchmarks'] in st.session_state
    
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.markdown(f"**Metrics:** {'✅' if has_metrics else '❌'}")
    with status_col2:
        st.markdown(f"**Details:** {'✅' if has_details else '❌'}")
    with status_col3:
        st.markdown(f"**Benchmarks:** {'✅' if has_benchmarks else '❌'}")
    
    # Last update timestamp
    if CACHE_KEYS['last_update'] in st.session_state:
        st.caption(f"Last updated: {st.session_state[CACHE_KEYS['last_update']]}")
    
    st.markdown("---")
    
    # Upload section
    with st.expander("📤 Upload New Data Files", expanded=not (has_metrics and has_details and has_benchmarks)):
        st.info("""
        **Upload your data files to update the app:**
        - `fund_metrics.xlsx` or `fund_metrics.pkl` - Fund performance metrics
        - `funds_info.pkl` - Fund quota history (required)
        - `benchmarks_data.xlsx` or `benchmarks_data.pkl` - Benchmark returns
        
        💡 For permanent storage, commit files to your GitHub repo using Git LFS.
        """)
        
        upload_col1, upload_col2 = st.columns(2)
        
        with upload_col1:
            metrics_file = st.file_uploader(
                "Fund Metrics",
                type=['xlsx', 'pkl'],
                key='upload_metrics',
                help="Upload fund_metrics.xlsx or .pkl"
            )
            
            details_file = st.file_uploader(
                "Fund Details (PKL)",
                type=['pkl'],
                key='upload_details',
                help="Upload funds_info.pkl"
            )
        
        with upload_col2:
            benchmarks_file = st.file_uploader(
                "Benchmarks",
                type=['xlsx', 'pkl'],
                key='upload_benchmarks',
                help="Upload benchmarks_data.xlsx or .pkl"
            )
        
        if st.button("🔄 Load Uploaded Files", use_container_width=True):
            updated = False
            
            if metrics_file:
                data = load_fund_metrics(uploaded_file=metrics_file)
                if data is not None:
                    st.success(f"✅ Loaded fund metrics: {len(data)} funds")
                    updated = True
            
            if details_file:
                data = load_fund_details(uploaded_file=details_file)
                if data is not None:
                    st.success(f"✅ Loaded fund details")
                    updated = True
            
            if benchmarks_file:
                data = load_benchmarks(uploaded_file=benchmarks_file)
                if data is not None:
                    st.success(f"✅ Loaded benchmarks: {len(data.columns)} benchmarks")
                    updated = True
            
            if updated:
                st.session_state[CACHE_KEYS['last_update']] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.rerun()
    
    # Clear cache button
    if st.button("🗑️ Clear Cached Data", use_container_width=True):
        for key in CACHE_KEYS.values():
            if key in st.session_state:
                del st.session_state[key]
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()
    
    # Return current data
    return (
        st.session_state.get(CACHE_KEYS['fund_metrics']),
        st.session_state.get(CACHE_KEYS['fund_details']),
        st.session_state.get(CACHE_KEYS['benchmarks'])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def convert_xlsx_to_pkl(xlsx_path: str, pkl_path: str) -> bool:
    """Convert xlsx file to pkl for faster loading."""
    try:
        if xlsx_path.endswith('fund_metrics'):
            df = pd.read_excel(xlsx_path)
        else:
            df = pd.read_excel(xlsx_path, index_col=0)
        joblib.dump(df, pkl_path)
        return True
    except Exception as e:
        print(f"Error converting {xlsx_path}: {e}")
        return False


def get_data_info() -> Dict[str, Any]:
    """Get information about currently loaded data."""
    info = {}
    
    if CACHE_KEYS['fund_metrics'] in st.session_state:
        df = st.session_state[CACHE_KEYS['fund_metrics']]
        info['fund_metrics'] = {
            'loaded': True,
            'rows': len(df),
            'columns': len(df.columns)
        }
    else:
        info['fund_metrics'] = {'loaded': False}
    
    if CACHE_KEYS['fund_details'] in st.session_state:
        df = st.session_state[CACHE_KEYS['fund_details']]
        info['fund_details'] = {
            'loaded': True,
            'rows': len(df) if hasattr(df, '__len__') else 'N/A'
        }
    else:
        info['fund_details'] = {'loaded': False}
    
    if CACHE_KEYS['benchmarks'] in st.session_state:
        df = st.session_state[CACHE_KEYS['benchmarks']]
        info['benchmarks'] = {
            'loaded': True,
            'rows': len(df),
            'columns': list(df.columns)
        }
    else:
        info['benchmarks'] = {'loaded': False}
    
    info['last_update'] = st.session_state.get(CACHE_KEYS['last_update'], 'Never')
    
    return info


def standardize_cnpj(cnpj: Any) -> str:
    """Standardize CNPJ format for matching."""
    if pd.isna(cnpj):
        return ''
    cnpj_str = str(cnpj)
    cnpj_digits = ''.join(filter(str.isdigit, cnpj_str))
    return cnpj_digits.zfill(14)
