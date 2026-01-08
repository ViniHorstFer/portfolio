"""
═══════════════════════════════════════════════════════════════════════════════
PROFESSIONAL FUND ANALYTICS PLATFORM
Black & Gold Theme - Bloomberg Style
Built with Streamlit + Plotly

REFACTORED VERSION:
- Unified components in components.py
- Data management in data_storage.py  
- Reduced redundancy across tabs
- Optimized with caching and chart downsampling
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
import io
import warnings
import json
import os
import hashlib
import joblib

# Import unified components
from components import (
    PortfolioMetrics,
    create_cumulative_returns_chart,
    create_rolling_sharpe_chart,
    create_rolling_volatility_chart,
    create_underwater_chart,
    create_omega_gauge,
    create_rachev_gauge,
    create_var_cvar_chart,
    create_portfolio_pie_chart,
    style_returns_table,
    style_book_analysis_table,
    downsample_for_chart
)

from data_storage import (
    load_fund_metrics as load_fund_data_from_storage,
    load_fund_details as load_fund_details_from_storage,
    load_benchmarks as load_benchmarks_from_storage,
    render_data_management_panel,
    CACHE_KEYS
)

# GitHub Releases integration
from github_releases import (
    is_github_configured,
    load_fund_metrics_from_github,
    load_fund_details_from_github,
    load_benchmarks_from_github,
    render_github_data_panel,
    clear_github_cache
)

# Wasserstein DRO Optimizer
try:
    from wasserstein_dro_optimizer import WassersteinDROOptimizer, WassersteinDROConfig, OptimizationResult
    DRO_AVAILABLE = True
except ImportError:
    DRO_AVAILABLE = False

# Supabase integration (optional)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

# Data source priority: GitHub Releases (cloud) > Local Files > Upload

# OPTION 1: GitHub Releases (recommended for Streamlit Cloud)
# Configure in .streamlit/secrets.toml - see secrets.toml.example

# OPTION 2: Local file paths (for local development)
def get_data_path(base_name, extensions=['pkl', 'xlsx']):
    """Find data file in common locations."""
    locations = ['data/', 'Sheets/', '']
    for loc in locations:
        for ext in extensions:
            path = f"{loc}{base_name}.{ext}"
            if os.path.exists(path):
                return path
    return None

# Try to find files automatically, fallback to defaults
DEFAULT_METRICS_PATH = get_data_path('fund_metrics') or "Sheets/fund_metrics.xlsx"
DEFAULT_DETAILS_PATH = get_data_path('funds_info', ['pkl']) or "Sheets/funds_info.pkl"
DEFAULT_BENCHMARKS_PATH = get_data_path('benchmarks_data') or "Sheets/benchmarks_data.xlsx"

# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Set these values to enable Supabase portfolio storage
# You can also set them as environment variables or Streamlit secrets
SUPABASE_URL = ""  # e.g., "https://xxxxx.supabase.co"
SUPABASE_KEY = ""  # Your Supabase anon/public key


def get_supabase_client() -> Client:
    """Get Supabase client using config or Streamlit secrets."""
    if not SUPABASE_AVAILABLE:
        return None
    
    # Try Streamlit secrets first
    try:
        url = st.secrets.get("SUPABASE_URL", SUPABASE_URL)
        key = st.secrets.get("SUPABASE_KEY", SUPABASE_KEY)
    except:
        url = SUPABASE_URL
        key = SUPABASE_KEY
    
    if not url or not key:
        return None
    
    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None


def save_portfolio_to_supabase(portfolio_name: str, portfolio: dict, user_id: str = "default") -> bool:
    """Save portfolio to Supabase database."""
    client = get_supabase_client()
    if not client:
        return False
    
    try:
        data = {
            "user_id": user_id,
            "portfolio_name": portfolio_name,
            "portfolio_data": json.dumps(portfolio),
            "updated_at": datetime.now().isoformat()
        }
        
        # Upsert (insert or update if exists)
        result = client.table("recommended_portfolios").upsert(
            data, 
            on_conflict="user_id,portfolio_name"
        ).execute()
        
        return True
    except Exception as e:
        st.error(f"Failed to save portfolio: {e}")
        return False


def load_portfolio_from_supabase(portfolio_name: str, user_id: str = "default") -> dict:
    """Load portfolio from Supabase database."""
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        result = client.table("recommended_portfolios").select("*").eq(
            "user_id", user_id
        ).eq(
            "portfolio_name", portfolio_name
        ).execute()
        
        if result.data and len(result.data) > 0:
            return json.loads(result.data[0]["portfolio_data"])
        return None
    except Exception as e:
        st.error(f"Failed to load portfolio: {e}")
        return None


def list_portfolios_from_supabase(user_id: str = "default") -> list:
    """List all saved portfolios for a user from Supabase."""
    client = get_supabase_client()
    if not client:
        return []
    
    try:
        result = client.table("recommended_portfolios").select(
            "portfolio_name, updated_at"
        ).eq(
            "user_id", user_id
        ).order(
            "updated_at", desc=True
        ).execute()
        
        return result.data if result.data else []
    except Exception as e:
        st.error(f"Failed to list portfolios: {e}")
        return []


def delete_portfolio_from_supabase(portfolio_name: str, user_id: str = "default") -> bool:
    """Delete portfolio from Supabase database."""
    client = get_supabase_client()
    if not client:
        return False
    
    try:
        client.table("recommended_portfolios").delete().eq(
            "user_id", user_id
        ).eq(
            "portfolio_name", portfolio_name
        ).execute()
        return True
    except Exception as e:
        st.error(f"Failed to delete portfolio: {e}")
        return False


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# AUTHENTICATION SYSTEM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

import hashlib

# Predefined users and passwords (stored as SHA-256 hashes for security)
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "analyst": hashlib.sha256("analyst456".encode()).hexdigest(),
    "manager": hashlib.sha256("manager789".encode()).hexdigest(),
    # Add more users here in format: "username": hashlib.sha256("password".encode()).hexdigest()
}

def check_password(username, password):
    """Verify username and password."""
    if username in USERS:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return USERS[username] == hashed_password
    return False

def login_page():
    """Display login page."""
    st.markdown("""
        <style>
        .login-container {
            max-width: 200px;
            margin: 30px auto; /* reduced top margin */
            padding: 40px;
            background-image: url('https://aquamarine-worthy-zebra-762.mypinata.cloud/ipfs/bafybeigayrnnsuwglzkbhikm32ksvucxecuorcj4k36l4de7na6wcdpjsa');
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            background-color: black;
            border: 2px solid #D4AF37;
            border-radius: 10px;
            aspect-ratio: 16 / 16;
        }

        .login-title {
            color: #D4AF37;
            text-align: center;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: 2px;
        }

        .stApp {
            background-image: url('https://aquamarine-worthy-zebra-762.mypinata.cloud/ipfs/bafybeia6qj2jol4spdjraxdlohre7yg7wofe33awh2udn6harmg3an4mdq');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<p class="login-title">FUND ANALYTICS</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #888888; text-align: center; margin-bottom: 20px;">Please sign in to continue</p>', unsafe_allow_html=True)
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("LOGIN", use_container_width=True):
                if check_password(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("âŒ Invalid username or password")
        
        st.markdown('</div>', unsafe_allow_html=True)

def logout_button():
    """Display logout button in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state.get('username', 'Unknown')}")
        if st.button("LOGOUT", use_container_width=True):
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fund Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM STYLING - BLACK & GOLD THEME
# ═══════════════════════════════════════════════════════════════════════════════

BLACK_GOLD_STYLE = """
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-right: 1px solid #D4AF37;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #D4AF37 !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #D4AF37 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #888888 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #D4AF37 0%, #B8941E 100%);
        color: #000000;
        font-weight: 700;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton button:hover {
        background: linear-gradient(90deg, #FFD700 0%, #D4AF37 100%);
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
    }
    
    /* Tables */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #D4AF37 !important;
    }
    
    .dataframe th {
        background-color: #D4AF37 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .dataframe td {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Selectbox, multiselect */
    .stSelectbox label, .stMultiSelect label {
        color: #D4AF37 !important;
        font-weight: 600 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        color: #D4AF37 !important;
        border: 1px solid #D4AF37 !important;
        font-weight: 700 !important;
    }
    
    /* Divider */
    hr {
        border-color: #D4AF37 !important;
        opacity: 0.3;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1a1a1a !important;
        border-left: 4px solid #D4AF37 !important;
        color: #ffffff !important;
    }
    
    /* File uploader */
    .stFileUploader label {
        color: #D4AF37 !important;
        font-weight: 600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-bottom: 2px solid #D4AF37;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888888;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        color: #D4AF37 !important;
        border-bottom: 3px solid #D4AF37 !important;
    }
</style>
"""

st.markdown(BLACK_GOLD_STYLE, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME - BLACK & GOLD
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': '#0a0a0a',  # Match app background exactly
        'plot_bgcolor': '#0a0a0a',   # Match app background exactly
        'font': {'color': '#D4AF37', 'family': 'Arial, sans-serif'},
        'title': {'font': {'size': 20, 'color': '#D4AF37'}},
        'xaxis': {
            'gridcolor': '#333333',
            'linecolor': '#D4AF37',
            'zerolinecolor': '#D4AF37'
        },
        'yaxis': {
            'gridcolor': '#333333',
            'linecolor': '#D4AF37',
            'zerolinecolor': '#D4AF37'
        },
        'legend': {
            'bgcolor': '#0a0a0a',
            'bordercolor': '#D4AF37',
            'borderwidth': 1
        },
        'colorway': ['#D4AF37', '#FFD700', '#B8941E', '#FFA500', '#FF8C00']
    }
}

# Contrasting colors for benchmarks
BENCHMARK_COLORS = ['#00CED1', '#FF69B4', '#32CD32', '#FF6347', '#9370DB', '#FFA500']

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def standardize_cnpj(cnpj):
    """Standardize CNPJ format by removing dots, slashes, and dashes."""
    if pd.isna(cnpj):
        return None
    return str(cnpj).replace('.', '').replace('/', '').replace('-', '')

# ═══════════════════════════════════════════════════════════════════════════════
# COPULA FUNCTIONS FOR EXPOSURE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def to_empirical_cdf(returns_series):
    """Transform returns to uniform [0,1] using empirical CDF (rank-based)."""
    ranks = returns_series.rank()
    n = len(returns_series)
    uniform = ranks / (n + 1)
    return uniform


def gumbel_270_loglik(u, v, theta):
    """Gumbel 270° rotation: captures LOWER tail dependence."""
    u_rot = 1 - u
    u_rot = np.clip(u_rot, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    
    try:
        log_u = -np.log(u_rot)
        log_v = -np.log(v)
        
        sum_term = (log_u**theta + log_v**theta)**(1/theta)
        copula = np.exp(-sum_term)
        
        term1 = copula * sum_term
        term2 = (log_u * log_v)**(theta - 1)
        term3 = (log_u**theta + log_v**theta)**((1/theta) - 2)
        term4 = (1 + (theta - 1) * (log_u**theta + log_v**theta)**(-1/theta))
        
        c = term1 * term2 * term3 * term4 / (u_rot * v)
        loglik = np.sum(np.log(np.maximum(c, 1e-10)))
        
        return loglik
    except:
        return -1e10


def estimate_gumbel_270_parameter(u, v):
    """Estimate Gumbel 270° copula parameter using MLE."""
    tau_empirical = stats.kendalltau(u, v)[0]
    
    if tau_empirical <= 0.01:
        return 1.1, False
    
    theta_init = max(1.01, 1 / (1 - tau_empirical))
    
    def neg_loglik(theta):
        if theta <= 1.0:
            return 1e10
        try:
            return -gumbel_270_loglik(u, v, theta)
        except:
            return 1e10
    
    result = minimize_scalar(
        neg_loglik,
        bounds=(1.01, 20),
        method='bounded',
        options={'xatol': 1e-4}
    )
    
    return result.x, result.success


def gumbel_270_tail_dependence(theta):
    """Calculate tail dependence for Gumbel 270°."""
    lambda_lower = 2 - 2 ** (1 / theta)
    lambda_upper = 0.0
    return lambda_lower, lambda_upper


def gumbel_180_loglik(u, v, theta):
    """Survival Gumbel (180° rotation): captures UPPER tail dependence."""
    u_rot = 1 - u
    v_rot = 1 - v
    
    u_rot = np.clip(u_rot, 1e-10, 1 - 1e-10)
    v_rot = np.clip(v_rot, 1e-10, 1 - 1e-10)
    
    try:
        log_u = -np.log(u_rot)
        log_v = -np.log(v_rot)
        
        sum_term = (log_u**theta + log_v**theta)**(1/theta)
        copula = np.exp(-sum_term)
        
        term1 = copula * sum_term
        term2 = (log_u * log_v)**(theta - 1)
        term3 = (log_u**theta + log_v**theta)**((1/theta) - 2)
        term4 = (1 + (theta - 1) * (log_u**theta + log_v**theta)**(-1/theta))
        
        c = term1 * term2 * term3 * term4 / (u_rot * v_rot)
        loglik = np.sum(np.log(np.maximum(c, 1e-10)))
        
        return loglik
    except:
        return -1e10


def estimate_gumbel_180_parameter(u, v):
    """Estimate Survival Gumbel (180°) parameter using MLE."""
    tau_empirical = stats.kendalltau(u, v)[0]
    
    if tau_empirical <= 0.01:
        return 1.1, False
    
    theta_init = max(1.01, 1 / (1 - tau_empirical))
    
    def neg_loglik(theta):
        if theta <= 1.0:
            return 1e10
        try:
            return -gumbel_180_loglik(u, v, theta)
        except:
            return 1e10
    
    result = minimize_scalar(
        neg_loglik,
        bounds=(1.01, 20),
        method='bounded',
        options={'xatol': 1e-4}
    )
    
    return result.x, result.success


def gumbel_180_tail_dependence(theta):
    """Calculate tail dependence for Survival Gumbel (180°)."""
    lambda_upper = 2 - 2 ** (1 / theta)
    lambda_lower = 0.0
    return lambda_lower, lambda_upper


def clayton_tail_dependence(theta):
    """Calculate lower tail dependence for Clayton copula."""
    return 2 ** (-1 / theta)


def estimate_rolling_copula_for_chart(fund_returns, benchmark_returns, window=250):
    """
    Calculate rolling copula metrics for visualization.
    Returns DataFrame with kendall_tau, tail_lower, tail_upper, asymmetry_index.
    """
    # Align benchmark to fund's dates
    benchmark_aligned = benchmark_returns.reindex(fund_returns.index, method='ffill').fillna(0)
    
    # Create aligned dataframe
    aligned = pd.DataFrame({
        'fund': fund_returns,
        'benchmark': benchmark_aligned
    }).dropna()
    
    n = len(aligned)
    
    # Adaptive window sizing
    effective_window = window
    if n < window + 25:
        if n >= 150:
            effective_window = n - 25
        else:
            return None
    
    n_windows = n - effective_window + 1
    
    # Pre-allocate arrays
    tau_series = np.zeros(n_windows)
    tail_lower_series = np.zeros(n_windows)
    tail_upper_series = np.zeros(n_windows)
    asymmetry_series = np.zeros(n_windows)
    dates = []
    
    # Rolling window estimation
    for i in range(n_windows):
        # Extract window
        window_fund = aligned['fund'].iloc[i:i+effective_window]
        window_bench = aligned['benchmark'].iloc[i:i+effective_window]
        
        # Transform to empirical CDF
        u = to_empirical_cdf(window_fund)
        v = to_empirical_cdf(window_bench)
        
        # Calculate Kendall's tau
        tau = stats.kendalltau(u.values, v.values)[0]
        tau_series[i] = tau
        
        # Fit Gumbel 270° for LOWER tail
        theta_lower, success_lower = estimate_gumbel_270_parameter(u.values, v.values)
        
        if success_lower:
            lambda_lower, _ = gumbel_270_tail_dependence(theta_lower)
            tail_lower_series[i] = lambda_lower
        else:
            # Conservative estimate
            tail_lower_series[i] = 0.1
        
        # Fit Gumbel 180° for UPPER tail
        theta_upper, success_upper = estimate_gumbel_180_parameter(u.values, v.values)
        
        if success_upper:
            _, lambda_upper = gumbel_180_tail_dependence(theta_upper)
            tail_upper_series[i] = lambda_upper
        else:
            tail_upper_series[i] = tail_lower_series[i] / 3.0
        
        # Calculate asymmetry
        lambda_l = tail_lower_series[i]
        lambda_u = tail_upper_series[i]
        
        if lambda_l + lambda_u > 0:
            asymmetry_series[i] = (lambda_l - lambda_u) / (lambda_l + lambda_u)
        else:
            asymmetry_series[i] = 0.0
        
        dates.append(aligned.index[i + effective_window - 1])
    
    # Create results DataFrame
    results = pd.DataFrame({
        'kendall_tau': tau_series,
        'tail_lower': tail_lower_series,
        'tail_upper': tail_upper_series,
        'asymmetry_index': asymmetry_series
    }, index=dates)
    
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
@st.cache_data(ttl=3600, show_spinner="Loading fund metrics...")
def load_fund_data(file_path=None, uploaded_file=None):
    """Load fund metrics from file. Supports xlsx and pkl formats."""
    try:
        if uploaded_file is not None:
            if hasattr(uploaded_file, 'name') and uploaded_file.name.endswith('.pkl'):
                df = joblib.load(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        elif file_path is not None:
            if file_path.endswith('.pkl'):
                df = joblib.load(file_path)
            else:
                df = pd.read_excel(file_path)
        else:
            return None
        
        # Clean data
        df = df.replace('n/a', np.nan)
        
        # Standardize CNPJ
        if 'CNPJ' in df.columns:
            df['CNPJ_STANDARD'] = df['CNPJ'].apply(standardize_cnpj)
        
        # Convert numeric columns
        numeric_cols = df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if col not in ['FUNDO DE INVESTIMENTO', 'CNPJ', 'CNPJ_STANDARD', 'GESTOR', 
                          'CATEGORIA BTG', 'SUBCATEGORIA BTG', 'STATUS', 'LAST_UPDATE', 'TRIBUTAÇÃO', 'LIQUIDEZ', 'SUITABILITY']:
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_data
def load_fund_details(file_path=None, uploaded_file=None):
    """
    Load detailed fund data with VL_QUOTA from joblib file.
    Much faster than Excel and handles large DataFrames.
    """
    try:
        if uploaded_file is not None:
            # Load from uploaded file
            df = joblib.load(uploaded_file)
        elif file_path is not None:
            # Load from file path
            df = joblib.load(file_path)
        else:
            return None
        
        # Verify it's a DataFrame
        if not isinstance(df, pd.DataFrame):
            st.error("Loaded object is not a pandas DataFrame")
            return None
        
        # Check if date index already set
        if not isinstance(df.index, pd.DatetimeIndex):
            # The first column should be the date if not already indexed
            if len(df.columns) > 0:
                date_col = df.columns[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col])
                    df = df.set_index(date_col)
                except Exception as e:
                    st.warning(f"Could not parse dates in fund details: {str(e)}")
                    # Continue anyway - might already be properly formatted
        
        # Standardize CNPJ if column exists
        if 'CNPJ_FUNDO' in df.columns:
            df['CNPJ_STANDARD'] = df['CNPJ_FUNDO'].apply(standardize_cnpj)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading fund details from joblib: {str(e)}")
        st.error("Make sure the file is a valid joblib/pickle file containing a pandas DataFrame")
        return None


@st.cache_data(ttl=3600, show_spinner="Loading benchmarks...")
def load_benchmarks(file_path=None, uploaded_file=None):
    """Load benchmark returns data. Supports xlsx and pkl formats."""
    try:
        if uploaded_file is not None:
            if hasattr(uploaded_file, 'name') and uploaded_file.name.endswith('.pkl'):
                df = joblib.load(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        elif file_path is not None:
            if file_path.endswith('.pkl'):
                df = joblib.load(file_path)
            else:
                df = pd.read_excel(file_path)
        else:
            return None
        
        # If already has DatetimeIndex, return as is
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Parse dates
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        
        return df
    except Exception as e:
        st.error(f"Error loading benchmarks: {str(e)}")
        return None


def calculate_cumulative_returns(returns_series):
    """Calculate cumulative returns from daily returns."""
    return (1 + returns_series).cumprod() - 1


def get_fund_returns(fund_details, cnpj_standard, period_months=None):
    """Extract returns for a specific fund - handle duplicate dates by keeping max NR_COTST."""
    if fund_details is None:
        return None
    
    # Use standardized CNPJ for lookup
    fund_data = fund_details[fund_details['CNPJ_STANDARD'] == cnpj_standard].sort_index()
    
    if len(fund_data) == 0:
        return None
    
    # Handle duplicate dates - keep row with highest NR_COTST (largest class)
    if 'NR_COTST' in fund_data.columns:
        fund_data = fund_data.reset_index()
        date_col = fund_data.columns[0]
        fund_data = fund_data.sort_values('NR_COTST', ascending=False)
        fund_data = fund_data.drop_duplicates(subset=[date_col], keep='first')
        fund_data = fund_data.set_index(date_col)
        fund_data = fund_data.sort_index()
    
    quota_series = fund_data['VL_QUOTA'].dropna()
    # Remove zero values (errors in data)
    quota_series = quota_series[quota_series > 0]
    
    if len(quota_series) == 0:
        return None
    
    daily_returns = quota_series.pct_change().dropna()
    
    # Filter by period if specified
    if period_months is not None:
        cutoff_date = daily_returns.index[-1] - pd.DateOffset(months=period_months)
        daily_returns_filtered = daily_returns[daily_returns.index >= cutoff_date]
        return daily_returns_filtered, daily_returns
    
    return daily_returns, daily_returns


def calculate_benchmark_returns(benchmark_data, fund_dates, period_months=None):
    """Calculate benchmark returns aligned to fund dates."""
    aligned = benchmark_data.reindex(fund_dates, method='ffill').fillna(0)
    
    if period_months is not None:
        cutoff_date = fund_dates[-1] - pd.DateOffset(months=period_months)
        aligned = aligned[aligned.index >= cutoff_date]
    
    # Calculate period return
    if len(aligned) > 0:
        return (1 + aligned).prod() - 1
    return np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# MONTHLY RETURNS CALENDAR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_relative_performance(fund_ret, bench_ret):
    """Calculate relative performance handling positive and negative returns correctly."""
    if pd.isna(fund_ret) or pd.isna(bench_ret):
        return np.nan
    
    if bench_ret == 0:
        return np.nan
    
    if fund_ret >= 0 and bench_ret >= 0:
        return fund_ret / bench_ret
    elif fund_ret < 0 and bench_ret < 0:
        return bench_ret / fund_ret
    elif fund_ret > 0 and bench_ret < 0:
        return (fund_ret - bench_ret) / abs(bench_ret)
    else:
        return fund_ret / bench_ret


def create_monthly_returns_table(fund_returns_full, benchmark_data, comparison_method='Relative Performance'):
    """Create monthly returns table organized by year with fund, benchmark, and comparison."""
    # Convert daily returns to monthly
    fund_monthly = fund_returns_full.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Align benchmark with fund dates
    aligned_benchmark = benchmark_data.reindex(fund_returns_full.index, method='ffill').fillna(0)
    benchmark_monthly = aligned_benchmark.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Get years with data (reverse order - latest first)
    years = sorted(fund_monthly.index.year.unique(), reverse=True)
    
    # Month names
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Build the data structure
    table_data = []
    
    # First pass: calculate all cumulative values forward
    cumulative_data = {}
    cumulative_fund_temp = 1.0
    cumulative_benchmark_temp = 1.0
    
    for year in sorted(fund_monthly.index.year.unique()):
        year_fund = fund_monthly[fund_monthly.index.year == year]
        year_benchmark = benchmark_monthly[benchmark_monthly.index.year == year]
        
        ytd_fund = 1.0
        ytd_benchmark = 1.0
        
        for month_num in range(1, 13):
            fund_month = year_fund[year_fund.index.month == month_num]
            bench_month = year_benchmark[year_benchmark.index.month == month_num]
            
            if len(fund_month) > 0:
                ret = fund_month.iloc[0]
                ytd_fund *= (1 + ret)
                cumulative_fund_temp *= (1 + ret)
            
            if len(bench_month) > 0:
                ret = bench_month.iloc[0]
                ytd_benchmark *= (1 + ret)
                cumulative_benchmark_temp *= (1 + ret)
        
        cumulative_data[year] = {
            'ytd_fund': ytd_fund - 1,
            'ytd_benchmark': ytd_benchmark - 1,
            'cumulative_fund': cumulative_fund_temp - 1,
            'cumulative_benchmark': cumulative_benchmark_temp - 1
        }
    
    # Second pass: build table in reverse order (latest first)
    for year in years:
        year_fund = fund_monthly[fund_monthly.index.year == year]
        year_benchmark = benchmark_monthly[benchmark_monthly.index.year == year]
        
        # Fund row
        fund_row = {'Year': year, 'Type': 'Investment Fund'}
        for month_name in months:
            month_num = months.index(month_name) + 1
            month_data = year_fund[year_fund.index.month == month_num]
            fund_row[month_name] = month_data.iloc[0] if len(month_data) > 0 else np.nan
        fund_row['YTD'] = cumulative_data[year]['ytd_fund']
        fund_row['Total'] = cumulative_data[year]['cumulative_fund']
        table_data.append(fund_row)
        
        # Benchmark row - only if NOT "Benchmark Performance"
        if comparison_method != 'Benchmark Performance':
            benchmark_row = {'Year': year, 'Type': 'Benchmark'}
            for month_name in months:
                month_num = months.index(month_name) + 1
                month_data = year_benchmark[year_benchmark.index.month == month_num]
                benchmark_row[month_name] = month_data.iloc[0] if len(month_data) > 0 else np.nan
            benchmark_row['YTD'] = cumulative_data[year]['ytd_benchmark']
            benchmark_row['Total'] = cumulative_data[year]['cumulative_benchmark']
            table_data.append(benchmark_row)
        
        # Comparison row
        comparison_row = {'Year': year, 'Type': comparison_method}
        for month_name in months:
            month_num = months.index(month_name) + 1
            fund_month = year_fund[year_fund.index.month == month_num]
            bench_month = year_benchmark[year_benchmark.index.month == month_num]
            
            if len(fund_month) > 0 and len(bench_month) > 0:
                fund_ret = fund_month.iloc[0]
                bench_ret = bench_month.iloc[0]
                
                if comparison_method == 'Relative Performance':
                    comparison_row[month_name] = calculate_relative_performance(fund_ret, bench_ret)
                elif comparison_method == 'Percentage Points':
                    comparison_row[month_name] = fund_ret - bench_ret
                else:  # Benchmark Performance
                    comparison_row[month_name] = bench_ret
            else:
                comparison_row[month_name] = np.nan
        
        # YTD and Total for comparison row
        ytd_fund = cumulative_data[year]['ytd_fund']
        ytd_benchmark = cumulative_data[year]['ytd_benchmark']
        cumul_fund = cumulative_data[year]['cumulative_fund']
        cumul_benchmark = cumulative_data[year]['cumulative_benchmark']
        
        if comparison_method == 'Relative Performance':
            comparison_row['YTD'] = calculate_relative_performance(ytd_fund, ytd_benchmark)
            comparison_row['Total'] = calculate_relative_performance(cumul_fund, cumul_benchmark)
        elif comparison_method == 'Percentage Points':
            comparison_row['YTD'] = ytd_fund - ytd_benchmark
            comparison_row['Total'] = cumul_fund - cumul_benchmark
        else:  # Benchmark Performance
            comparison_row['YTD'] = ytd_benchmark
            comparison_row['Total'] = cumul_benchmark
        
        table_data.append(comparison_row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    cols = ['Year', 'Type'] + months + ['YTD', 'Total']
    df = df[cols]
    
    return df


def style_monthly_returns_table(df, comparison_method):
    """Apply styling to monthly returns table - returns HTML."""
    # Rows per year (2 if Benchmark Performance, 3 otherwise)
    rows_per_year = 2 if comparison_method == 'Benchmark Performance' else 3
    
    def format_value(val, row_type):
        """Format value as percentage with red color for negatives."""
        if pd.isna(val):
            return ''
        
        # All values displayed as percentages
        formatted = f"{val*100:.2f}%"
        
        # Red color for negative values
        if val < 0:
            return f'<span style="color: #FF0000; font-weight: 600;">{formatted}</span>'
        return formatted
    
    # Build HTML table
    html = '<div style="overflow-x: auto;">'
    html += '<table style="width: 100%; border-collapse: collapse; font-size: 12px; border: 2px solid #D4AF37;">'
    
    # Header
    html += '<thead><tr style="background-color: #D4AF37; color: #000000; font-weight: 700; text-transform: uppercase;">'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">Year</th>'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; border-right: 3px solid #D4AF37; text-align: center;">Type</th>'
    
    # Monthly columns
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in months:
        html += f'<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">{month}</th>'
    
    # YTD and Total
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; border-left: 3px solid #D4AF37; text-align: center;">YTD</th>'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">Total</th>'
    html += '</tr></thead>'
    
    # Body
    html += '<tbody>'
    
    current_year = None
    row_in_year = 0
    bg_color = '#1a1a1a'
    
    for idx, row in df.iterrows():
        year = row['Year']
        
        if year != current_year:
            current_year = year
            row_in_year = 0
        
        top_border = '3px solid #D4AF37' if row_in_year == 0 else '1px solid #333333'
        bottom_border = '3px solid #D4AF37' if row_in_year == (rows_per_year - 1) else '1px solid #333333'
        
        html += f'<tr style="background-color: {bg_color};">'
        
        if row_in_year == 0:
            html += f'<td rowspan="{rows_per_year}" style="padding: 10px; border-left: 2px solid #D4AF37; '
            html += f'border-right: 1px solid #333333; border-top: {top_border}; border-bottom: {bottom_border}; '
            html += f'color: #FFD700; font-weight: 700; font-size: 16px; text-align: center; vertical-align: middle;">{year}</td>'
        
        html += f'<td style="padding: 10px; border-top: {top_border}; border-bottom: {bottom_border}; '
        html += f'border-right: 3px solid #D4AF37; border-left: 1px solid #333333; '
        html += f'color: #D4AF37; font-weight: 600; text-align: left;">{row["Type"]}</td>'
        
        for col in months:
            val = row[col]
            formatted_val = format_value(val, row['Type'])
            html += f'<td style="padding: 8px; border-top: {top_border}; border-bottom: {bottom_border}; '
            html += f'border-left: 1px solid #333333; border-right: 1px solid #333333; '
            html += f'color: #FFFFFF; text-align: right;">{formatted_val}</td>'
        
        val = row['YTD']
        formatted_val = format_value(val, row['Type'])
        html += f'<td style="padding: 8px; border-top: {top_border}; border-bottom: {bottom_border}; '
        html += f'border-left: 3px solid #D4AF37; border-right: 1px solid #333333; '
        html += f'color: #FFFFFF; font-weight: 600; text-align: right;">{formatted_val}</td>'
        
        val = row['Total']
        formatted_val = format_value(val, row['Type'])
        html += f'<td style="padding: 8px; border-top: {top_border}; border-bottom: {bottom_border}; '
        html += f'border-left: 1px solid #333333; border-right: 2px solid #D4AF37; '
        html += f'color: #FFFFFF; font-weight: 700; text-align: right;">{formatted_val}</td>'
        
        html += '</tr>'
        row_in_year += 1
    
    html += '</tbody></table></div>'
    
    return html

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_returns_chart(fund_returns, benchmark_returns, fund_name, period_label):
    """Create cumulative returns comparison chart - SOLID LINES ONLY."""
    fig = go.Figure()
    
    # Calculate cumulative returns
    fund_cum = calculate_cumulative_returns(fund_returns) * 100
    
    # Add fund line (GOLD, SOLID)
    fig.add_trace(go.Scatter(
        x=fund_cum.index,
        y=fund_cum.values,
        name=fund_name,
        line=dict(color='#D4AF37', width=3),
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    
    # Add benchmark lines (CONTRASTING COLORS, SOLID)
    for i, (bench_name, bench_data) in enumerate(benchmark_returns.items()):
        aligned_bench = bench_data.reindex(fund_returns.index, method='ffill').fillna(0)
        bench_cum = calculate_cumulative_returns(aligned_bench) * 100
        
        fig.add_trace(go.Scatter(
            x=bench_cum.index,
            y=bench_cum.values,
            name=bench_name,
            line=dict(color=BENCHMARK_COLORS[i % len(BENCHMARK_COLORS)], width=2),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Cumulative Returns - {period_label}',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_returns_comparison_table(fund_info, fund_name, benchmarks, fund_returns, selected_benchmarks):
    """Create returns comparison table with fund + benchmarks as rows."""
    periods = ['MTD', '3M', '6M', 'YTD', '12M', '24M', '36M', 'TOTAL']
    
    # Build data structure
    data = []
    
    # Fund row
    fund_row = {'Asset': fund_name}
    for period in periods:
        col_name = f'RETURN_{period}'
        if col_name in fund_info.index:
            fund_row[f'Return {period}'] = fund_info[col_name]
        else:
            fund_row[f'Return {period}'] = np.nan
    data.append(fund_row)
    
    # Benchmark rows
    if benchmarks is not None and fund_returns is not None:
        for bench_name in selected_benchmarks:
            if bench_name in benchmarks.columns:
                bench_row = {'Asset': bench_name}
                bench_data = benchmarks[bench_name]
                
                # Calculate returns for each period
                period_map = {
                    'MTD': (fund_returns.index[-1].replace(day=1), None),
                    '3M': (3, None),
                    '6M': (6, None),
                    'YTD': (fund_returns.index[-1].replace(month=1, day=1), None),
                    '12M': (12, None),
                    '24M': (24, None),
                    '36M': (36, None),
                    'TOTAL': (None, 'total')
                }
                
                for period in periods:
                    if period == 'TOTAL':
                        period_return = calculate_benchmark_returns(
                            bench_data, 
                            fund_returns.index,
                            period_months=None
                        )
                        bench_row[f'Return {period}'] = period_return
                    elif period in ['MTD', 'YTD']:
                        start_date = period_map[period][0]
                        period_return = calculate_benchmark_returns(
                            bench_data, 
                            fund_returns.index,
                            period_months=None
                        )
                        bench_row[f'Return {period}'] = period_return
                    else:
                        months = int(period.replace('M', ''))
                        period_return = calculate_benchmark_returns(
                            bench_data,
                            fund_returns.index,
                            period_months=months
                        )
                        bench_row[f'Return {period}'] = period_return
                
                data.append(bench_row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Format as percentages
    for col in df.columns:
        if col != 'Asset':
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    
    return df


def create_rolling_sharpe_chart(fund_returns_full, window_months=12):
    """Create rolling Sharpe ratio chart - REMOVE INITIAL GAP."""
    window_days = window_months * 21
    
    rolling_returns = fund_returns_full.rolling(window=window_days).mean() * 252
    rolling_vol = fund_returns_full.rolling(window=window_days).std() * np.sqrt(252)
    rolling_sharpe = rolling_returns / rolling_vol
    
    # Remove NaN values (initial gap)
    rolling_sharpe_clean = rolling_sharpe.dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe_clean.index,
        y=rolling_sharpe_clean.values,
        fill='tozeroy',
        fillcolor='rgba(212, 175, 55, 0.2)',
        line=dict(color='#D4AF37', width=2),
        name='Rolling Sharpe',
        hovertemplate='%{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", opacity=0.5)
    
    fig.update_layout(
        title=f'{window_months}-Month Rolling Sharpe Ratio (Full Period)',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return fig


def create_rolling_vol_chart(fund_returns_full, window_months=12):
    """Create rolling volatility chart - REMOVE INITIAL GAP."""
    window_days = window_months * 21
    rolling_vol = fund_returns_full.rolling(window=window_days).std() * np.sqrt(252) * 100
    
    # Remove NaN values
    rolling_vol_clean = rolling_vol.dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_vol_clean.index,
        y=rolling_vol_clean.values,
        fill='tozeroy',
        fillcolor='rgba(212, 175, 55, 0.2)',
        line=dict(color='#D4AF37', width=2),
        name='Rolling Volatility',
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{window_months}-Month Rolling Volatility (Full Period)',
        xaxis_title='Date',
        yaxis_title='Annualized Volatility (%)',
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return fig

def create_portfolio_pie_chart(weights_series, chart_type, fund_categories, fund_subcategories):
    """
    Create pie charts for portfolio composition.
    
    Parameters:
    -----------
    weights_series : pd.Series
        Portfolio weights
    chart_type : str
        'fund', 'category', or 'subcategory'
    fund_categories : dict
        Mapping of fund to category
    fund_subcategories : dict
        Mapping of fund to subcategory
    """
    
    if chart_type == 'fund':
        # By individual fund
        labels = weights_series.index.tolist()
        values = weights_series.values
        title = 'Portfolio Allocation by Investment Fund'
        
        # Color palette for funds
        colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
        
    elif chart_type == 'category':
        # By category
        category_weights = {}
        for fund, weight in weights_series.items():
            cat = fund_categories.get(fund, 'Unknown')
            category_weights[cat] = category_weights.get(cat, 0) + weight
        
        labels = list(category_weights.keys())
        values = list(category_weights.values())
        title = 'Portfolio Allocation by Category'
        
        # Distinct colors for categories
        colors = px.colors.qualitative.Bold
        
    else:  # subcategory
        # By subcategory
        subcat_weights = {}
        for fund, weight in weights_series.items():
            subcat = fund_subcategories.get(fund, 'Unknown')
            subcat_weights[subcat] = subcat_weights.get(subcat, 0) + weight
        
        labels = list(subcat_weights.keys())
        values = list(subcat_weights.values())
        title = 'Portfolio Allocation by Subcategory'
        
        # Color palette for subcategories
        colors = px.colors.qualitative.Vivid
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.4f}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_underwater_plot(fund_returns_full):
    """Create underwater plot with MAX DD highlighted - YELLOW base, RED max DD (no markers)."""
    cumulative = (1 + fund_returns_full).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100

    # Calculate CDaR (95%) - Conditional Drawdown at Risk
    cdar_95 = PortfolioMetrics.cdar(fund_returns_full, confidence=0.95) * 100
    
    # Find longest drawdown period
    drawdown_values = drawdown.values
    drawdown_dates = drawdown.index.to_numpy()
    is_underwater = drawdown_values < -0.01
    
    underwater_periods = []
    current_start = None
    current_start_idx = None
    current_length = 0
    current_depth = 0
    
    for idx, is_under in enumerate(is_underwater):
        if is_under:
            if current_start is None:
                current_start = drawdown_dates[idx]
                current_start_idx = idx
            current_length += 1
            current_depth = min(current_depth, drawdown_values[idx])
        else:
            if current_start is not None:
                underwater_periods.append({
                    'start': current_start,
                    'start_idx': current_start_idx,
                    'end_idx': idx - 1,
                    'length': current_length,
                    'depth': current_depth
                })
                current_start = None
                current_start_idx = None
                current_length = 0
                current_depth = 0
    
    if current_start is not None:
        underwater_periods.append({
            'start': current_start,
            'start_idx': current_start_idx,
            'end_idx': len(drawdown_values) - 1,
            'length': current_length,
            'depth': current_depth
        })
    
    # Find max drawdown period
    max_dd_period = None
    if underwater_periods:
        max_dd_period = max(underwater_periods, key=lambda x: abs(x['depth']))
    
    fig = go.Figure()
    
    # Base drawdown (YELLOW/GOLD)
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        fillcolor='rgba(212, 175, 55, 0.3)',
        line=dict(color='#D4AF37', width=2),
        mode='lines',
        name='Drawdown',
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    
    # Highlight MAX DD period (RED) - NO MARKERS
    if max_dd_period:
        start_idx = max_dd_period['start_idx']
        end_idx = max_dd_period['end_idx']
        
        fig.add_trace(go.Scatter(
            x=drawdown.index[start_idx:end_idx+1],
            y=drawdown.values[start_idx:end_idx+1],
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.5)',
            line=dict(color='#FF0000', width=3),
            mode='lines',
            name='Max Drawdown Period',
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#D4AF37", opacity=0.5)
    
    # Add CDaR (95%) reference line - ORANGE DASHED
    fig.add_hline(
        y=cdar_95,
        line_dash="dash",
        line_color='#FF8C00',
        line_width=2,
        annotation_text=f"CDaR 95%: {cdar_95:.2f}%",
        annotation_position="right",
        annotation=dict(
            font=dict(size=12, color='#FF8C00'),
            bgcolor='rgba(255, 140, 0, 0.1)',
            bordercolor='#FF8C00',
            borderwidth=1
        )
    )
    
    fig.update_layout(
        title='Underwater Plot (Drawdown from Peak) with CDaR 95%',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template=PLOTLY_TEMPLATE,
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Return figure and info dictionary with both max DD and CDaR
    max_dd_info = {
        'value': max_dd_period['depth'] if max_dd_period else 0,
        'start': max_dd_period['start'] if max_dd_period else None,
        'length': max_dd_period['length'] if max_dd_period else 0,
        'cdar_95': cdar_95
    }
    
    return fig, max_dd_info


def create_omega_gauge(omega_value, frequency='Daily'):
    """Create gauge chart for Omega ratio - BLUE bar, DARK BLACK background."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=omega_value if not np.isinf(omega_value) else 5.0,
        title={'text': f"Omega Ratio ({frequency})", 'font': {'color': '#D4AF37', 'size': 18}},
        number={'font': {'color': '#D4AF37', 'size': 40}},
        gauge={
            'axis': {'range': [0, 5], 'tickcolor': '#D4AF37'},
            'bar': {'color': '#1E90FF', 'thickness': 0.7},
            'bgcolor': '#0a0a0a',
            'bordercolor': '#D4AF37',
            'borderwidth': 2,
            'steps': [
                {'range': [0, 1.0], 'color': '#8B0000', 'name': 'Poor'},
                {'range': [1.0, 1.5], 'color': '#FF4500', 'name': 'Below Avg'},
                {'range': [1.5, 2.0], 'color': '#FFD700', 'name': 'Average'},
                {'range': [2.0, 3.0], 'color': '#90EE90', 'name': 'Very Good'},
                {'range': [3.0, 5.0], 'color': '#32CD32', 'name': 'Excellent'}
            ],
            'threshold': {
                'line': {'color': '#FFFFFF', 'width': 4},
                'thickness': 0.75,
                'value': omega_value if not np.isinf(omega_value) else 5.0
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_rachev_gauge(rachev_value, frequency='Daily'):
    """Create gauge chart for Rachev ratio - BLUE bar, DARK BLACK background."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rachev_value if not np.isinf(rachev_value) else 2.0,
        title={'text': f"Rachev Ratio ({frequency})", 'font': {'color': '#D4AF37', 'size': 18}},
        number={'font': {'color': '#D4AF37', 'size': 40}, 'valueformat': '.2f'},
        gauge={
            'axis': {'range': [0, 2.0], 'tickcolor': '#D4AF37'},
            'bar': {'color': '#1E90FF', 'thickness': 0.7},
            'bgcolor': '#0a0a0a',
            'bordercolor': '#D4AF37',
            'borderwidth': 2,
            'steps': [
                {'range': [0, 0.5], 'color': '#32CD32', 'name': 'Excellent'},
                {'range': [0.5, 0.75], 'color': '#90EE90', 'name': 'Very Good'},
                {'range': [0.75, 1.0], 'color': '#FFD700', 'name': 'Good'},
                {'range': [1.0, 1.5], 'color': '#FF4500', 'name': 'Below Avg'},
                {'range': [1.5, 2.0], 'color': '#8B0000', 'name': 'Poor'}
            ],
            'threshold': {
                'line': {'color': '#FFFFFF', 'width': 4},
                'thickness': 0.75,
                'value': rachev_value if not np.isinf(rachev_value) else 2.0
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_omega_cdf_chart(returns_data, threshold=0, frequency='daily'):
    """Create CDF showing Omega ratio - NO VALUE IN TITLE."""
    returns_pct = returns_data * 100
    sorted_returns = np.sort(returns_pct)
    cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    
    below_threshold = sorted_returns <= threshold
    above_threshold = sorted_returns > threshold
    
    gains = returns_pct[returns_pct > threshold].sum()
    losses = abs(returns_pct[returns_pct <= threshold].sum())
    omega = gains / losses if losses > 0 else np.inf
    
    fig = go.Figure()
    
    # Red area (losses) - NO MARKERS
    if below_threshold.any():
        fig.add_trace(go.Scatter(
            x=sorted_returns[below_threshold],
            y=cdf[below_threshold],
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='#FF0000', width=0),
            mode='lines',
            name=f'Losses',
            hovertemplate='Return: %{x:.2f}%<br>CDF: %{y:.2f}<extra></extra>'
        ))
    
    # Green area (gains) - NO MARKERS
    if above_threshold.any():
        fig.add_trace(go.Scatter(
            x=sorted_returns[above_threshold],
            y=cdf[above_threshold],
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.3)',
            line=dict(color='#00FF00', width=0),
            mode='lines',
            name=f'Gains',
            hovertemplate='Return: %{x:.2f}%<br>CDF: %{y:.2f}<extra></extra>'
        ))
    
    # CDF line
    fig.add_trace(go.Scatter(
        x=sorted_returns,
        y=cdf,
        mode='lines',
        line=dict(color='#D4AF37', width=2),
        name='CDF',
        hovertemplate='Return: %{x:.2f}%<br>CDF: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_vline(x=threshold, line_dash="dash", line_color="#FFFFFF")
    
    fig.update_layout(
        title=f'Omega Ratio Visualization - {frequency.title()} Returns',
        xaxis_title=f'{frequency.title()} Return (%)',
        yaxis_title='Cumulative Probability',
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return fig


def create_combined_rachev_var_chart(returns_data, var_val, cvar_val, frequency='daily'):
    """Combined Rachev/VaR/CVaR chart with highlighted tails."""
    returns_pct = returns_data * 100
    
    # Calculate tail thresholds
    lower_threshold = np.percentile(returns_pct, 5)
    upper_threshold = np.percentile(returns_pct, 95)
    
    expected_loss = returns_pct[returns_pct <= lower_threshold].mean()
    expected_gain = returns_pct[returns_pct >= upper_threshold].mean()
    
    rachev_ratio = expected_gain / abs(expected_loss) if expected_loss > 0 else np.inf
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns_pct,
        nbinsx=50,
        name='Distribution',
        marker=dict(color='#D4AF37', opacity=0.5),
        histnorm='probability density'
    ))
    
    # KDE curve
    if len(returns_pct) > 1:
        kde = gaussian_kde(returns_pct.dropna())
        x_range = np.linspace(returns_pct.min(), returns_pct.max(), 500)
        kde_values = kde(x_range)
        
        # Full KDE (gold)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_values,
            mode='lines',
            name='PDF',
            line=dict(color='#FFD700', width=2)
        ))
        
        # Lower tail (red) - Rachev
        lower_mask = x_range <= lower_threshold
        fig.add_trace(go.Scatter(
            x=x_range[lower_mask],
            y=kde_values[lower_mask],
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.4)',
            line=dict(width=0),
            name=f'Lower Tail 5% (E[Loss]: {expected_loss:.2f}%)',
            showlegend=True
        ))
        
        # Upper tail (green) - Rachev
        upper_mask = x_range >= upper_threshold
        fig.add_trace(go.Scatter(
            x=x_range[upper_mask],
            y=kde_values[upper_mask],
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.4)',
            line=dict(width=0),
            name=f'Upper Tail 5% (E[Gain]: {expected_gain:.2f}%)',
            showlegend=True
        ))
    
    # VaR line
    fig.add_vline(
        x=var_val * 100,
        line_dash="dash",
        line_color="#FF4500",
        annotation_text=f"VaR 95%: {var_val*100:.2f}%",
        annotation_position="top left"
    )
    
    # CVaR line
    fig.add_vline(
        x=cvar_val * 100,
        line_dash="dot",
        line_color="#FF0000",
        annotation_text=f"CVaR 95%: {cvar_val*100:.2f}%",
        annotation_position="bottom left"
    )
    
    fig.update_layout(
        title=f'Rachev Ratio & VaR/CVaR - {frequency.title()} (R = {rachev_ratio:.2f})',
        xaxis_title=f'{frequency.title()} Return (%)',
        yaxis_title='Probability Density',
        template=PLOTLY_TEMPLATE,
        height=450
    )
    
    return fig


def create_aum_chart(fund_details, cnpj_standard):
    """Create AUM time series chart - handle duplicates by keeping max NR_COTST."""
    if fund_details is None:
        return None
    
    fund_data = fund_details[fund_details['CNPJ_STANDARD'] == cnpj_standard].sort_index()
    
    if len(fund_data) == 0:
        return None
    
    if 'VL_PATRIM_LIQ' not in fund_data.columns:
        return None
    
    # Handle duplicate dates - keep row with highest NR_COTST
    if 'NR_COTST' in fund_data.columns:
        fund_data = fund_data.reset_index()
        date_col = fund_data.columns[0]
        fund_data = fund_data.sort_values('NR_COTST', ascending=False)
        fund_data = fund_data.drop_duplicates(subset=[date_col], keep='first')
        fund_data = fund_data.set_index(date_col)
        fund_data = fund_data.sort_index()
    
    aum_series = fund_data['VL_PATRIM_LIQ'].dropna() / 1_000_000
    
    if len(aum_series) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=aum_series.index,
        y=aum_series.values,
        mode='lines',
        line=dict(color='#D4AF37', width=2),
        name='AUM',
        hovertemplate='R$ %{y:.2f}M<extra></extra>'
    ))
    
    fig.update_layout(
        title='Assets Under Management (AUM)',
        xaxis_title='Date',
        yaxis_title='AUM (R$ Millions)',
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return fig


def create_shareholders_chart(fund_details, cnpj_standard):
    """Create shareholders time series chart - handle duplicates by keeping max NR_COTST."""
    if fund_details is None:
        return None
    
    fund_data = fund_details[fund_details['CNPJ_STANDARD'] == cnpj_standard].sort_index()
    
    if len(fund_data) == 0:
        return None
    
    if 'NR_COTST' not in fund_data.columns:
        return None
    
    # Handle duplicate dates - keep row with highest NR_COTST
    fund_data = fund_data.reset_index()
    date_col = fund_data.columns[0]
    fund_data = fund_data.sort_values('NR_COTST', ascending=False)
    fund_data = fund_data.drop_duplicates(subset=[date_col], keep='first')
    fund_data = fund_data.set_index(date_col)
    fund_data = fund_data.sort_index()
    
    shareholders_series = fund_data['NR_COTST'].dropna()
    
    if len(shareholders_series) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=shareholders_series.index,
        y=shareholders_series.values,
        mode='lines',
        line=dict(color='#FFD700', width=2),
        name='Shareholders',
        hovertemplate='%{y:,.0f} shareholders<extra></extra>'
    ))
    
    fig.update_layout(
        title='Number of Shareholders',
        xaxis_title='Date',
        yaxis_title='Shareholders',
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return fig


def create_exposure_time_series_chart(copula_results, metric_name, last_value, avg_value, benchmark_name):
    """
    Create exposure time series chart with yellow line, red dot for last value, and blue line for average.
    
    Parameters:
    -----------
    copula_results : pd.DataFrame
        Results from estimate_rolling_copula_for_chart
    metric_name : str
        Column name ('kendall_tau', 'tail_lower', 'tail_upper', 'asymmetry_index')
    last_value : float
        Last window value
    avg_value : float
        Average value
    benchmark_name : str
        Name of benchmark
    
    Returns:
    --------
    plotly figure
    """
    metric_series = copula_results[metric_name]
    
    # Title mapping
    title_map = {
        'kendall_tau': 'Kendall Tau',
        'tail_lower': 'Lower Tail Dependence',
        'tail_upper': 'Upper Tail Dependence',
        'asymmetry_index': 'Asymmetry Index'
    }
    
    title = f'{title_map[metric_name]} - {benchmark_name}'
    
    fig = go.Figure()
    
    # Yellow solid line for time series
    fig.add_trace(go.Scatter(
        x=metric_series.index,
        y=metric_series.values,
        mode='lines',
        line=dict(color='#D4AF37', width=2),
        name='Time Series',
        hovertemplate='%{y:.4f}<extra></extra>'
    ))
    
    # Red dot for last value
    fig.add_trace(go.Scatter(
        x=[metric_series.index[-1]],
        y=[last_value],
        mode='markers',
        marker=dict(color='#FF0000', size=10),
        name=f'Last: {last_value:.4f}',
        hovertemplate='Last: %{y:.4f}<extra></extra>'
    ))
    
    # Blue horizontal line for average
    fig.add_hline(
        y=avg_value,
        line_dash="solid",
        line_color="#1E90FF",
        line_width=2,
        annotation_text=f"Avg: {avg_value:.4f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=title_map[metric_name],
        template=PLOTLY_TEMPLATE,
        height=350,
        showlegend=True
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def create_dendrogram_plot(linkage_matrix, asset_names):
    """Create dendrogram visualization."""
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    dendrogram(
        linkage_matrix,
        labels=asset_names,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
        color_threshold=0
    )
    
    ax.set_title('Hierarchical Clustering Dendrogram (Average Linkage)', 
                 fontsize=14, color='#D4AF37', pad=20)
    ax.set_xlabel('Assets', fontsize=12, color='#D4AF37')
    ax.set_ylabel('Distance', fontsize=12, color='#D4AF37')
    ax.tick_params(colors='#D4AF37')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#D4AF37')
    
    plt.tight_layout()
    return fig


def create_correlation_heatmap(corr_matrix, asset_names):
    """Create correlation heatmap."""
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=asset_names,
        y=asset_names,
        colorscale='RdYlGn',
        zmid=0,
        text=corr_matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Kendall Tau")
    ))
    
    fig.update_layout(
        title='Kendall Tau Correlation Matrix',
        template=PLOTLY_TEMPLATE,
        height=max(400, len(asset_names) * 20),
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    return fig

class PortfolioMetrics:
    """Portfolio risk and performance metrics."""
    
    TRADING_DAYS_PER_YEAR = 252
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.0):
        excess_returns = returns - risk_free_rate / PortfolioMetrics.TRADING_DAYS_PER_YEAR
        if excess_returns.std() == 0:
            return 0.0
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(PortfolioMetrics.TRADING_DAYS_PER_YEAR)
    
    @staticmethod
    def omega_ratio(returns, threshold=0.0):
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        if losses == 0:
            return np.inf if gains > 0 else 1.0
        return gains / losses
    
    @staticmethod
    def cagr(returns):
        cumulative_return = (1 + returns).prod()
        n_years = len(returns) / PortfolioMetrics.TRADING_DAYS_PER_YEAR
        if n_years == 0:
            return 0.0
        return cumulative_return ** (1 / n_years) - 1
    
    @staticmethod
    def annualized_volatility(returns):
        return returns.std() * np.sqrt(PortfolioMetrics.TRADING_DAYS_PER_YEAR)
    
    @staticmethod
    def var(returns, confidence=0.95):
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def cvar(returns, confidence=0.95):
        var_threshold = PortfolioMetrics.var(returns, confidence)
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) == 0:
            return var_threshold
        return tail_losses.mean()
    
    @staticmethod
    def drawdown_series(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        return (cumulative - running_max) / running_max
    
    @staticmethod
    def max_drawdown(returns):
        return PortfolioMetrics.drawdown_series(returns).min()
    
    @staticmethod
    def cdar(returns, confidence=0.95):
        drawdowns = PortfolioMetrics.drawdown_series(returns)
        threshold = np.percentile(drawdowns, (1 - confidence) * 100)
        tail_drawdowns = drawdowns[drawdowns <= threshold]
        if len(tail_drawdowns) == 0:
            return threshold
        return tail_drawdowns.mean()
    
    @staticmethod
    def information_ratio(returns, benchmark_returns):
        excess_ret = returns - benchmark_returns
        if excess_ret.std() == 0:
            return 0.0
        return (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
    
    @staticmethod
    def rachev_ratio(returns, alpha=0.05):
        """Calculate Rachev ratio: ratio of expected loss in worst alpha% to expected gain in best alpha%."""
        n = len(returns)
        if n == 0:
            return np.nan
        tail_size = max(1, int(n * alpha))
        sorted_returns = np.sort(returns.values)
        # Lower tail (worst returns - losses)
        lower_tail = sorted_returns[:tail_size]
        expected_loss = -np.mean(lower_tail)  # Make positive for ratio
        # Upper tail (best returns - gains)
        upper_tail = sorted_returns[-tail_size:]
        expected_gain = np.mean(upper_tail)
        if expected_gain == 0:
            return np.inf if expected_loss > 0 else 1.0
        return expected_gain / np.abs(expected_loss)


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDED PORTFOLIO HELPER FUNCTIONS  
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_portfolio_returns(fund_returns_dict, allocations):
    """Calculate portfolio returns based on weighted average of fund returns."""
    total_alloc = sum(allocations.values())
    if total_alloc == 0:
        return None
    weights = {k: v / total_alloc for k, v in allocations.items()}
    all_dates = None
    for fund_name, returns in fund_returns_dict.items():
        if fund_name in weights:
            if all_dates is None:
                all_dates = set(returns.index)
            else:
                all_dates = all_dates.intersection(set(returns.index))
    if not all_dates:
        return None
    all_dates = sorted(list(all_dates))
    returns_df = pd.DataFrame(index=all_dates)
    for fund_name, weight in weights.items():
        if fund_name in fund_returns_dict:
            returns_df[fund_name] = fund_returns_dict[fund_name].reindex(all_dates)
    portfolio_returns = pd.Series(0.0, index=all_dates)
    for fund_name, weight in weights.items():
        if fund_name in returns_df.columns:
            portfolio_returns += returns_df[fund_name].fillna(0) * weight
    return portfolio_returns


def create_portfolio_template():
    return pd.DataFrame({'Fund Name': ['Fund 1', 'Fund 2', 'Fund 3'], 'Allocation (%)': [40.0, 35.0, 25.0]})


def create_monthly_returns_comparison_table(fund_returns_dict, cdi_returns, last_n_months=12):
    max_date = None
    for returns in fund_returns_dict.values():
        if len(returns) > 0:
            fund_max = returns.index.max()
            if max_date is None or fund_max > max_date:
                max_date = fund_max
    if max_date is None:
        return None, None
    current_month_end = max_date + pd.offsets.MonthEnd(0)
    start_date = (current_month_end - pd.DateOffset(months=last_n_months-1)).replace(day=1)
    months = pd.date_range(start=start_date, end=current_month_end, freq='M')
    cdi_monthly_returns = {}
    for month_end in months:
        month_start = month_end.replace(day=1)
        actual_end = min(month_end, max_date)
        cdi_month = cdi_returns[(cdi_returns.index >= month_start) & (cdi_returns.index <= actual_end)]
        cdi_monthly_returns[month_end.strftime('%b/%Y')] = (1 + cdi_month).prod() - 1 if len(cdi_month) > 0 else 0
    table_data = []
    for fund_name in sorted(fund_returns_dict.keys()):
        returns = fund_returns_dict[fund_name]
        row = {'Fund': fund_name}
        for month_end in months:
            month_start = month_end.replace(day=1)
            actual_end = min(month_end, max_date)
            month_returns = returns[(returns.index >= month_start) & (returns.index <= actual_end)]
            row[month_end.strftime('%b/%Y')] = (1 + month_returns).prod() - 1 if len(month_returns) > 0 else np.nan
        table_data.append(row)
    cdi_row = {'Fund': 'CDI'}
    for month_end in months:
        cdi_row[month_end.strftime('%b/%Y')] = cdi_monthly_returns.get(month_end.strftime('%b/%Y'), np.nan)
    table_data.append(cdi_row)
    return pd.DataFrame(table_data), cdi_monthly_returns


def create_cumulative_returns_comparison_table(fund_returns_dict, cdi_returns):
    periods = {'Last 5 Days': 5, 'Current Month': 'MTD', '3 Months': 63, '6 Months': 126, '12 Months': 252}
    max_date = None
    for returns in fund_returns_dict.values():
        if len(returns) > 0:
            fund_max = returns.index.max()
            if max_date is None or fund_max > max_date:
                max_date = fund_max
    if max_date is None:
        return None, None
    cdi_period_returns = {}
    for period_name, period_val in periods.items():
        if period_val == 'MTD':
            cdi_period = cdi_returns[(cdi_returns.index >= max_date.replace(day=1)) & (cdi_returns.index <= max_date)]
        else:
            cdi_period = cdi_returns.tail(period_val)
        cdi_period_returns[period_name] = (1 + cdi_period).prod() - 1 if len(cdi_period) > 0 else 0
    table_data = []
    for fund_name in sorted(fund_returns_dict.keys()):
        returns = fund_returns_dict[fund_name]
        row = {'Fund': fund_name}
        for period_name, period_val in periods.items():
            if period_val == 'MTD':
                period_returns = returns[(returns.index >= max_date.replace(day=1)) & (returns.index <= max_date)]
            else:
                period_returns = returns.tail(period_val)
            row[period_name] = (1 + period_returns).prod() - 1 if len(period_returns) > 0 else np.nan
        table_data.append(row)
    cdi_row = {'Fund': 'CDI'}
    for period_name in periods.keys():
        cdi_row[period_name] = cdi_period_returns.get(period_name, np.nan)
    table_data.append(cdi_row)
    return pd.DataFrame(table_data), cdi_period_returns


def style_returns_table_with_colors(df, cdi_returns_dict):
    html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 12px; border: 2px solid #D4AF37;">'
    html += '<thead><tr style="background-color: #D4AF37; color: #000; font-weight: 700;">'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; text-align: left; position: sticky; left: 0; background: #D4AF37; z-index: 1;">Fund</th>'
    for col in df.columns[1:]:
        html += f'<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        fund_name, is_cdi = row['Fund'], row['Fund'] == 'CDI'
        html += '<tr style="background: #1a1a1a;">'
        html += f'<td style="padding: 10px; border: 1px solid #333; color: #D4AF37; font-weight: {"700" if is_cdi else "400"}; position: sticky; left: 0; background: #1a1a1a; z-index: 1;">{fund_name}</td>'
        for col in df.columns[1:]:
            val, cdi_val = row[col], cdi_returns_dict.get(col, 0)
            if pd.isna(val):
                fv, color = '-', '#888'
            else:
                fv = f"{val*100:.2f}%"
                color = '#FFF' if is_cdi else ('#F44' if val < 0 else ('#48F' if val <= cdi_val else '#FFF'))
            html += f'<td style="padding: 10px; border: 1px solid #333; color: {color}; text-align: right; font-weight: {"700" if is_cdi else "400"};">{fv}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html


def style_returns_table_relative(df, cdi_returns_dict):
    """Style table for relative performance (vs CDI). Values shown as percentage of CDI."""
    html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 12px; border: 2px solid #D4AF37;">'
    html += '<thead><tr style="background-color: #D4AF37; color: #000; font-weight: 700;">'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; text-align: left; position: sticky; left: 0; background: #D4AF37; z-index: 1;">Fund</th>'
    for col in df.columns[1:]:
        html += f'<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        fund_name = row['Fund']
        is_cdi = fund_name == 'CDI'
        html += '<tr style="background: #1a1a1a;">'
        html += f'<td style="padding: 10px; border: 1px solid #333; color: #D4AF37; font-weight: {"700" if is_cdi else "400"}; position: sticky; left: 0; background: #1a1a1a; z-index: 1;">{fund_name}</td>'
        for col in df.columns[1:]:
            val = row[col]
            if pd.isna(val):
                fv, color = '-', '#888'
            else:
                fv = f"{val*100:.1f}%"
                if is_cdi:
                    color = '#FFF'
                elif val > 1.0:
                    color = '#FFF'  # Outperformed CDI
                elif val >= 0:
                    color = '#48F'  # 0-100% of CDI
                else:
                    color = '#F44'  # Negative relative performance
            html += f'<td style="padding: 10px; border: 1px solid #333; color: {color}; text-align: right; font-weight: {"700" if is_cdi else "400"};">{fv}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html


def style_book_analysis_table(df, period_cols):
    """Style table for Book Analysis with category grouping and contribution values."""
    html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 14px; border: 2px solid #D4AF37;">'
    html += '<thead><tr style="background-color: #D4AF37; color: #000; font-weight: 700;">'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; text-align: left; position: sticky; left: 0; background: #D4AF37; z-index: 1;">Fund</th>'
    for col in period_cols:
        html += f'<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">{col}</th>'
    html += '</tr></thead><tbody>'
    
    for _, row in df.iterrows():
        fund_name = row['Fund']
        is_total = 'TOTAL' in fund_name
        is_cdi = fund_name == '📈 CDI'  # Exact match for the CDI benchmark row
        is_category_total = fund_name.startswith('📁')
        is_portfolio_total = fund_name.startswith('📊')
        
        # Background color based on row type
        if is_portfolio_total:
            bg_color = '#2a2a2a'
            text_color = '#D4AF37'
            font_weight = '700'
        elif is_cdi:
            bg_color = '#1a1a1a'
            text_color = '#00CED1'
            font_weight = '700'
        elif is_category_total:
            bg_color = '#252525'
            text_color = '#FFA500'
            font_weight = '600'
        else:
            bg_color = '#1a1a1a'
            text_color = '#FFF'
            font_weight = '400'
        
        html += f'<tr style="background: {bg_color};">'
        html += f'<td style="padding: 10px; border: 1px solid #333; color: {text_color}; font-weight: {font_weight}; position: sticky; left: 0; background: {bg_color}; z-index: 1;">{fund_name}</td>'
        
        for col in period_cols:
            val = row.get(col, np.nan)
            if pd.isna(val):
                fv, color = '-', '#888'
            else:
                fv = f"{val*100:.4f}%"
                if is_cdi:
                    color = '#00CED1'
                elif is_portfolio_total:
                    color = '#D4AF37'
                elif is_category_total:
                    color = '#FFA500'
                elif val < 0:
                    color = '#F44'  # Red for negative
                else:
                    color = '#FFF'  # White for positive/zero
            html += f'<td style="padding: 10px; border: 1px solid #333; color: {color}; text-align: right; font-weight: {font_weight};">{fv}</td>'
        html += '</tr>'
    
    html += '</tbody></table></div>'
    return html


def style_sortable_returns_table(df, cdi_returns_dict, sort_col=None, sort_ascending=True):
    """Style table with sorting applied. Returns HTML and sorted dataframe."""
    # Sort dataframe if sort column specified
    if sort_col and sort_col in df.columns:
        df_sorted = df.copy()
        # Convert to numeric for sorting
        df_sorted[sort_col] = pd.to_numeric(df_sorted[sort_col], errors='coerce')
        df_sorted = df_sorted.sort_values(by=sort_col, ascending=sort_ascending, na_position='last')
    else:
        df_sorted = df
    
    html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 12px; border: 2px solid #D4AF37;">'
    html += '<thead><tr style="background-color: #D4AF37; color: #000; font-weight: 700;">'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; text-align: left; position: sticky; left: 0; background: #D4AF37; z-index: 1;">Fund</th>'
    for col in df.columns[1:]:
        html += f'<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">{col}</th>'
    html += '</tr></thead><tbody>'
    
    for _, row in df_sorted.iterrows():
        fund_name = row['Fund']
        is_cdi = fund_name == 'CDI'
        html += '<tr style="background: #1a1a1a;">'
        html += f'<td style="padding: 10px; border: 1px solid #333; color: #D4AF37; font-weight: {"700" if is_cdi else "400"}; position: sticky; left: 0; background: #1a1a1a; z-index: 1;">{fund_name}</td>'
        for col in df.columns[1:]:
            val = row[col]
            cdi_val = cdi_returns_dict.get(col, 0)
            if pd.isna(val):
                fv, color = '-', '#888'
            else:
                fv = f"{val*100:.2f}%"
                color = '#FFF' if is_cdi else ('#F44' if val < 0 else ('#48F' if val <= cdi_val else '#FFF'))
            html += f'<td style="padding: 10px; border: 1px solid #333; color: {color}; text-align: right; font-weight: {"700" if is_cdi else "400"};">{fv}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html


def style_sortable_relative_table(df, sort_col=None, sort_ascending=True):
    """Style relative performance table with sorting applied."""
    if sort_col and sort_col in df.columns:
        df_sorted = df.copy()
        df_sorted[sort_col] = pd.to_numeric(df_sorted[sort_col], errors='coerce')
        df_sorted = df_sorted.sort_values(by=sort_col, ascending=sort_ascending, na_position='last')
    else:
        df_sorted = df
    
    html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 12px; border: 2px solid #D4AF37;">'
    html += '<thead><tr style="background-color: #D4AF37; color: #000; font-weight: 700;">'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; text-align: left; position: sticky; left: 0; background: #D4AF37; z-index: 1;">Fund</th>'
    for col in df.columns[1:]:
        html += f'<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">{col}</th>'
    html += '</tr></thead><tbody>'
    
    for _, row in df_sorted.iterrows():
        fund_name = row['Fund']
        is_cdi = fund_name == 'CDI'
        html += '<tr style="background: #1a1a1a;">'
        html += f'<td style="padding: 10px; border: 1px solid #333; color: #D4AF37; font-weight: {"700" if is_cdi else "400"}; position: sticky; left: 0; background: #1a1a1a; z-index: 1;">{fund_name}</td>'
        for col in df.columns[1:]:
            val = row[col]
            if pd.isna(val):
                fv, color = '-', '#888'
            else:
                fv = f"{val*100:.1f}%"
                if is_cdi:
                    color = '#FFF'
                elif val > 1.0:
                    color = '#FFF'
                elif val >= 0:
                    color = '#48F'
                else:
                    color = '#F44'
            html += f'<td style="padding: 10px; border: 1px solid #333; color: {color}; text-align: right; font-weight: {"700" if is_cdi else "400"};">{fv}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html


@st.cache_data(ttl=3600, show_spinner=False)
def get_fund_returns_by_name_cached(fund_name, fund_metrics_hash, fund_details_hash, _fund_metrics, _fund_details):
    """Cached version of get_fund_returns_by_name for performance."""
    if _fund_metrics is None or _fund_details is None:
        return None
    fund_row = _fund_metrics[_fund_metrics['FUNDO DE INVESTIMENTO'] == fund_name]
    if len(fund_row) == 0:
        return None
    cnpj_standard = standardize_cnpj(fund_row.iloc[0]['CNPJ'])
    if cnpj_standard is None:
        return None
    result = get_fund_returns(_fund_details, cnpj_standard, period_months=None)
    return result[1] if result else None


def get_fund_returns_by_name(fund_name, fund_metrics, fund_details):
    if fund_metrics is None or fund_details is None:
        return None
    # Use cached version with hash-based keys
    try:
        metrics_hash = hash(tuple(fund_metrics['FUNDO DE INVESTIMENTO'].head(10).tolist()))
        details_hash = hash(len(fund_details)) if fund_details is not None else 0
        return get_fund_returns_by_name_cached(fund_name, metrics_hash, details_hash, fund_metrics, fund_details)
    except:
        # Fallback to non-cached version
        fund_row = fund_metrics[fund_metrics['FUNDO DE INVESTIMENTO'] == fund_name]
        if len(fund_row) == 0:
            return None
        cnpj_standard = standardize_cnpj(fund_row.iloc[0]['CNPJ'])
        if cnpj_standard is None:
            return None
        result = get_fund_returns(fund_details, cnpj_standard, period_months=None)
        return result[1] if result else None



def show_dro_configuration_panel():
    """Show advanced configuration panel for Wasserstein DRO optimizer V2."""
    
    st.markdown("### ⚙️ Advanced DRO Configuration")
    st.info("💡 Configure state-of-the-art Wasserstein DRO parameters. Hover over labels for explanations.")
    
    with st.expander("🔧 Core DRO Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            wasserstein_order = st.selectbox(
                "Wasserstein Metric Type",
                options=[1, 2],
                index=1,
                help="Type-1: More robust to outliers. Type-2: Standard, theoretical guarantees.",
                key="dro_wasserstein_order"
            )
            
            radius_method = st.selectbox(
                "Radius Selection Method",
                options=['rwpi', 'cv', 'bootstrap', 'manual'],
                index=0,
                help="""
                - RWPI (Recommended): Robust Wasserstein Profile Inference - data-driven, theoretically grounded
                - CV: Cross-validation based selection
                - Bootstrap: Bootstrap-based estimation
                - Manual: Set radius manually
                """,
                key="dro_radius_method"
            )
            
            if radius_method == 'manual':
                radius_manual = st.number_input(
                    "Manual Radius Value",
                    min_value=0.0001,
                    max_value=1.0,
                    value=0.01,
                    step=0.001,
                    format="%.4f",
                    help="Wasserstein ball radius - larger values = more conservative",
                    key="dro_radius_manual"
                )
            else:
                radius_manual = None
            
            if radius_method == 'rwpi':
                rwpi_confidence = st.slider(
                    "RWPI Confidence Level",
                    min_value=0.80,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    help="Higher confidence = more conservative portfolio",
                    key="dro_rwpi_confidence"
                )
            else:
                rwpi_confidence = 0.95
        
        with col2:
            covariance_method = st.selectbox(
                "Covariance Estimation",
                options=['ledoit_wolf', 'oas', 'sample'],
                index=0,
                help="""
                - Ledoit-Wolf: Shrinkage estimator, recommended for most cases
                - OAS: Oracle Approximating Shrinkage
                - Sample: Standard sample covariance
                """,
                key="dro_covariance_method"
            )
            
            scenario_reduction = st.selectbox(
                "Scenario Reduction Method",
                options=['fast_forward', 'kmeans', 'none'],
                index=0,
                help="Reduce computational complexity while maintaining distribution properties",
                key="dro_scenario_reduction"
            )
            
            max_scenarios = st.number_input(
                "Max Scenarios (after reduction)",
                min_value=50,
                max_value=500,
                value=200,
                step=10,
                help="Maximum number of scenarios after reduction",
                key="dro_max_scenarios"
            )
            
            solver = st.selectbox(
                "Optimization Solver",
                options=['CLARABEL', 'MOSEK', 'OSQP', 'SCS'],
                index=0,
                help="CLARABEL: Fast, reliable. MOSEK: Commercial, very fast (requires license)",
                key="dro_solver"
            )
    
    with st.expander("📊 Statistical Validation Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            compute_deflated_sharpe = st.checkbox(
                "Compute Deflated Sharpe Ratio",
                value=True,
                help="Adjusts Sharpe ratio for multiple testing and selection bias",
                key="dro_deflated_sharpe"
            )
            
            compute_pbo = st.checkbox(
                "Compute Probability of Backtest Overfitting",
                value=True,
                help="Estimates probability that performance is due to overfitting. High PBO (>0.5) is bad.",
                key="dro_compute_pbo"
            )
        
        with col2:
            train_ratio = st.slider(
                "Training Data Ratio",
                min_value=0.40,
                max_value=0.80,
                value=0.60,
                step=0.05,
                help="Portion of data for training",
                key="dro_train_ratio"
            )
            
            validation_ratio = st.slider(
                "Validation Data Ratio",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                help="Portion of data for validation",
                key="dro_validation_ratio"
            )
            
            # Test ratio is automatic
            test_ratio = 1.0 - train_ratio - validation_ratio
            st.metric("Test Data Ratio", f"{test_ratio:.2%}")
    
    with st.expander("🔍 Advanced Solver Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            solver_verbose = st.checkbox(
                "Verbose Solver Output",
                value=True,
                help="Show detailed solver progress",
                key="dro_solver_verbose"
            )
            
            solver_tolerance = st.select_slider(
                "Solver Tolerance",
                options=[1e-6, 1e-7, 1e-8, 1e-9],
                value=1e-8,
                format_func=lambda x: f"{x:.0e}",
                help="Convergence tolerance - smaller = more accurate",
                key="dro_tolerance"
            )
        
        with col2:
            solver_max_iters = st.number_input(
                "Max Solver Iterations",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=500,
                help="Maximum iterations for solver",
                key="dro_max_iters"
            )
            
            
    # Build configuration object
    config = WassersteinDROConfig(
        wasserstein_order=wasserstein_order,
        radius_method=radius_method,
        radius_manual=radius_manual,
        rwpi_confidence=rwpi_confidence,
        covariance_method=covariance_method,
        scenario_reduction_method=scenario_reduction,
        max_scenarios=max_scenarios,
        solver=solver,
        solver_verbose=solver_verbose,
        solver_max_iters=solver_max_iters,
        solver_tolerance=solver_tolerance,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        compute_deflated_sharpe=compute_deflated_sharpe,
        compute_pbo=compute_pbo,
        verbose=True
    )
    
    return config


def display_dro_results_v2(result, all_returns_df, cdi_returns, fund_categories, fund_subcategories):
    """Display comprehensive DRO optimization results with V2 metrics."""
    
    if not result.success:
        st.error(f"❌ Optimization failed: {result.solver_status}")
        with st.expander("📋 Optimization Log"):
            for log_entry in result.optimization_log:
                st.text(log_entry)
        return
    
    st.success(f"✅ DRO Optimization complete in {result.computation_time:.2f}s!")
    
    # === STATISTICAL VALIDATION METRICS (PROMINENT) ===
    if result.deflated_sharpe_ratio is not None or result.pbo_score is not None:
        st.markdown("### 📊 Statistical Validation")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            if result.deflated_sharpe_ratio is not None:
                # Color code based on value
                if result.deflated_sharpe_ratio > 1.0:
                    color = "normal"
                elif result.deflated_sharpe_ratio > 0.5:
                    color = "off"
                else:
                    color = "inverse"
                st.metric(
                    "🎯 Deflated Sharpe Ratio",
                    f"{result.deflated_sharpe_ratio:.3f}",
                    help="Sharpe ratio adjusted for selection bias and non-normality. Values > 1.0 are good."
                )
            else:
                st.info("Deflated Sharpe not computed")
        
        with stat_col2:
            if result.pbo_score is not None:
                pbo_warning = "⚠️ HIGH RISK" if result.pbo_score > 0.5 else "✅ Low Risk"
                delta_color = "inverse" if result.pbo_score > 0.5 else "normal"
                st.metric(
                    "📈 Prob. of Backtest Overfitting",
                    f"{result.pbo_score:.1%}",
                    delta=pbo_warning,
                    delta_color=delta_color,
                    help="Probability that performance is due to overfitting. >50% is concerning!"
                )
            else:
                st.info("PBO not computed")
        
        with stat_col3:
            # Interpretation
            if result.pbo_score is not None and result.deflated_sharpe_ratio is not None:
                if result.pbo_score <= 0.3 and result.deflated_sharpe_ratio > 0.8:
                    st.success("✅ **Good**: Low overfitting risk, solid risk-adjusted returns")
                elif result.pbo_score <= 0.5 and result.deflated_sharpe_ratio > 0.5:
                    st.info("ℹ️ **Moderate**: Acceptable but monitor out-of-sample")
                else:
                    st.warning("⚠️ **Caution**: High overfitting risk or weak adjusted returns")
        
        st.markdown("---")
    
    # === OPTIMIZATION DETAILS ===
    with st.expander("🔍 Optimization Details", expanded=False):
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.metric("Solver Status", result.solver_status)
            st.metric("Computation Time", f"{result.computation_time:.2f}s")
            st.metric("Objective Value", f"{result.objective_value:.6f}")
        
        with detail_col2:
            st.metric("Wasserstein Radius", f"{result.wasserstein_radius:.6e}")
            st.metric("Scenarios Used", result.n_scenarios_used)
            if result.covariance_shrinkage:
                st.metric("Covariance Shrinkage", f"{result.covariance_shrinkage:.4f}")
        
        with detail_col3:
            if result.deflated_sharpe_ratio is not None:
                st.metric(
                    "Deflated Sharpe Ratio",
                    f"{result.deflated_sharpe_ratio:.3f}",
                    help="Sharpe ratio adjusted for selection bias and non-normality"
                )
            if result.pbo_score is not None:
                delta_text = "⚠️ High" if result.pbo_score > 0.5 else "✅ Low"
                st.metric(
                    "PBO Score",
                    f"{result.pbo_score:.3f}",
                    delta=delta_text,
                    help="Probability of Backtest Overfitting. >0.5 suggests overfitting!"
                )
        
        # Show optimization log
        st.markdown("#### Optimization Log")
        log_text = "\n".join(result.optimization_log)
        st.text_area("Log Output", log_text, height=200, key="opt_log_display")
    
    st.markdown("---")
    
    # === PERFORMANCE COMPARISON: IN-SAMPLE vs OUT-OF-SAMPLE ===
    st.markdown("### 📊 Performance Analysis: In-Sample vs Out-of-Sample")
    
    # Create comparison table
    comparison_data = {
        'Metric': [],
        'In-Sample': [],
        'Validation': [],
        'Test (OOS)': []
    }
    
    metrics_to_compare = [
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('annual_return', 'Annual Return'),
        ('annual_volatility', 'Annual Volatility'),
        ('max_drawdown', 'Max Drawdown'),
        ('cvar_95', 'CVaR (95%)'),
        ('omega_ratio', 'Omega Ratio'),
    ]
    
    for metric_key, metric_label in metrics_to_compare:
        comparison_data['Metric'].append(metric_label)
        
        # Format percentages
        is_percentage = 'return' in metric_key.lower() or 'volatility' in metric_key.lower() or 'drawdown' in metric_key.lower() or 'cvar_95' in metric_key.lower()
        
        for dataset, dataset_metrics in [
            ('In-Sample', result.in_sample_metrics),
            ('Validation', result.validation_metrics),
            ('Test (OOS)', result.test_metrics)
        ]:
            value = dataset_metrics.get(metric_key, 0)
            if is_percentage:
                comparison_data[dataset].append(f"{value * 100:.2f}%")
            else:
                comparison_data[dataset].append(f"{value:.3f}")
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df.style.set_properties(**{
            'background-color': '#1a1a1a',
            'color': '#D4AF37',
            'border-color': '#D4AF37'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Performance highlights
    highlight_col1, highlight_col2, highlight_col3, highlight_col4 = st.columns(4)
    
    with highlight_col1:
        st.metric(
            "Test Sharpe Ratio",
            f"{result.test_metrics.get('sharpe_ratio', 0):.3f}",
            delta=f"{(result.test_metrics.get('sharpe_ratio', 0) - result.in_sample_metrics.get('sharpe_ratio', 0)):.3f}"
        )
    
    with highlight_col2:
        st.metric(
            "Test Annual Return",
            f"{result.test_metrics.get('annual_return', 0) * 100:.2f}%",
            delta=f"{(result.test_metrics.get('annual_return', 0) - result.in_sample_metrics.get('annual_return', 0)) * 100:.2f}%"
        )
    
    with highlight_col3:
        st.metric(
            "Test Volatility",
            f"{result.test_metrics.get('annual_volatility', 0) * 100:.2f}%",
            delta=f"{(result.test_metrics.get('annual_volatility', 0) - result.in_sample_metrics.get('annual_volatility', 0)) * 100:.2f}%",
            delta_color="inverse"
        )
    
    with highlight_col4:
        st.metric(
            "Test Max Drawdown",
            f"{result.test_metrics.get('max_drawdown', 0) * 100:.2f}%",
            delta=f"{(result.test_metrics.get('max_drawdown', 0) - result.in_sample_metrics.get('max_drawdown', 0)) * 100:.2f}%",
        )
    
    # Interpretation guide
    with st.expander("📖 How to Interpret Out-of-Sample Performance"):
        st.markdown("""
        **Good signs:**
        - Test Sharpe Ratio close to or higher than In-Sample
        - Deflated Sharpe Ratio > 1.5 (statistically significant)
        - PBO Score < 0.5 (low overfitting probability)
        - Stable performance across validation and test sets
        
        **Warning signs:**
        - Test Sharpe significantly lower than In-Sample (>30% degradation)
        - PBO Score > 0.5 (suggests overfitting)
        - Large performance gaps between validation and test
        
        **This optimizer uses:**
        - Proper train/validation/test splits
        - Deflated Sharpe Ratio (adjusts for selection bias)
        - PBO score (quantifies overfitting risk)
        - Robust Wasserstein DRO (distribution-free guarantees)
        """)
    
    st.markdown("---")
    
    # === PORTFOLIO WEIGHTS ===
    st.markdown("### 🎯 Final Portfolio Weights")
    
    weights_series = result.weights
    weights_series = weights_series[weights_series > 1e-6].sort_values(ascending=False)
    
    weights_display = pd.DataFrame({
        'Fund': weights_series.index,
        'Weight %': weights_series.values * 100,
        'Category': [fund_categories.get(f, 'Unknown') for f in weights_series.index],
        'Subcategory': [fund_subcategories.get(f, 'Unknown') for f in weights_series.index]
    })
    
    st.dataframe(
        weights_display.style.format({'Weight %': '{:.2f}%'}).background_gradient(
            cmap='YlOrBr',
            subset=['Weight %']
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # Portfolio concentration metrics
    concentration_col1, concentration_col2, concentration_col3 = st.columns(3)
    
    with concentration_col1:
        n_holdings = len(weights_series)
        st.metric("Number of Holdings", n_holdings)
    
    with concentration_col2:
        effective_n = 1 / np.sum(weights_series.values**2)
        st.metric("Effective N° of Assets", f"{effective_n:.1f}")
    
    with concentration_col3:
        top5_weight = weights_series.nlargest(5).sum() * 100
        st.metric("Top 5 Concentration", f"{top5_weight:.1f}%")
    
    # Store results in session state
    portfolio_returns_series = pd.Series(
        all_returns_df.values @ result.weights.values,
        index=all_returns_df.index
    )
    
    st.session_state['portfolio_result'] = {
        'final_weights': weights_series,
        'portfolio_returns': portfolio_returns_series,
        'dro_result': result,  # Store full result object
        'success': True
    }
    st.session_state['portfolio_returns_df'] = all_returns_df
    st.session_state['portfolio_cdi'] = cdi_returns
    st.session_state['fund_categories'] = fund_categories
    st.session_state['fund_subcategories'] = fund_subcategories


def main():
    # Check authentication
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated']:
        login_page()
        return  # Stop execution until logged in
    
    # Add logout button to sidebar
    logout_button()

    # Sidebar - Data Source Selection
    with st.sidebar:
        st.image("https://aquamarine-worthy-zebra-762.mypinata.cloud/ipfs/bafybeigayrnnsuwglzkbhikm32ksvucxecuorcj4k36l4de7na6wcdpjsa", 
                use_container_width=True)
        st.markdown("---")
        
        st.header("📁 Data Source")
        
        # Determine default data source based on what's configured
        default_source_index = 0  # GitHub Releases
        if not is_github_configured():
            if os.path.exists(DEFAULT_METRICS_PATH) or os.path.exists(DEFAULT_DETAILS_PATH):
                default_source_index = 1  # Local Files
            else:
                default_source_index = 2  # Upload
        
        # Data source selection
        data_source = st.radio(
            "Load data from:",
            options=['📦 GitHub Releases', '📂 Local Files', '📤 Upload'],
            index=default_source_index,
            help="GitHub: Cloud storage via releases | Local: Load from disk | Upload: Upload files manually"
        )
        
        uploaded_metrics = None
        uploaded_details = None
        uploaded_benchmarks = None
        fund_metrics = None
        fund_details = None
        benchmarks = None
        
        if data_source == '📦 GitHub Releases':
            # Use the GitHub panel which handles everything
            fund_metrics, fund_details, benchmarks = render_github_data_panel()
            
            # Process fund_metrics if loaded
            if fund_metrics is not None:
                fund_metrics = fund_metrics.replace('n/a', np.nan)
                if 'CNPJ' in fund_metrics.columns:
                    fund_metrics['CNPJ_STANDARD'] = fund_metrics['CNPJ'].apply(standardize_cnpj)
            
            # Process fund_details if loaded
            if fund_details is not None and 'CNPJ_FUNDO' in fund_details.columns:
                fund_details['CNPJ_STANDARD'] = fund_details['CNPJ_FUNDO'].apply(standardize_cnpj)
        
        elif data_source == '📂 Local Files':
            st.info("📂 Using local files...")
            files_found = []
            if os.path.exists(DEFAULT_METRICS_PATH):
                files_found.append(f"✓ {DEFAULT_METRICS_PATH.split('/')[-1]}")
            if os.path.exists(DEFAULT_DETAILS_PATH):
                files_found.append(f"✓ {DEFAULT_DETAILS_PATH.split('/')[-1]}")
            if os.path.exists(DEFAULT_BENCHMARKS_PATH):
                files_found.append(f"✓ {DEFAULT_BENCHMARKS_PATH.split('/')[-1]}")
            
            if files_found:
                for f in files_found:
                    st.success(f)
            else:
                st.warning("No local files found")
        
        else:  # Upload
            uploaded_metrics = st.file_uploader(
                "Fund Metrics (xlsx/pkl)", 
                type=['xlsx', 'pkl'],
                help="Upload fund_metrics file"
            )
            uploaded_details = st.file_uploader(
                "Fund Details (pkl)",
                type=['pkl'],
                help="Upload funds_info.pkl"
            )
            uploaded_benchmarks = st.file_uploader(
                "Benchmarks (xlsx/pkl)",
                type=['xlsx', 'pkl'],
                help="Upload benchmarks_data file"
            )
        
        st.markdown("---")
    
    # Load data for non-GitHub sources (GitHub already loaded above)
    if data_source == '📂 Local Files':
        # Load from local paths
        fund_metrics = load_fund_data(file_path=DEFAULT_METRICS_PATH if os.path.exists(DEFAULT_METRICS_PATH) else None)
        fund_details = load_fund_details(file_path=DEFAULT_DETAILS_PATH if os.path.exists(DEFAULT_DETAILS_PATH) else None)
        benchmarks = load_benchmarks(file_path=DEFAULT_BENCHMARKS_PATH if os.path.exists(DEFAULT_BENCHMARKS_PATH) else None)
    
    elif data_source == '📤 Upload':
        # Load from uploaded files
        fund_metrics = load_fund_data(uploaded_file=uploaded_metrics)
        fund_details = load_fund_details(uploaded_file=uploaded_details)
        benchmarks = load_benchmarks(uploaded_file=uploaded_benchmarks)
    
    if fund_metrics is None:
        st.title("🏆 PROFESSIONAL FUND ANALYTICS PLATFORM")
        st.markdown("### Configure data source in the sidebar to begin")
        
        if data_source == '📦 GitHub Releases' and not is_github_configured():
            st.info("""
            **To use GitHub Releases:**
            1. Create a Personal Access Token at [GitHub Settings](https://github.com/settings/tokens)
            2. Create a Release in your repository with tag `data`
            3. Add credentials to Streamlit secrets
            
            See the sidebar for detailed setup instructions.
            """)
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 FUND DATABASE", "📊 DETAILED ANALYSIS", "🔍 ADVANCED COMPARISON", "🎯 PORTFOLIO CONSTRUCTION", "💼 RECOMMENDED PORTFOLIO"])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: FUND DATABASE
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.title("📋 INVESTMENT FUNDS DATABASE")
        st.markdown("---")
        
        display_cols = ['FUNDO DE INVESTIMENTO', 'CNPJ', 'GESTOR', 'CATEGORIA BTG', 
                       'SUBCATEGORIA BTG', 'STATUS', 'LAST_UPDATE', 'VL_PATRIM_LIQ', 
                       'NR_COTST']
        
        fund_list = fund_metrics[display_cols].copy()
        
        if 'VL_PATRIM_LIQ' in fund_list.columns:
            fund_list['VL_PATRIM_LIQ'] = fund_list['VL_PATRIM_LIQ'].apply(
                lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "N/A"
            )
        
        if 'NR_COTST' in fund_list.columns:
            fund_list['NR_COTST'] = fund_list['NR_COTST'].apply(
                lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
            )
        
        st.dataframe(fund_list, use_container_width=True, height=600)
        st.info("💡 Navigate to 'DETAILED ANALYSIS' tab to explore individual fund performance")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: DETAILED FUND ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.title("📊 DETAILED FUND ANALYSIS")
        st.markdown("---")
        
        # Fund selection
        fund_names = fund_metrics['FUNDO DE INVESTIMENTO'].tolist()
        fund_cnpjs = fund_metrics['CNPJ'].tolist()
        fund_mapping = dict(zip(fund_names, fund_cnpjs))
        
        selected_fund_name = st.selectbox(
            "Select a fund to analyze:",
            options=fund_names,
            key="fund_selector_detail"
        )
        
        selected_fund_cnpj = fund_mapping[selected_fund_name]
        selected_fund_cnpj_standard = standardize_cnpj(selected_fund_cnpj)
        
        if selected_fund_cnpj_standard:
            fund_info = fund_metrics[fund_metrics['CNPJ'] == selected_fund_cnpj].iloc[0]
            fund_name = fund_info['FUNDO DE INVESTIMENTO']
            
            # ═══════════════════════════════════════════════════════════════════
            # BASIC FUND INFORMATION SECTION
            # ═══════════════════════════════════════════════════════════════════
            
            st.markdown("### 📋 Fund Information")
            
            # Create info display in columns
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            
            with info_col1:
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px;'>TAX ID (CNPJ)</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>{fund_info.get('CNPJ', 'N/A')}</p>", unsafe_allow_html=True)
                
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px; margin-top: 15px;'>INVESTMENT MANAGER</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>{fund_info.get('GESTOR', 'N/A')}</p>", unsafe_allow_html=True)
            
            with info_col2:
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px;'>CATEGORY</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>{fund_info.get('CATEGORIA BTG', 'N/A')}</p>", unsafe_allow_html=True)
                
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px; margin-top: 15px;'>SUB-CATEGORY</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>{fund_info.get('SUBCATEGORIA BTG', 'N/A')}</p>", unsafe_allow_html=True)
            
            with info_col3:
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px;'>STATUS</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>{fund_info.get('STATUS', 'N/A')}</p>", unsafe_allow_html=True)
                
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px; margin-top: 15px;'>LAST UPDATE</p>", unsafe_allow_html=True)
                last_update = fund_info.get('LAST_UPDATE', 'N/A')
                if pd.notna(last_update) and last_update != 'N/A':
                    st.markdown(f"<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>{pd.to_datetime(last_update).strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>N/A</p>", unsafe_allow_html=True)
            
            with info_col4:
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px;'>FUND SIZE (AUM)</p>", unsafe_allow_html=True)
                fund_size = fund_info.get('VL_PATRIM_LIQ', np.nan)
                if pd.notna(fund_size):
                    st.markdown(f"<p style='color: #FFFFFF; font-size: 14px; font-weight: 600; margin-top: 0px;'>R$ {fund_size:,.2f}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>N/A</p>", unsafe_allow_html=True)
                
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px; margin-top: 15px;'>SHAREHOLDERS</p>", unsafe_allow_html=True)
                shareholders = fund_info.get('NR_COTST', np.nan)
                if pd.notna(shareholders):
                    st.markdown(f"<p style='color: #FFFFFF; font-size: 14px; font-weight: 600; margin-top: 0px;'>{int(shareholders):,}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>N/A</p>", unsafe_allow_html=True)
            
            # Additional row for Tributação, Liquidez and Suitability
            info_col5, info_col6, info_col7, info_col8 = st.columns(4)
            
            with info_col5:
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px; margin-top: 15px;'>TAXATION</p>", unsafe_allow_html=True)
                tributacao = fund_info.get('TRIBUTAÇÃO', 'N/A')
                st.markdown(f"<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>{tributacao}</p>", unsafe_allow_html=True)
            
            with info_col6:
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px; margin-top: 15px;'>LIQUIDITY</p>", unsafe_allow_html=True)
                liquidez = fund_info.get('LIQUIDEZ', 'N/A')
                st.markdown(f"<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>{liquidez}</p>", unsafe_allow_html=True)

            with info_col7:
                st.markdown("<p style='color: #FFD700; font-weight: 700; font-size: 13px; margin-bottom: 2px; margin-top: 15px;'>SUITABILITY</p>", unsafe_allow_html=True)
                suitability = fund_info.get('SUITABILITY', 'N/A')
                st.markdown(f"<p style='color: #FFFFFF; font-size: 13px; margin-top: 0px;'>{suitability}</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Get fund returns for analysis
            returns_result = get_fund_returns(fund_details, selected_fund_cnpj_standard, period_months=None)
            
            # Store fund returns in session state for later use
            if returns_result is not None:
                fund_returns_filtered, fund_returns_full = returns_result
                if 'fund_returns_full' not in st.session_state:
                    st.session_state['fund_returns_full'] = {}
                st.session_state['fund_returns_full'][selected_fund_cnpj_standard] = fund_returns_full
            
            # ═══════════════════════════════════════════════════════════════════
            # RETURNS ANALYSIS SECTION
            # ═══════════════════════════════════════════════════════════════════
            
            st.markdown("### 📈 Returns Analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                period_map = {'3M': 3, '6M': 6, '12M': 12, '24M': 24, '36M': 36, 'All': None}
                period_list = list(period_map.keys())
                default_period_idx = period_list.index('All')
                selected_period = st.selectbox("Select Period:", period_list, index=default_period_idx)
            
            with col2:
                if benchmarks is not None:
                    benchmark_cols = benchmarks.columns.tolist()
                    # Set CDI and IBOVESPA as defaults if available
                    default_benches = []
                    if 'CDI' in benchmark_cols:
                        default_benches.append('CDI')
                    if 'IBOVESPA' in benchmark_cols:
                        default_benches.append('IBOVESPA')
                    if not default_benches:
                        default_benches = benchmark_cols[:1]
                    
                    selected_benchmarks = st.multiselect(
                        "Select Benchmarks:",
                        options=benchmark_cols,
                        default=default_benches
                    )
                else:
                    selected_benchmarks = []
            
            returns_result = get_fund_returns(fund_details, selected_fund_cnpj_standard, period_map[selected_period])
            
            if returns_result is not None:
                fund_returns_filtered, fund_returns_full = returns_result
                
                # Benchmark dict
                benchmark_dict = {}
                if benchmarks is not None:
                    for bench in selected_benchmarks:
                        if bench in benchmarks.columns:
                            benchmark_dict[bench] = benchmarks[bench]
                
                # Cumulative returns chart
                fig_returns = create_returns_chart(
                    fund_returns_filtered, 
                    benchmark_dict, 
                    fund_name,
                    selected_period
                )
                st.plotly_chart(fig_returns, use_container_width=True)
                
                # Returns comparison table
                st.markdown("#### Returns Comparison")
                returns_table = create_returns_comparison_table(
                    fund_info, fund_name, benchmarks, fund_returns_full, selected_benchmarks
                )
                st.dataframe(returns_table, use_container_width=True, hide_index=True)
                
                # Monthly statistics
                st.markdown("#### Monthly Statistics")
                
                # Calculate total months from fund returns
                total_months = len(fund_returns_full.resample('M').apply(lambda x: (1 + x).prod() - 1))
                
                # Get monthly return series for calculations
                monthly_returns = fund_returns_full.resample('M').apply(lambda x: (1 + x).prod() - 1)
                
                # Calculate months above/below zero
                months_above_zero = (monthly_returns > 0).sum()
                months_below_zero = (monthly_returns <= 0).sum()
                pct_above_zero = months_above_zero / total_months if total_months > 0 else 0
                pct_below_zero = months_below_zero / total_months if total_months > 0 else 0
                
                # Calculate months above/below benchmark (use first selected benchmark or CDI)
                if selected_benchmarks and benchmarks is not None:
                    bench_name = selected_benchmarks[0]
                    if bench_name in benchmarks.columns:
                        aligned_bench = benchmarks[bench_name].reindex(fund_returns_full.index, method='ffill').fillna(0)
                        monthly_bench = aligned_bench.resample('M').apply(lambda x: (1 + x).prod() - 1)
                        
                        months_above_bench = (monthly_returns > monthly_bench).sum()
                        months_below_bench = (monthly_returns <= monthly_bench).sum()
                        pct_above_bench = months_above_bench / total_months if total_months > 0 else 0
                        pct_below_bench = months_below_bench / total_months if total_months > 0 else 0
                    else:
                        months_above_bench = months_below_bench = np.nan
                        pct_above_bench = pct_below_bench = np.nan
                else:
                    months_above_bench = months_below_bench = np.nan
                    pct_above_bench = pct_below_bench = np.nan
                
                # Get best and worst month
                best_month = fund_info.get('BEST_MONTH', np.nan)
                worst_month = fund_info.get('WORST_MONTH', np.nan)
                
                # Create the monthly statistics dictionary with new format
                monthly_data = {
                    'Months Above Zero': f"{months_above_zero} ({pct_above_zero*100:.1f}%)",
                    'Months Below Zero': f"{months_below_zero} ({pct_below_zero*100:.1f}%)",
                }
                
                if pd.notna(months_above_bench):
                    monthly_data['Months Above Benchmark'] = f"{int(months_above_bench)} ({pct_above_bench*100:.1f}%)"
                    monthly_data['Months Below Benchmark'] = f"{int(months_below_bench)} ({pct_below_bench*100:.1f}%)"
                
                monthly_data['Best Month'] = f"{best_month*100:.2f}%" if pd.notna(best_month) else "N/A"
                monthly_data['Worst Month'] = f"{worst_month*100:.2f}%" if pd.notna(worst_month) else "N/A"
                
                if monthly_data:
                    monthly_df = pd.DataFrame([monthly_data])
                    st.dataframe(monthly_df, use_container_width=True, hide_index=True)
                
                # Monthly Returns Calendar
                st.markdown("---")
                st.markdown("#### Monthly Returns Calendar")
                
                # Benchmark and comparison method selection
                cal_col1, cal_col2 = st.columns([1, 1])
                
                with cal_col1:
                    if benchmarks is not None:
                        benchmark_cols = benchmarks.columns.tolist()
                        default_bench_idx = benchmark_cols.index('CDI') if 'CDI' in benchmark_cols else 0
                        selected_calendar_benchmark = st.selectbox(
                            "Select Benchmark for Comparison:",
                            options=benchmark_cols,
                            index=default_bench_idx,
                            key="calendar_benchmark"
                        )
                    else:
                        selected_calendar_benchmark = None
                        st.warning("No benchmarks available")
                
                with cal_col2:
                    comparison_method = st.selectbox(
                        "Comparison Method:",
                        options=['Relative Performance', 'Percentage Points', 'Benchmark Performance'],
                        index=0,
                        key="comparison_method"
                    )
                
                if selected_calendar_benchmark and selected_calendar_benchmark in benchmarks.columns:
                    # Create monthly returns table
                    benchmark_series = benchmarks[selected_calendar_benchmark]
                    monthly_table = create_monthly_returns_table(
                        fund_returns_full,
                        benchmark_series,
                        comparison_method
                    )
                    
                    # Style the table as HTML
                    styled_html = style_monthly_returns_table(monthly_table, comparison_method)
                    
                    # Display HTML table
                    st.markdown(styled_html, unsafe_allow_html=True)
                    
                    # Add explanation
                    with st.expander("ℹ️ Understanding the Monthly Returns Calendar"):
                        explanation_text = f"""
                        **How to read this table:**
                        
                        For each year, {"two" if comparison_method == "Benchmark Performance" else "three"} rows are displayed:
                        - **Fund**: Monthly returns of the investment fund
                        """
                        
                        if comparison_method != 'Benchmark Performance':
                            explanation_text += f"- **Benchmark**: Monthly returns of the selected benchmark ({selected_calendar_benchmark})\n"
                        
                        explanation_text += f"""- **{comparison_method}**: The comparison metric between fund and benchmark
                        
                        **Comparison Methods:**
                        - **Relative Performance**: Ratio showing fund performance relative to benchmark
                          - Same returns: 100%
                          - Fund 2%, Benchmark 1%: 200% (fund returned twice as much)
                          - Fund 0.5%, Benchmark 1%: 50% (fund returned half as much)
                          - When both negative: inverted ratio (smaller loss = better performance)
                          - Example: Fund -1%, Benchmark -2%: 200% (fund lost half, outperformed)
                        
                        - **Percentage Points**: Fund return minus benchmark return in absolute terms
                          - Example: +2.5% means fund outperformed by 2.5 percentage points
                        
                        - **Benchmark Performance**: Displays the benchmark's monthly returns for reference
                          - The Benchmark row is hidden when this option is selected (redundant)
                        
                        **Columns:**
                        - **Year**: The calendar year (merged cell for all rows in that year)
                        - **Type**: Fund, Benchmark (if shown), or Comparison metric
                        - **Jan-Dec**: Monthly returns/comparison for each month
                        - **YTD**: Year-to-date accumulated performance
                        - **Total**: Cumulative performance since the beginning of the fund's history
                        
                        **Visual Guide:**
                        - Negative values are displayed in **red** for easy identification
                        - Bold gold borders separate year groups and column sections
                        - Latest year appears first (reverse chronological order)
                        - The table can display up to 5 years of historical data
                        """
                        
                        st.markdown(explanation_text)
            else:
                st.warning("⚠️ Fund time series data not available. Please upload Fund Details file.")
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════
            # RISK-ADJUSTED PERFORMANCE SECTION
            # ═══════════════════════════════════════════════════════════════════
            
            st.markdown("### ⚖️ Risk-Adjusted Performance")
            
            if returns_result is not None:
                fund_returns_filtered, fund_returns_full = returns_result
                
                # Frequency selection
                st.markdown("#### Data Frequency Selection")
                frequency_choice = st.radio(
                    "Select frequency for Omega, Rachev, VaR and CVaR analysis:",
                    options=['Daily', 'Weekly', 'Monthly'],
                    horizontal=True,
                    help="Choose whether to analyze daily, weekly, or monthly returns data"
                )

                freq_suffix = 'DAILY' if frequency_choice == 'Daily' else ('WEEKLY' if frequency_choice == 'Weekly' else 'MONTHLY')
                freq_label = frequency_choice.lower()

                # Use appropriate returns data
                if frequency_choice == 'Daily':
                    returns_data = fund_returns_full
                elif frequency_choice == 'Weekly':
                    # Convert to weekly returns
                    returns_data = fund_returns_full.resample('W').apply(lambda x: (1 + x).prod() - 1)
                else:
                    # Convert to monthly returns
                    returns_data = fund_returns_full.resample('M').apply(lambda x: (1 + x).prod() - 1)
                
                st.markdown("---")
                
                # === OMEGA SECTION (Top Row) ===
                st.markdown("#### Omega Ratio")
                
                omega_chart_col, omega_gauge_col = st.columns([2, 1])
                
                with omega_chart_col:
                    # Omega CDF chart
                    fig_omega = create_omega_cdf_chart(returns_data, threshold=0, frequency=freq_label)
                    st.plotly_chart(fig_omega, use_container_width=True)
                
                with omega_gauge_col:
                    # Omega gauge with selected frequency
                    omega_val = fund_info.get(f'OMEGA_{freq_suffix}', np.nan)
                    if pd.notna(omega_val) and not np.isinf(omega_val):
                        fig_omega_gauge = create_omega_gauge(omega_val, frequency=frequency_choice)
                        st.plotly_chart(fig_omega_gauge, use_container_width=True)
                    else:
                        st.metric(f"Omega Ratio ({frequency_choice})", "N/A")
                
                st.markdown("---")
                
                # === RACHEV / VAR / CVAR SECTION (Bottom Row) ===
                st.markdown("#### Rachev Ratio & Tail Risk")
                
                rachev_chart_col, rachev_metrics_col = st.columns([2, 1])
                
                with rachev_chart_col:
                    # Combined chart with selected frequency
                    var_col = 'VAR_95_D' if frequency_choice == 'Daily' else ('VAR_95_W' if frequency_choice == 'Weekly' else 'VAR_95_M')
                    cvar_col = 'CVAR_95_D' if frequency_choice == 'Daily' else ('CVAR_95_W' if frequency_choice == 'Weekly' else 'CVAR_95_M')
                    
                    var_val = fund_info.get(var_col, np.nan)
                    cvar_val = fund_info.get(cvar_col, np.nan)
                    
                    fig_rachev = create_combined_rachev_var_chart(
                        returns_data, var_val, cvar_val, frequency=freq_label
                    )
                    st.plotly_chart(fig_rachev, use_container_width=True)
                
                with rachev_metrics_col:
                    # Rachev gauge with selected frequency
                    rachev_val = fund_info.get(f'RACHEV_{freq_suffix}', np.nan)
                    if pd.notna(rachev_val) and not np.isinf(rachev_val):
                        fig_rachev_gauge = create_rachev_gauge(rachev_val, frequency=frequency_choice)
                        st.plotly_chart(fig_rachev_gauge, use_container_width=True)
                    else:
                        st.metric(f"Rachev Ratio ({frequency_choice})", "N/A")
                
                # Explanation guide
                with st.expander("📚 Understanding Omega & Rachev Ratios"):
                    st.markdown("""
                    ### Omega Ratio
                    
                    **What it measures:** Probability-weighted ratio of gains vs losses above/below a threshold (usually 0%).
                    
                    **Formula:** Ω = (Sum of gains above threshold) / (Sum of losses below threshold)
                    
                    **Interpretation:**
                    - **Ω < 1.0**: Poor - Losses exceed gains
                    - **Ω 1.0-1.5**: Below average
                    - **Ω 1.5-2.0**: Average/Good
                    - **Ω 2.0-3.0**: Very good
                    - **Ω > 3.0**: Excellent
                    - **Higher is better** - more gains relative to losses
                    
                    **Visualization:** The CDF chart shows cumulative probability. Red area (below threshold) represents losses, green area (above threshold) represents gains. The gauge shows performance quality.
                    
                    ---
                    
                    ### Rachev Ratio (5% Tails)
                    
                    **What it measures:** Expected loss in worst 5% scenarios vs expected gain in best 5% scenarios.
                    
                    **Formula:** R = E[Loss | worst 5%] / E[Gain | best 5%]
                    
                    **Interpretation:**
                    - **R < 0.5**: Excellent - Very asymmetric (small losses, large gains)
                    - **R 0.5-0.75**: Very good
                    - **R 0.75-1.0**: Good/Average (symmetric risk)
                    - **R 1.0-1.5**: Below average (losses exceeding gains)
                    - **R > 1.5**: Poor - High tail risk
                    - **Lower is better** - smaller downside relative to upside
                    
                    **Why 5% tails?** Captures significant tail events while maintaining statistical robustness. Focuses on the extreme 5% of returns in each tail.
                    
                    **Visualization:** The PDF chart highlights the 5% tails. Red shows expected loss in crashes, green shows expected gain in rallies. The gauge shows risk quality.
                    
                    ---
                    
                    ### VaR (Value at Risk) & CVaR (Conditional VaR)
                    
                    **VaR 95%**: The maximum loss expected 95% of the time (5% worst case threshold)
                    
                    **CVaR 95%**: The average loss in the worst 5% of cases (expected shortfall)
                    
                    Both metrics help quantify downside risk in monetary terms.
                    
                    ---
                    
                    ### Daily vs Weekly vs Monthly

                    **Daily**: Most granular, captures short-term volatility and daily trading risk. Best for high-frequency trading strategies.

                    **Weekly**: Balanced view that smooths out daily noise while still capturing medium-term patterns. Useful for swing trading and tactical allocation.

                    **Monthly**: Focuses on longer-term performance patterns, smooths out short-term volatility. Best for strategic asset allocation.

                    All three frequencies provide valuable complementary insights into fund behavior at different time scales.
                    """)
            else:
                st.info("Upload Fund Details to see risk-adjusted performance visualizations")
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════
            # SHARPE RATIO ANALYSIS
            # ═══════════════════════════════════════════════════════════════════
            
            if returns_result is not None:
                fund_returns_filtered, fund_returns_full = returns_result
                
                st.markdown("### 📊 Sharpe Ratio Analysis")
                
                sharpe_chart_col, sharpe_metrics_col = st.columns([3, 1])
                
                with sharpe_chart_col:
                    # Rolling Sharpe chart
                    fig_sharpe = create_rolling_sharpe_chart(fund_returns_full, window_months=12)
                    st.plotly_chart(fig_sharpe, use_container_width=True)
                
                with sharpe_metrics_col:
                    # Sharpe metrics (4x1 format - added SHARPE_TOTAL)
                    sharpe_12m = fund_info.get('SHARPE_12M', np.nan)
                    st.metric("Sharpe 12M", f"{sharpe_12m:.2f}" if pd.notna(sharpe_12m) else "N/A")
                    
                    sharpe_24m = fund_info.get('SHARPE_24M', np.nan)
                    st.metric("Sharpe 24M", f"{sharpe_24m:.2f}" if pd.notna(sharpe_24m) else "N/A")
                    
                    sharpe_36m = fund_info.get('SHARPE_36M', np.nan)
                    st.metric("Sharpe 36M", f"{sharpe_36m:.2f}" if pd.notna(sharpe_36m) else "N/A")

                    sharpe_total = fund_info.get('SHARPE_TOTAL', np.nan)
                    st.metric("Sharpe Total", f"{sharpe_total:.2f}" if pd.notna(sharpe_total) else "N/A")
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════
            # RISK METRICS SECTION
            # ═══════════════════════════════════════════════════════════════════
            
            st.markdown("### 🎯 Risk Metrics Dashboard")
            
            if returns_result is not None:
                fund_returns_filtered, fund_returns_full = returns_result
                
                # Volatility section
                st.markdown("#### Volatility Analysis")
                
                vol_chart_col, vol_metrics_col = st.columns([3, 1])
                
                with vol_chart_col:
                    # Rolling volatility chart
                    fig_vol = create_rolling_vol_chart(fund_returns_full, window_months=12)
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                with vol_metrics_col:
                    # Volatility metrics (4x1 format - added VOL_36M)
                    vol_12m = fund_info.get('VOL_12M', np.nan)
                    st.metric("Vol 12M", f"{vol_12m*100:.2f}%" if pd.notna(vol_12m) else "N/A")
                    
                    vol_24m = fund_info.get('VOL_24M', np.nan)
                    st.metric("Vol 24M", f"{vol_24m*100:.2f}%" if pd.notna(vol_24m) else "N/A")

                    vol_36m = fund_info.get('VOL_36M', np.nan)
                    st.metric("Vol 36M", f"{vol_36m*100:.2f}%" if pd.notna(vol_36m) else "N/A")
                    
                    vol_total = fund_info.get('VOL_TOTAL', np.nan)
                    st.metric("Vol Total", f"{vol_total*100:.2f}%" if pd.notna(vol_total) else "N/A")
                
                st.markdown("---")
                
                # Drawdown section
                st.markdown("#### Drawdown Analysis")
                
                dd_chart_col, dd_metrics_col = st.columns([3, 1])
                
                with dd_chart_col:
                    # Underwater plot
                    fig_underwater, max_dd_info = create_underwater_plot(fund_returns_full)
                    st.plotly_chart(fig_underwater, use_container_width=True)
                
                with dd_metrics_col:
                    # Drawdown metrics (1x2 format)
                    mdd = fund_info.get('MDD', np.nan)
                    st.metric("Max Drawdown", f"{mdd*100:.2f}%" if pd.notna(mdd) else "N/A")
                    
                    mdd_days = fund_info.get('MDD_DAYS', np.nan)
                    st.metric("MDD Duration", f"{int(mdd_days)} days" if pd.notna(mdd_days) else "N/A")
                    
                    # Display CDaR metric
                    if max_dd_info and 'cdar_95' in max_dd_info:
                        cdar_95 = max_dd_info['cdar_95']
                        st.metric("CDaR (95%)", f"{cdar_95:.2f}%",
                                 help="Conditional Drawdown at Risk: Average of worst 5% drawdowns")
            else:
                st.info("Upload Fund Details to see risk metrics visualizations")
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════
            # EXPOSURE ANALYSIS SECTION - CALCULATED ON-DEMAND
            # ═══════════════════════════════════════════════════════════════════
            
            st.markdown("### 🌍 Benchmark Exposures")
            
            if benchmarks is not None and fund_details is not None and returns_result is not None:
                available_benchmarks = [b for b in ['CDI', 'USDBRL', 'GOLD', 'IBOVESPA', 'SP500', 'BITCOIN'] if b in benchmarks.columns]
                default_exposure = []
                if 'CDI' in available_benchmarks:
                    default_exposure.append('CDI')
                if 'IBOVESPA' in available_benchmarks:
                    default_exposure.append('IBOVESPA')
                
                selected_exposure_benches = st.multiselect(
                    "Select Benchmarks for Exposure Analysis:",
                    options=available_benchmarks,
                    default=default_exposure,
                    key="detailed_exposure_select"
                )
                
                if selected_exposure_benches:
                    exposure_data = []
                    
                    for bench in selected_exposure_benches:
                        # Calculate rolling copula metrics to match time series analysis
                        with st.spinner(f'Calculating exposure for {bench}...'):
                            copula_results = estimate_rolling_copula_for_chart(
                                fund_returns_full,
                                benchmarks[bench],
                                window=250
                            )
                            
                            if copula_results is not None:
                                # Last window values (most recent)
                                last_kendall = copula_results['kendall_tau'].iloc[-1]
                                last_tail_lower = copula_results['tail_lower'].iloc[-1]
                                last_tail_upper = copula_results['tail_upper'].iloc[-1]
                                last_asymmetry = copula_results['asymmetry_index'].iloc[-1]
                                
                                # Average values across all windows
                                avg_kendall = copula_results['kendall_tau'].mean()
                                avg_tail_lower = copula_results['tail_lower'].mean()
                                avg_tail_upper = copula_results['tail_upper'].mean()
                                avg_asymmetry = copula_results['asymmetry_index'].mean()
                                
                                # Add last window row
                                exposure_data.append({
                                    'Benchmark': f'{bench} - Last Window',
                                    'Kendall Tau': last_kendall,
                                    'Tail Lower': last_tail_lower,
                                    'Tail Upper': last_tail_upper,
                                    'Asymmetry': last_asymmetry
                                })
                                
                                # Add average row
                                exposure_data.append({
                                    'Benchmark': f'{bench} - Average',
                                    'Kendall Tau': avg_kendall,
                                    'Tail Lower': avg_tail_lower,
                                    'Tail Upper': avg_tail_upper,
                                    'Asymmetry': avg_asymmetry
                                })
                            else:
                                # Insufficient data - use full data calculation as fallback
                                bench_returns = benchmarks[bench].reindex(fund_returns_full.index, method='ffill').fillna(0)
                                
                                u = to_empirical_cdf(fund_returns_full)
                                v = to_empirical_cdf(bench_returns)
                                
                                tau = stats.kendalltau(u.values, v.values)[0]
                                
                                theta_lower, _ = estimate_gumbel_270_parameter(u.values, v.values)
                                lambda_lower, _ = gumbel_270_tail_dependence(theta_lower)
                                
                                theta_upper, _ = estimate_gumbel_180_parameter(u.values, v.values)
                                _, lambda_upper = gumbel_180_tail_dependence(theta_upper)
                                
                                asymmetry = (lambda_lower - lambda_upper) / (lambda_lower + lambda_upper) if (lambda_lower + lambda_upper) > 0 else 0
                                
                                exposure_data.append({
                                    'Benchmark': f'{bench} - Full Period',
                                    'Kendall Tau': tau,
                                    'Tail Lower': lambda_lower,
                                    'Tail Upper': lambda_upper,
                                    'Asymmetry': asymmetry
                                })
                    
                    exposure_df = pd.DataFrame(exposure_data)
                    
                    # Color-coded gradient
                    st.dataframe(
                        exposure_df.style.format(
                            {col: "{:.4f}" for col in exposure_df.columns if col != 'Benchmark'}
                        ).background_gradient(
                            cmap='RdYlGn',
                            subset=[col for col in exposure_df.columns if col != 'Benchmark'],
                            vmin=-1, vmax=1
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    with st.expander("📚 Exposure Metrics Guide"):
                        st.markdown("""
                        **Kendall Tau**: Overall correlation (-1 to +1)
                        - Positive: moves together | Zero: independent | Negative: moves opposite
                        
                        **Tail Lower**: Crash correlation (0 to 1)
                        - High: fund crashes together with benchmark
                        
                        **Tail Upper**: Boom correlation (0 to 1)
                        - High: fund rallies together with benchmark
                        
                        **Asymmetry**: Crash vs Boom bias (-1 to +1)
                        - Positive: stronger crash correlation | Negative: stronger boom correlation
                        """)
                else:
                    st.info("Select at least one benchmark to view exposures")
            else:
                st.warning("⚠️ Benchmark data and fund details are required for exposure analysis")
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════
            # EXPOSURE TIME SERIES ANALYSIS FOR INDIVIDUAL FUND
            # ═══════════════════════════════════════════════════════════════════
            
            st.markdown("### 📈 Fund Exposure Time Series Analysis")
            st.info("💡 Select a benchmark below to visualize the evolution of fund exposure metrics over time")
            
            if benchmarks is not None and fund_details is not None and returns_result is not None:
                # Benchmark selection for time series
                available_ts_benchmarks = ['None'] + [b for b in ['CDI', 'USDBRL', 'GOLD', 'IBOVESPA', 'SP500', 'BITCOIN'] if b in benchmarks.columns]
                selected_fund_ts_benchmark = st.selectbox(
                    "Select Benchmark for Time Series:",
                    options=available_ts_benchmarks,
                    index=0,
                    key="fund_ts_benchmark_selector"
                )
                
                if selected_fund_ts_benchmark != 'None' and selected_fund_ts_benchmark in benchmarks.columns:
                    with st.spinner(f'Calculating fund exposure time series for {selected_fund_ts_benchmark}...'):
                        # Calculate rolling copula metrics for fund
                        copula_results = estimate_rolling_copula_for_chart(
                            fund_returns_full,
                            benchmarks[selected_fund_ts_benchmark],
                            window=250
                        )
                        
                        if copula_results is not None:
                            # Calculate current and average values
                            current_kendall = copula_results['kendall_tau'].iloc[-1]
                            avg_kendall = copula_results['kendall_tau'].mean()
                            
                            current_tail_lower = copula_results['tail_lower'].iloc[-1]
                            avg_tail_lower = copula_results['tail_lower'].mean()
                            
                            current_tail_upper = copula_results['tail_upper'].iloc[-1]
                            avg_tail_upper = copula_results['tail_upper'].mean()
                            
                            current_asymmetry = copula_results['asymmetry_index'].iloc[-1]
                            avg_asymmetry = copula_results['asymmetry_index'].mean()
                            
                            # Create 2x2 grid of charts
                            st.markdown(f"##### Fund Exposure Evolution - {selected_fund_ts_benchmark}")
                            
                            # Row 1: Kendall Tau and Asymmetry
                            row1_col1, row1_col2 = st.columns(2)
                            
                            with row1_col1:
                                fig_kendall = create_exposure_time_series_chart(
                                    copula_results,
                                    'kendall_tau',
                                    current_kendall,
                                    avg_kendall,
                                    selected_fund_ts_benchmark
                                )
                                st.plotly_chart(fig_kendall, use_container_width=True)
                            
                            with row1_col2:
                                fig_asymmetry = create_exposure_time_series_chart(
                                    copula_results,
                                    'asymmetry_index',
                                    current_asymmetry,
                                    avg_asymmetry,
                                    selected_fund_ts_benchmark
                                )
                                st.plotly_chart(fig_asymmetry, use_container_width=True)
                            
                            # Row 2: Tail Lower and Tail Upper
                            row2_col1, row2_col2 = st.columns(2)
                            
                            with row2_col1:
                                fig_tail_lower = create_exposure_time_series_chart(
                                    copula_results,
                                    'tail_lower',
                                    current_tail_lower,
                                    avg_tail_lower,
                                    selected_fund_ts_benchmark
                                )
                                st.plotly_chart(fig_tail_lower, use_container_width=True)
                            
                            with row2_col2:
                                fig_tail_upper = create_exposure_time_series_chart(
                                    copula_results,
                                    'tail_upper',
                                    current_tail_upper,
                                    avg_tail_upper,
                                    selected_fund_ts_benchmark
                                )
                                st.plotly_chart(fig_tail_upper, use_container_width=True)
                            
                            # Summary metrics below charts
                            st.markdown("##### Summary Statistics")
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.metric(
                                    "Kendall Tau",
                                    f"{current_kendall:.4f}",
                                    delta=f"Avg: {avg_kendall:.4f}",
                                    help="Overall correlation between fund and benchmark"
                                )
                            
                            with metric_col2:
                                st.metric(
                                    "Tail Lower",
                                    f"{current_tail_lower:.4f}",
                                    delta=f"Avg: {avg_tail_lower:.4f}",
                                    help="Crash correlation strength"
                                )
                            
                            with metric_col3:
                                st.metric(
                                    "Tail Upper",
                                    f"{current_tail_upper:.4f}",
                                    delta=f"Avg: {avg_tail_upper:.4f}",
                                    help="Boom correlation strength"
                                )
                            
                            with metric_col4:
                                st.metric(
                                    "Asymmetry",
                                    f"{current_asymmetry:.4f}",
                                    delta=f"Avg: {avg_asymmetry:.4f}",
                                    help="Crash vs boom bias"
                                )
                        else:
                            st.warning("Insufficient data for time series analysis (need at least 275 observations)")
            else:
                st.warning("⚠️ Benchmark data and fund details are required for time series exposure analysis")
            
                # ═══════════════════════════════════════════════════════════════
                # EXPOSURE TIME SERIES CHARTS (NEW SECTION)
                # ═══════════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("#### 📈 Exposure Time Series Analysis")
                st.info("💡 Select a benchmark below to visualize the evolution of exposure metrics over time")
                
                # Benchmark selection for time series
                selected_ts_benchmark = st.selectbox(
                    "Select Benchmark for Time Series:",
                    options=['None'] + available_benchmarks,
                    index=0,
                    key="ts_benchmark_selector"
                )
                
                if selected_ts_benchmark != 'None' and selected_ts_benchmark in benchmarks.columns:
                    # Check if we have fund returns
                    if returns_result is not None:
                        fund_returns_filtered, fund_returns_full = returns_result
                        
                        with st.spinner(f'Calculating exposure time series for {selected_ts_benchmark}...'):
                            # Calculate rolling copula metrics
                            copula_results = estimate_rolling_copula_for_chart(
                                fund_returns_full,
                                benchmarks[selected_ts_benchmark],
                                window=250
                            )
                            
                            if copula_results is not None:
                                # Get last and average values from fund_info
                                last_kendall = fund_info.get(f'KENDALL_TAU_{selected_ts_benchmark}', np.nan)
                                avg_kendall = fund_info.get(f'KENDALL_TAU_AVG_{selected_ts_benchmark}', np.nan)
                                
                                last_tail_lower = fund_info.get(f'TAIL_LOWER_{selected_ts_benchmark}', np.nan)
                                avg_tail_lower = fund_info.get(f'TAIL_LOWER_AVG_{selected_ts_benchmark}', np.nan)
                                
                                last_tail_upper = fund_info.get(f'TAIL_UPPER_{selected_ts_benchmark}', np.nan)
                                avg_tail_upper = fund_info.get(f'TAIL_UPPER_AVG_{selected_ts_benchmark}', np.nan)
                                
                                last_asymmetry = fund_info.get(f'ASYMMETRY_{selected_ts_benchmark}', np.nan)
                                avg_asymmetry = fund_info.get(f'ASYMMETRY_AVG_{selected_ts_benchmark}', np.nan)
                                
                                # Create 2x2 grid of charts
                                st.markdown(f"##### Exposure Evolution - {selected_ts_benchmark}")
                                
                                # Row 1: Kendall Tau and Asymmetry
                                row1_col1, row1_col2 = st.columns(2)
                                
                                with row1_col1:
                                    fig_kendall = create_exposure_time_series_chart(
                                        copula_results,
                                        'kendall_tau',
                                        last_kendall,
                                        avg_kendall,
                                        selected_ts_benchmark
                                    )
                                    st.plotly_chart(fig_kendall, use_container_width=True)
                                
                                with row1_col2:
                                    fig_asymmetry = create_exposure_time_series_chart(
                                        copula_results,
                                        'asymmetry_index',
                                        last_asymmetry,
                                        avg_asymmetry,
                                        selected_ts_benchmark
                                    )
                                    st.plotly_chart(fig_asymmetry, use_container_width=True)
                                
                                # Row 2: Lower Tail and Upper Tail
                                row2_col1, row2_col2 = st.columns(2)
                                
                                with row2_col1:
                                    fig_tail_lower = create_exposure_time_series_chart(
                                        copula_results,
                                        'tail_lower',
                                        last_tail_lower,
                                        avg_tail_lower,
                                        selected_ts_benchmark
                                    )
                                    st.plotly_chart(fig_tail_lower, use_container_width=True)
                                
                                with row2_col2:
                                    fig_tail_upper = create_exposure_time_series_chart(
                                        copula_results,
                                        'tail_upper',
                                        last_tail_upper,
                                        avg_tail_upper,
                                        selected_ts_benchmark
                                    )
                                    st.plotly_chart(fig_tail_upper, use_container_width=True)
                                
                                # Summary metrics
                                st.markdown("##### Summary Statistics")
                                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                                
                                with summary_col1:
                                    st.metric(
                                        "Kendall Tau (Last)",
                                        f"{last_kendall:.4f}" if pd.notna(last_kendall) else "N/A",
                                        delta=f"{(last_kendall - avg_kendall):.4f}" if pd.notna(last_kendall) and pd.notna(avg_kendall) else None
                                    )
                                
                                with summary_col2:
                                    st.metric(
                                        "Lower Tail (Last)",
                                        f"{last_tail_lower:.4f}" if pd.notna(last_tail_lower) else "N/A",
                                        delta=f"{(last_tail_lower - avg_tail_lower):.4f}" if pd.notna(last_tail_lower) and pd.notna(avg_tail_lower) else None
                                    )
                                
                                with summary_col3:
                                    st.metric(
                                        "Upper Tail (Last)",
                                        f"{last_tail_upper:.4f}" if pd.notna(last_tail_upper) else "N/A",
                                        delta=f"{(last_tail_upper - avg_tail_upper):.4f}" if pd.notna(last_tail_upper) and pd.notna(avg_tail_upper) else None
                                    )
                                
                                with summary_col4:
                                    st.metric(
                                        "Asymmetry (Last)",
                                        f"{last_asymmetry:.4f}" if pd.notna(last_asymmetry) else "N/A",
                                        delta=f"{(last_asymmetry - avg_asymmetry):.4f}" if pd.notna(last_asymmetry) and pd.notna(avg_asymmetry) else None
                                    )
                                
                                # Interpretation guide
                                with st.expander("📖 How to Read These Charts"):
                                    st.markdown("""
                                    **Chart Elements:**
                                    - **Yellow Line**: Time series of the exposure metric across all rolling windows
                                    - **Red Dot**: Most recent value (last window)
                                    - **Blue Line**: Average value across all windows (horizontal reference)
                                    
                                    **What This Shows:**
                                    - The evolution of the fund's relationship with the selected benchmark over time
                                    - Whether current exposure is above or below historical average
                                    - Trends and changes in the nature of the dependence
                                    
                                    **Interpreting Trends:**
                                    - **Kendall Tau**: Overall correlation strength - higher means more synchronized movements
                                    - **Lower Tail**: Crash correlation - higher means fund tends to fall when benchmark crashes
                                    - **Upper Tail**: Boom correlation - higher means fund tends to rise when benchmark rallies
                                    - **Asymmetry**: Positive = stronger crash link, Negative = stronger boom link
                                    
                                    **Rolling Window**: The calculations use a 250-day (≈1 year) rolling window, providing a dynamic view of how the relationship evolves over time.
                                    """)
                            else:
                                st.warning("⚠️ Insufficient data to calculate exposure time series for this benchmark")
                    else:
                        st.info("📊 Upload Fund Details file to enable exposure time series analysis")
                else:
                    st.info("Select at least one benchmark to view exposures")
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════
            # AUM & SHAREHOLDERS SECTION
            # ═══════════════════════════════════════════════════════════════════
            
            st.markdown("### 💰 Fund Size & Growth")
            
            if fund_details is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    # AUM chart
                    fig_aum = create_aum_chart(fund_details, selected_fund_cnpj_standard)
                    if fig_aum is not None:
                        st.plotly_chart(fig_aum, use_container_width=True)
                    else:
                        st.info("AUM data not available")
                
                with col2:
                    # Shareholders chart
                    fig_shareholders = create_shareholders_chart(fund_details, selected_fund_cnpj_standard)
                    if fig_shareholders is not None:
                        st.plotly_chart(fig_shareholders, use_container_width=True)
                    else:
                        st.info("Shareholders data not available")
            else:
                st.info("Upload Fund Details to see AUM and Shareholders time series")

    
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: ADVANCED COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.title("🔍 ADVANCED FUND COMPARISON")
        st.markdown("### Select metrics and filter funds for comparison")
        st.markdown("---")
        
        if fund_metrics is None:
            st.warning("⚠️ Please upload fund metrics data to use this feature")
            return
        
        # ═══════════════════════════════════════════════════════════════════════
        # COLUMN SELECTION
        # ═══════════════════════════════════════════════════════════════════════
        
        st.markdown("#### 📊 Select Columns to Display")
        
        # Define column categories for easier selection
        basic_info_cols = ['FUNDO DE INVESTIMENTO', 'CNPJ', 'GESTOR', 'CATEGORIA BTG', 
                          'SUBCATEGORIA BTG', 'STATUS', 'VL_PATRIM_LIQ', 'NR_COTST', 'TRIBUTAÇÃO', 'LIQUIDEZ', 'SUITABILITY']
        
        return_cols = [col for col in fund_metrics.columns if 'RETURN' in col] + \
                      [col for col in fund_metrics.columns if 'EXCESS' in col]
        
        risk_cols = ['VOL_12M', 'VOL_24M', 'VOL_36M', 'VOL_TOTAL', 
                    'SHARPE_12M', 'SHARPE_24M', 'SHARPE_36M', 'SHARPE_TOTAL',
                    'MDD', 'CDAR_95', 'MDD_DAYS']
        
        advanced_cols = ['OMEGA_DAILY', 'OMEGA_MONTHLY', 'OMEGA_WEEKLY', 'RACHEV_DAILY', 'RACHEV_MONTHLY', 'RACHEV_WEEKLY',
                        'VAR_95_D', 'VAR_95_M', 'CVAR_95_D', 'CVAR_95_M']
        
        exposure_cols = [col for col in fund_metrics.columns if any(x in col for x in 
                        ['KENDALL_TAU', 'TAIL_LOWER', 'TAIL_UPPER', 'ASYMMETRY'])]
        
        monthly_cols = ['M_ABOVE_0', 'M_ABOVE_BCHMK', 'BEST_MONTH', 'WORST_MONTH']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_basic = st.multiselect(
                "Basic Information",
                options=basic_info_cols,
                default=['FUNDO DE INVESTIMENTO', 'CATEGORIA BTG', 'SUBCATEGORIA BTG', 'STATUS','VL_PATRIM_LIQ', 'NR_COTST']
            )
        
        with col2:
            selected_returns = st.multiselect(
                "Return Metrics",
                options=return_cols,
                default=['EXCESS_12M', 'EXCESS_24M'] if all(x in return_cols for x in ['EXCESS_12M', 'EXCESS_24M']) else []
            )
        
        with col3:
            selected_risk = st.multiselect(
                "Risk Metrics",
                options=risk_cols,
                default=['VOL_TOTAL', 'MDD'] if all(x in risk_cols for x in ['VOL_TOTAL', 'MDD']) else []
            )
        
        col4, col5 = st.columns(2)
        
        with col4:
            selected_advanced = st.multiselect(
                "Advanced Risk Metrics",
                options=advanced_cols,
                default=['OMEGA_DAILY', 'RACHEV_DAILY', 'CVAR_95_M'] if all(x in advanced_cols for x in ['OMEGA_DAILY', 'RACHEV_DAILY', 'CVAR_95_M']) else []
            )
        
        with col5:
            # Simplified exposure selection
            exposure_benchmarks_display = ['CDI', 'IBOVESPA', 'GOLD', 'USDBRL', 'SP500', 'BITCOIN']
            selected_exposure_display = st.multiselect(
                "Tail Dependence (select benchmarks)",
                options=exposure_benchmarks_display,
                default=[]
            )
            
            # Convert to actual column names
            selected_exposure = []
            for bench in selected_exposure_display:
                selected_exposure.extend([col for col in exposure_cols if bench in col])
        
        # Combine all selected columns
        all_selected_cols = selected_basic + selected_returns + selected_risk + selected_advanced + selected_exposure
        
        if not all_selected_cols:
            st.warning("⚠️ Please select at least one column to display")
            return
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FILTERING SECTION
        # ═══════════════════════════════════════════════════════════════════════

        st.markdown("#### 🔎 Filter Funds")

        # Create a copy of the dataframe with selected columns
        display_df = fund_metrics[all_selected_cols].copy()
        
        # Clean Liquidez column if present - extract numeric days from D+N format
        if 'LIQUIDEZ' in display_df.columns:
            def extract_liquidez_days(liquidez_str):
                """Extract numeric days from D+N format"""
                if pd.isna(liquidez_str) or liquidez_str == 'N/A' or liquidez_str == 'n/a':
                    return np.nan
                try:
                    # Convert to string and clean
                    liquidez_str = str(liquidez_str).strip().upper()
                    # Extract number after D+ or D +
                    if 'D+' in liquidez_str or 'D +' in liquidez_str:
                        # Remove D+ or D + prefix and extract number
                        num_str = liquidez_str.replace('D+', '').replace('D +', '').strip()
                        return float(num_str)
                    else:
                        # Try to convert directly to float
                        return float(liquidez_str)
                except:
                    return np.nan
            
            # Create numeric version of Liquidez
            display_df['LIQUIDEZ_DAYS'] = display_df['LIQUIDEZ'].apply(extract_liquidez_days)
            
            # If Liquidez was selected, add the numeric version to the dataframe
            # and update column lists
            if 'LIQUIDEZ' in all_selected_cols:
                # Insert LIQUIDEZ_DAYS right after LIQUIDEZ
                liquidez_idx = all_selected_cols.index('LIQUIDEZ')
                all_selected_cols.insert(liquidez_idx + 1, 'LIQUIDEZ_DAYS')

        # Separate numerical and categorical columns
        numerical_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in all_selected_cols if col not in numerical_cols]

        # Apply filters
        with st.expander("📈 Numerical Filters (Min/Max Ranges)", expanded=True):
            if numerical_cols:
                filter_cols = st.columns(3)
                active_filters = {}
                
                for idx, col in enumerate(numerical_cols):
                    with filter_cols[idx % 3]:
                        # Get min/max values
                        col_data = display_df[col].dropna()
                        if len(col_data) > 0:
                            global_min = float(col_data.min())
                            global_max = float(col_data.max())
                            
                            # Centered column name
                            st.markdown(f"<h6 style='text-align: center; color: #D4AF37'>{col}</h6>", unsafe_allow_html=True)
                            
                            # Text inputs for min/max with centered labels
                            num_col1, num_col2 = st.columns(2)
                            
                            with num_col1:
                                st.markdown("<p style='text-align: center; margin-bottom: -10px;'>Min</p>", unsafe_allow_html=True)
                                min_val = st.number_input(
                                    f"Min_{col}",
                                    min_value=global_min,
                                    max_value=global_max,
                                    value=global_min,
                                    key=f"min_{col}",
                                    label_visibility="collapsed"
                                )
                            
                            with num_col2:
                                st.markdown("<p style='text-align: center; margin-bottom: -10px;'>Max</p>", unsafe_allow_html=True)
                                max_val = st.number_input(
                                    f"Max_{col}",
                                    min_value=global_min,
                                    max_value=global_max,
                                    value=global_max,
                                    key=f"max_{col}",
                                    label_visibility="collapsed"
                                )
                            
                            # Ensure min <= max
                            if min_val > max_val:
                                st.markdown("<p style='text-align: center; color: red; font-size: 0.8em;'>Min cannot be greater than Max</p>", unsafe_allow_html=True)
                                # Auto-correct the values
                                min_val, max_val = global_min, global_max
                            
                            # Store filter if not default
                            if (min_val != global_min) or (max_val != global_max):
                                active_filters[col] = (min_val, max_val)
                                
            else:
                st.info("No numerical columns selected")

        with st.expander("📋 Categorical Filters", expanded=False):
            if categorical_cols:
                cat_filter_cols = st.columns(2)
                categorical_filters = {}
                
                for idx, col in enumerate(categorical_cols):
                    with cat_filter_cols[idx % 2]:
                        unique_vals = display_df[col].dropna().unique().tolist()
                        if unique_vals:
                            # Centered categorical filter label
                            st.markdown(f"<h6 style='text-align: center; color: #D4AF37'>{col}</h6>", unsafe_allow_html=True)
                            selected_vals = st.multiselect(
                                f"{col}_select",
                                options=unique_vals,
                                default=unique_vals,
                                key=f"cat_filter_{col}",
                                label_visibility="collapsed"
                            )
                            
                            if selected_vals != unique_vals:
                                categorical_filters[col] = selected_vals
            else:
                st.info("No categorical columns selected")

        # Apply all filters
        filtered_df = display_df.copy()

        # Apply numerical filters
        for col, (min_val, max_val) in active_filters.items():
            filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

        # Apply categorical filters
        for col, values in categorical_filters.items():
            filtered_df = filtered_df[filtered_df[col].isin(values)]

        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════════
        # RESULTS DISPLAY
        # ═══════════════════════════════════════════════════════════════════════
        
        st.markdown(f"#### 📈 Results: {len(filtered_df)} funds match your criteria")
        
        if len(filtered_df) > 0:
            # Format numerical columns for better display
            display_filtered = filtered_df.copy()
            
            # Format return columns as percentages
            for col in display_filtered.columns:
                if 'RETURN' in col or 'VOL' in col or 'SHARPE' in col or 'MDD' in col or \
                   'OMEGA' in col or 'RACHEV' in col or 'VAR' in col or 'CVAR' in col or \
                   'KENDALL' in col or 'TAIL' in col or 'ASYMMETRY' in col or 'CDAR' in col or\
                   'M_ABOVE' in col or 'BEST_MONTH' in col or 'WORST_MONTH' in col or \
                   'EXCESS' in col:
                    if display_filtered[col].dtype in [np.float64, np.float32]:
                        # Check if values are in decimal format (0.xx) or already percentage (xx.xx)
                        sample_val = display_filtered[col].dropna().iloc[0] if len(display_filtered[col].dropna()) > 0 else 0
                        if abs(sample_val) < 10:  # Likely decimal format
                            if 'SHARPE' not in col and 'OMEGA' not in col and 'RACHEV' not in col and 'KENDALL' not in col and 'ASYMMETRY' not in col:
                                display_filtered[col] = display_filtered[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                            else:
                                display_filtered[col] = display_filtered[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            # Format AUM
            if 'VL_PATRIM_LIQ' in display_filtered.columns:
                display_filtered['VL_PATRIM_LIQ'] = display_filtered['VL_PATRIM_LIQ'].apply(
                    lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "N/A"
                )
            
            # Format shareholders
            if 'NR_COTST' in display_filtered.columns:
                display_filtered['NR_COTST'] = display_filtered['NR_COTST'].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
                )
            
            # Display the dataframe
            st.dataframe(display_filtered, use_container_width=True, height=600)
            
            # Export option
            st.markdown("---")
            st.markdown("#### 💾 Export Results")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="fund_comparison_filtered.csv",
                    mime="text/csv",
                )
            
            with export_col2:
                # Export to Portfolio Construction
                if st.button("📤 Export to Portfolio Construction", use_container_width=True):
                    # Initialize session state for portfolio export if not exists
                    if 'selected_portfolio_funds' not in st.session_state:
                        st.session_state['selected_portfolio_funds'] = []
                    
                    # Get fund names from filtered results
                    if 'FUNDO DE INVESTIMENTO' in filtered_df.columns:
                        funds_to_export = filtered_df['FUNDO DE INVESTIMENTO'].tolist()
                        
                        # Add to portfolio construction list (merge, not replace)
                        new_funds = 0
                        for fund in funds_to_export:
                            if fund not in st.session_state['selected_portfolio_funds']:
                                st.session_state['selected_portfolio_funds'].append(fund)
                                new_funds += 1
                        
                        st.success(f"✅ Exported {new_funds} new funds to Portfolio Construction (Total: {len(st.session_state['selected_portfolio_funds'])} funds)")
                        st.info("💡 Navigate to the 'Portfolio Construction' tab to see your selection")
                    else:
                        st.error("❌ Cannot export: 'FUNDO DE INVESTIMENTO' column not in results")
            
            # Summary statistics
            with st.expander("📊 Summary Statistics"):
                st.markdown("**Numerical Columns Summary**")
                if numerical_cols:
                    summary_stats = filtered_df[numerical_cols].describe().T
                    summary_stats = summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                    st.dataframe(summary_stats.style.format("{:.4f}"), use_container_width=True)
        else:
            st.warning("⚠️ No funds match your filter criteria. Try adjusting the filters.")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 4: PORTFOLIO CONSTRUCTION - WASSERSTEIN DRO
    # ═══════════════════════════════════════════════════════════════════════════════
    
    with tab4:
        st.title("🎯 PORTFOLIO CONSTRUCTION")
        st.markdown("### Build an optimized portfolio using Wasserstein Distributionally Robust Optimization (DRO)")
        st.markdown("---")
        
        if fund_metrics is None or fund_details is None or benchmarks is None:
            st.warning("⚠️ Please upload all required data files (Fund Metrics, Fund Details, and Benchmarks) to use Portfolio Construction")
        
        # Initialize session state for selected funds (moved outside condition)
        if 'selected_portfolio_funds' not in st.session_state:
            st.session_state['selected_portfolio_funds'] = []
        
        # ═══════════════════════════════════════════════════════════════════════════
        # FUND SELECTION INTERFACE
        # ═══════════════════════════════════════════════════════════════════════════
        
        st.markdown("#### 📝 Step 1: Select Funds for Portfolio")
        
        # Selection method
        selection_method = st.radio(
            "Choose selection method:",
            ["🔍 Search and Select Funds", "📤 Upload Excel File"],
            horizontal=True,
            key="portfolio_selection_method"
        )
        
        if selection_method == "📤 Upload Excel File":
            st.markdown("---")
            st.markdown("##### Upload Fund List")
            
            # Create template for download
            template_df = pd.DataFrame({
                'Fund Name': ['Fund Name 1', 'Fund Name 2', 'Fund Name 3']
            })
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False)
            buffer.seek(0)
            
            tc1, tc2 = st.columns([1, 2])
            with tc1:
                st.download_button(
                    "📥 Download Template",
                    buffer,
                    "fund_selection_template.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with tc2:
                st.info("💡 Fill template with exact fund names from Fund Metrics, then upload.")
            
            uploaded_funds = st.file_uploader("Upload fund list", type=['xlsx'], key="portfolio_funds_upload")
            
            if uploaded_funds:
                try:
                    uploaded_df = pd.read_excel(uploaded_funds)
                    if 'Fund Name' in uploaded_df.columns:
                        available_funds = fund_metrics['FUNDO DE INVESTIMENTO'].tolist()
                        valid_funds = []
                        invalid_funds = []
                        
                        for _, row in uploaded_df.iterrows():
                            fund_name = row['Fund Name']
                            if fund_name in available_funds:
                                if fund_name not in st.session_state['selected_portfolio_funds']:
                                    valid_funds.append(fund_name)
                            else:
                                invalid_funds.append(fund_name)
                        
                        if invalid_funds:
                            st.warning(f"⚠️ Funds not found: {', '.join(invalid_funds[:5])}{'...' if len(invalid_funds) > 5 else ''}")
                        
                        if valid_funds:
                            st.success(f"✅ Found {len(valid_funds)} valid funds")
                            if st.button("➕ Add All Valid Funds", key="add_uploaded_funds"):
                                st.session_state['selected_portfolio_funds'].extend(valid_funds)
                                st.rerun()
                    else:
                        st.error("❌ Excel file must have a 'Fund Name' column")
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        else:
            # Original dropdown selection
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Fund selector
                available_funds = fund_metrics['FUNDO DE INVESTIMENTO'].tolist()
                selected_fund = st.selectbox(
                    "Choose a fund to add:",
                    options=[f for f in available_funds if f not in st.session_state['selected_portfolio_funds']],
                    key="fund_selector_portfolio"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➕ ADD FUND", use_container_width=True):
                    if selected_fund and selected_fund not in st.session_state['selected_portfolio_funds']:
                        st.session_state['selected_portfolio_funds'].append(selected_fund)
                        st.rerun()
        
        # Display selected funds
        if st.session_state['selected_portfolio_funds']:
            st.markdown(f"**Selected Funds ({len(st.session_state['selected_portfolio_funds'])}):**")
            
            # Create dataframe for display
            selected_df = fund_metrics[fund_metrics['FUNDO DE INVESTIMENTO'].isin(st.session_state['selected_portfolio_funds'])]
            display_df = selected_df[['FUNDO DE INVESTIMENTO', 'CATEGORIA BTG', 'SUBCATEGORIA BTG', 'STATUS']].copy()
            
            # Add remove buttons
            for idx, row in display_df.iterrows():
                fund_name = row['FUNDO DE INVESTIMENTO']
                status = row.get('STATUS', 'N/A')
                
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
                
                with col1:
                    # Highlight "Fechado" funds in red
                    if status == 'Fechado':
                        st.markdown(f"<span style='color: red;'>{fund_name}</span>", unsafe_allow_html=True)
                    else:
                        st.text(fund_name)
                with col2:
                    st.text(row['CATEGORIA BTG'])
                with col3:
                    st.text(row['SUBCATEGORIA BTG'])
                with col4:
                    # Display status with color
                    if status == 'Fechado':
                        st.markdown(f"<span style='color: red; font-weight: bold;'>{status}</span>", unsafe_allow_html=True)
                    else:
                        st.text(status)
                with col5:
                    if st.button("❌", key=f"remove_{fund_name}"):
                        st.session_state['selected_portfolio_funds'].remove(fund_name)
                        st.rerun()
            
            # Clear all button
            if st.button("🗑️ CLEAR ALL", use_container_width=False):
                st.session_state['selected_portfolio_funds'] = []
                st.rerun()
        else:
            st.info("👆 Select funds above to build your portfolio")
        
        # Require minimum 3 funds
        if len(st.session_state['selected_portfolio_funds']) < 3:
            st.warning("⚠️ Please select at least 3 funds to proceed with portfolio optimization")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # OPTIMIZATION PARAMETERS - Only show if we have enough funds
        if fund_metrics is not None and fund_details is not None and benchmarks is not None and len(st.session_state.get('selected_portfolio_funds', [])) >= 3:
            st.markdown("---")
        # ═══════════════════════════════════════════════════════════════════════════
        
        st.markdown("#### ⚙️ Step 2: Configure Optimization Parameters")
        
        # Minimum History Requirement
        with st.expander("📅 Minimum History Requirement", expanded=True):
            st.markdown("""
            **What it does:** Filters funds based on minimum historical data requirement.
            
            **Options:**
            - **252 days (1 year)**: Minimum for statistical significance
            - **504 days (2 years)**: Better for risk metrics
            - **756 days (3 years)**: Recommended for stable optimization
            
            Funds with insufficient history will be automatically excluded.
            """)
            
            min_history_days = st.slider(
                "Minimum trading days:",
                min_value=252,
                max_value=1260,
                value=504,
                step=21,
                format="%d days"
            )
        
        # DRO Configuration - V2
        st.markdown("---")
        dro_config = show_dro_configuration_panel()
        st.markdown("---")
        
        # Objective Function
        with st.expander("🎯 Optimization Objective", expanded=True):
            st.markdown("""
            **What it does:** Determines what the portfolio optimizer tries to maximize or minimize.
            
            **Objectives:**
            - **Max Information Ratio**: Maximize excess return relative to tracking error vs CDI (risk-adjusted outperformance)
            - **Max Excess Return**: Maximize total return above CDI benchmark
            - **Max Omega Ratio**: Maximize probability-weighted gains vs losses
            - **Min CVaR (95%)**: Minimize expected tail loss (Conditional Value at Risk)
            - **Min Volatility**: Minimize the annualized volatility            
            """)
            
            objective = st.selectbox(
                "Select objective:",
                options=['max_return', 'min_volatility', 'min_cvar', 'max_omega'],
                index=0,
                format_func=lambda x: {
                    'max_return': 'Max Return (Wasserstein DRO)',
                    'max_omega': 'Max Omega Ratio (Wasserstein DRO)',
                    'min_volatility': 'Min Volatility (Wasserstein DRO)',
                    'min_cvar': 'Min CVaR 95% (Wasserstein DRO)'
                }[x]
            )
        
        # Constraints
        with st.expander("📊 Portfolio Constraints (Optional)", expanded=False):
            st.markdown("""
            **What it does:** Applies limits on portfolio risk and allocations.
            
            All constraints are optional. The optimizer will try to satisfy them, but may relax them if infeasible.
            """)
            
            const_col1, const_col2 = st.columns(2)
            
            with const_col1:
                st.markdown("**Risk Constraints:**")
                
                use_max_vol = st.checkbox("Max Volatility")
                max_volatility = st.slider(
                    "Annual volatility limit (%):",
                    0.5, 15.0, 5.0, 0.01,
                    disabled=not use_max_vol
                ) / 100 if use_max_vol else None
                
                use_max_cvar = st.checkbox("Max CVaR (95%)")
                max_cvar = st.slider(
                    "Maximum expected tail loss (%):",
                    -5.0, 0.0, -1.0, 0.01,
                    disabled=not use_max_cvar
                ) / 100 if use_max_cvar else None
            
            with const_col2:
                st.markdown("**Return Constraints:**")
                
                use_min_annual = st.checkbox("Min Excess Return")
                min_annual_return = st.slider(
                    "Minimum Annual Return:",
                    5.0, 20.0, 10.0, 0.1,
                    disabled=not use_min_annual
                ) / 100 if use_min_annual else None
                
                use_min_omega = st.checkbox("Min Omega Ratio")
                min_omega = st.slider(
                    "Minimum gains-to-losses ratio:",
                    1.0, 10.0, 2.0, 0.01,
                    disabled=not use_min_omega
                ) if use_min_omega else None
        
        # Weight Constraints
        with st.expander("⚖️ Weight Constraints", expanded=True):
            st.markdown("""
            **What it does:** Controls individual fund and category allocations.
            
            **Constraint Types:**
            1. **Global Per-Fund Constraints:** Min/max applied to ALL funds (optimization-level)
            2. **Individual Per-Fund Constraints:** Custom limits for specific funds (optimization-level)
            3. **Global Per-Category Constraints:** Max applied to ALL categories (optimization-level)
            4. **Individual Per-Category Constraints:** Custom limits for specific categories (optimization-level)
            
            **Note:** Global minimum per fund is applied post-optimization with proportional redistribution.
            """)
            
            # Tab structure for different constraint types
            constraint_tab1, constraint_tab2, constraint_tab3, constraint_tab4 = st.tabs([
                "🔹 Global Fund", 
                "🔸 Individual Fund", 
                "🔹 Global Category",
                "🔸 Individual Category"
            ])
            
            # ═══ TAB 1: GLOBAL PER-FUND CONSTRAINTS ═══
            with constraint_tab1:
                st.markdown("**Global Constraints Applied to All Funds:**")
                
                global_fund_col1, global_fund_col2 = st.columns(2)
                
                with global_fund_col1:
                    min_weight_global = st.slider(
                        "Global Minimum weight per fund (%):",
                        0.0, 10.0, 1.0, 0.1,
                        help="Minimum allocation to each fund (applied post-optimization with redistribution)",
                        key="global_min_fund_weight"
                    ) / 100
                
                with global_fund_col2:
                    max_weight_global = st.slider(
                        "Global Maximum weight per fund (%):",
                        1.0, 100.0, 20.0, 0.5,
                        help="Maximum allocation to any single fund (hard constraint in optimization)",
                        key="global_max_fund_weight"
                    ) / 100
                
                st.info(f"📊 All funds will have: {min_weight_global*100:.1f}% ≤ weight ≤ {max_weight_global*100:.1f}%")
            
            # ═══ TAB 2: INDIVIDUAL PER-FUND CONSTRAINTS ═══
            with constraint_tab2:
                st.markdown("**Custom Constraints for Specific Funds:**")
                st.caption("Set individual min/max limits that override global constraints for specific funds")
                
                # Initialize session state for individual fund constraints
                if 'individual_fund_constraints' not in st.session_state:
                    st.session_state['individual_fund_constraints'] = {}
                
                # Get unique funds in selection
                if st.session_state['selected_portfolio_funds']:
                    # Create editable dataframe
                    individual_fund_data = []
                    for fund in st.session_state['selected_portfolio_funds']:
                        existing = st.session_state['individual_fund_constraints'].get(fund, {})
                        individual_fund_data.append({
                            'Fund': fund,
                            'Min Weight (%)': existing.get('min', min_weight_global * 100),
                            'Max Weight (%)': existing.get('max', max_weight_global * 100),
                            'Active': fund in st.session_state['individual_fund_constraints']
                        })
                    
                    individual_fund_df = pd.DataFrame(individual_fund_data)
                    
                    st.markdown("**Edit Individual Fund Constraints:**")
                    st.caption("💡 Check 'Active' to override global constraints for a specific fund")
                    
                    # Display editable table
                    edited_fund_df = st.data_editor(
                        individual_fund_df,
                        column_config={
                            "Fund": st.column_config.TextColumn("Fund Name", disabled=True),
                            "Min Weight (%)": st.column_config.NumberColumn(
                                "Min Weight (%)",
                                min_value=0.0,
                                max_value=100.0,
                                step=0.1,
                                format="%.2f"
                            ),
                            "Max Weight (%)": st.column_config.NumberColumn(
                                "Max Weight (%)",
                                min_value=0.0,
                                max_value=100.0,
                                step=0.1,
                                format="%.2f"
                            ),
                            "Active": st.column_config.CheckboxColumn("Active", default=False)
                        },
                        hide_index=True,
                        use_container_width=True,
                        key="individual_fund_editor"
                    )
                    
                    # Update session state
                    st.session_state['individual_fund_constraints'] = {}
                    for _, row in edited_fund_df.iterrows():
                        if row['Active']:
                            st.session_state['individual_fund_constraints'][row['Fund']] = {
                                'min': row['Min Weight (%)'],
                                'max': row['Max Weight (%)']
                            }
                    
                    # Show summary
                    active_count = len(st.session_state['individual_fund_constraints'])
                    if active_count > 0:
                        st.success(f"✅ {active_count} funds have individual constraints")
                    else:
                        st.info("ℹ️ No individual fund constraints active (using global constraints)")
                else:
                    st.info("👆 Select funds first to set individual constraints")
            
            # ═══ TAB 3: GLOBAL PER-CATEGORY CONSTRAINTS ═══
            with constraint_tab3:
                st.markdown("**Global Constraint Applied to All Categories:**")
                
                use_max_category_global = st.checkbox("Enable Global Max per Category", value=True, key="global_cat_enabled")
                max_per_category_global = st.slider(
                    "Global Max weight per category (%):",
                    10.0, 100.0, 50.0, 5.0,
                    disabled=not use_max_category_global,
                    help="Maximum allocation to any single category (hard constraint in optimization)",
                    key="global_max_cat_weight"
                ) / 100 if use_max_category_global else None
                
                if use_max_category_global:
                    st.info(f"📊 All categories limited to ≤ {max_per_category_global*100:.1f}%")
            
            # ═══ TAB 4: INDIVIDUAL PER-CATEGORY CONSTRAINTS ═══
            with constraint_tab4:
                st.markdown("**Custom Constraints for Specific Categories:**")
                st.caption("Set individual min/max limits that override global constraints for specific categories")
                
                # Initialize session state for individual category constraints
                if 'individual_category_constraints' not in st.session_state:
                    st.session_state['individual_category_constraints'] = {}
                
                # Get unique categories in selection
                if st.session_state['selected_portfolio_funds']:
                    selected_df = fund_metrics[fund_metrics['FUNDO DE INVESTIMENTO'].isin(st.session_state['selected_portfolio_funds'])]
                    unique_categories = selected_df['CATEGORIA BTG'].unique().tolist()
                    
                    # Create editable dataframe
                    individual_cat_data = []
                    for category in sorted(unique_categories):
                        existing = st.session_state['individual_category_constraints'].get(category, {})
                        individual_cat_data.append({
                            'Category': category,
                            'Min Weight (%)': existing.get('min', 0.0),
                            'Max Weight (%)': existing.get('max', max_per_category_global * 100 if max_per_category_global else 100.0),
                            'Active': category in st.session_state['individual_category_constraints']
                        })
                    
                    individual_cat_df = pd.DataFrame(individual_cat_data)
                    
                    st.markdown("**Edit Individual Category Constraints:**")
                    st.caption("💡 Check 'Active' to override global constraints for a specific category")
                    
                    # Display editable table
                    edited_cat_df = st.data_editor(
                        individual_cat_df,
                        column_config={
                            "Category": st.column_config.TextColumn("Category Name", disabled=True),
                            "Min Weight (%)": st.column_config.NumberColumn(
                                "Min Weight (%)",
                                min_value=0.0,
                                max_value=100.0,
                                step=1.0,
                                format="%.1f"
                            ),
                            "Max Weight (%)": st.column_config.NumberColumn(
                                "Max Weight (%)",
                                min_value=0.0,
                                max_value=100.0,
                                step=1.0,
                                format="%.1f"
                            ),
                            "Active": st.column_config.CheckboxColumn("Active", default=False)
                        },
                        hide_index=True,
                        use_container_width=True,
                        key="individual_category_editor"
                    )
                    
                    # Update session state
                    st.session_state['individual_category_constraints'] = {}
                    for _, row in edited_cat_df.iterrows():
                        if row['Active']:
                            st.session_state['individual_category_constraints'][row['Category']] = {
                                'min': row['Min Weight (%)'],
                                'max': row['Max Weight (%)']
                            }
                    
                    # Show summary
                    active_cat_count = len(st.session_state['individual_category_constraints'])
                    if active_cat_count > 0:
                        st.success(f"✅ {active_cat_count} categories have individual constraints")
                    else:
                        st.info("ℹ️ No individual category constraints active (using global constraints)")
                else:
                    st.info("👆 Select funds first to set individual category constraints")
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # RUN OPTIMIZATION
        # ═══════════════════════════════════════════════════════════════════════════
        
        if st.button("🚀 RUN PORTFOLIO OPTIMIZATION", use_container_width=True, type="primary"):
            
            with st.spinner("Preparing data..."):
                # Get CNPJs for selected funds
                selected_cnpjs = []
                for fund_name in st.session_state['selected_portfolio_funds']:
                    cnpj = fund_metrics[fund_metrics['FUNDO DE INVESTIMENTO'] == fund_name]['CNPJ'].iloc[0]
                    selected_cnpjs.append(standardize_cnpj(cnpj))
                
                # Get returns for all selected funds
                fund_returns_dict = {}
                valid_funds = []
                fund_start_dates = {}
                
                for cnpj, fund_name in zip(selected_cnpjs, st.session_state['selected_portfolio_funds']):
                    returns_result = get_fund_returns(fund_details, cnpj, period_months=None)
                    if returns_result is not None:
                        _, full_returns = returns_result
                        if len(full_returns) >= min_history_days:
                            fund_returns_dict[fund_name] = full_returns
                            valid_funds.append(fund_name)
                            fund_start_dates[fund_name] = full_returns.first_valid_index()
                        else:
                            st.warning(f"⚠️ Excluding {fund_name}: Insufficient history ({len(full_returns)} days < {min_history_days})")
                
                if len(valid_funds) < 3:
                    st.error("❌ Insufficient valid funds after history filtering. Please select more funds or reduce minimum history requirement.")
                    st.stop()
                
                st.success(f"✅ {len(valid_funds)} funds meet individual minimum history requirement")
                
                # Align all returns to common date range
                all_returns_df = pd.DataFrame(fund_returns_dict)
                
                # Find youngest fund (latest start date)
                youngest_start = max(fund_start_dates.values())
                
                # Show alignment info
                st.info(f"📅 Youngest fund starts: {youngest_start.date()}")
                
                # Trim to common period
                all_returns_df = all_returns_df.loc[youngest_start:]
                all_returns_df = all_returns_df.fillna(0)
                
                # FIXED: Check if aligned period meets minimum requirement
                aligned_length = len(all_returns_df)
                
                if aligned_length < min_history_days:
                    st.error(f"❌ After alignment to common period: {aligned_length} days < {min_history_days} required")
                    st.error(f"💡 The youngest fund started on {youngest_start.date()}, leaving insufficient common history.")
                    st.error(f"")
                    st.error(f"**Options to fix this:**")
                    st.error(f"1. **Reduce minimum history** to {aligned_length} days or less")
                    st.error(f"2. **Remove newer funds** - Check which fund started on {youngest_start.date()}")
                    st.error(f"3. **Select older funds** with longer track records")
                    
                    # Show fund start dates to help user identify the problem
                    with st.expander("📊 Fund Start Dates (click to see which funds are newest)"):
                        start_dates_df = pd.DataFrame([
                            {'Fund': fund, 'Start Date': date.date(), 'Days Available': len(all_returns_df)}
                            for fund, date in sorted(fund_start_dates.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(start_dates_df, use_container_width=True, hide_index=True)
                    
                    st.stop()
                
                st.success(f"✅ Aligned period: {aligned_length} days (meets {min_history_days} requirement)")
                
                # Get CDI benchmark for aligned period
                cdi_returns = benchmarks['CDI'].reindex(all_returns_df.index, method='ffill').fillna(0)
                
                st.success(f"✅ Data prepared: {len(all_returns_df)} days, {all_returns_df.shape[1]} funds")
            
            # Run DRO Optimization
            st.markdown("---")
            st.markdown("### 🎯 Running Wasserstein DRO Optimization")

            # In fund_analytics_app_v2.py, in the optimization section:

            # Build fund_categories dictionary
            fund_categories = {}
            for fund_name in all_returns_df.columns:
                fund_row = fund_metrics[fund_metrics['FUNDO DE INVESTIMENTO'] == fund_name]
                if len(fund_row) > 0:
                    fund_categories[fund_name] = fund_row['CATEGORIA BTG'].iloc[0]

            fund_subcategories = {}
            for fund_name in all_returns_df.columns:
                fund_row = fund_metrics[fund_metrics['FUNDO DE INVESTIMENTO'] == fund_name]
                if len(fund_row) > 0:
                    fund_subcategories[fund_name] = fund_row['SUBCATEGORIA BTG'].iloc[0]
            
            try:
                # Initialize DRO optimizer with V2 configuration
                optimizer = WassersteinDROOptimizer(
                    returns=all_returns_df,
                    fund_categories=fund_categories,
                    config=dro_config
                )
                
                # Build weight constraints dictionary (4 levels)
                weight_constraints = {
                    'global_fund': {
                        'min': min_weight_global,
                        'max': max_weight_global  
                    }
                }

                # Add individual fund constraints (per-fund specific limits)
                if 'individual_fund_constraints' in st.session_state and st.session_state['individual_fund_constraints']:
                    weight_constraints['individual_fund'] = st.session_state['individual_fund_constraints']
                    st.info(f"ℹ️ Using individual constraints for {len(st.session_state['individual_fund_constraints'])} specific funds")
                
                # Add individual category constraints
                individual_category_dict = {}
                if 'individual_category_constraints' in st.session_state and st.session_state['individual_category_constraints']:
                    for category, limits in st.session_state['individual_category_constraints'].items():
                        individual_category_dict[category] = {
                            'min': limits['min'] / 100,
                            'max': limits['max'] / 100
                        }
                    weight_constraints['individual_category'] = individual_category_dict
                
                # Add global category constraints
                if max_per_category_global is not None:
                    global_category_dict = {}
                    for category in set(fund_categories.values()):
                        global_category_dict[category] = {'max': max_per_category_global}
                    weight_constraints['global_category'] = global_category_dict
                
                # Validate constraints before running optimization
                validation_errors = []
                
                # Check global fund weight constraints
                if min_weight_global > max_weight_global:
                    validation_errors.append(f"❌ Global fund constraints: min ({min_weight_global*100:.1f}%) > max ({max_weight_global*100:.1f}%)")
                
                # Check individual category constraints
                if 'individual_category' in weight_constraints:
                    for category, limits in weight_constraints['individual_category'].items():
                        if limits['min'] > limits['max']:
                            validation_errors.append(f"❌ Category '{category}': min ({limits['min']*100:.1f}%) > max ({limits['max']*100:.1f}%)")
                    
                    # Check if sum of minimums exceeds 100%
                    total_cat_min = sum(lim['min'] for lim in weight_constraints['individual_category'].values())
                    if total_cat_min > 1.0:
                        validation_errors.append(f"❌ Sum of category minimums ({total_cat_min*100:.1f}%) exceeds 100%")
                
                # Display validation errors and stop if any exist
                if validation_errors:
                    st.error("### ⚠️ Constraint Validation Errors")
                    for error in validation_errors:
                        st.error(error)
                    st.warning("Please fix the constraint conflicts above before running optimization.")
                    st.stop()
                else:
                    st.success("✅ All constraints validated successfully")
                
                # Build portfolio constraints dictionary
                portfolio_constraints = {}
                if max_volatility is not None:
                    portfolio_constraints['max_volatility'] = max_volatility
                if max_cvar is not None:
                    portfolio_constraints['max_cvar'] = max_cvar
                if min_annual_return is not None:
                    portfolio_constraints['min_annual_return'] = min_annual_return
                if min_omega is not None:
                    portfolio_constraints['min_omega'] = min_omega
                
                # Run optimization with fixed optimizer
                with st.spinner(f"Optimizing portfolio (objective: {objective})..."):
                    progress_bar = st.progress(0)
                    result = optimizer.optimize(
                        objective=objective,
                        constraints=portfolio_constraints,
                        weight_constraints=weight_constraints
                    )
                    progress_bar.progress(100)
                
                # Display results using V2 display function
                if result.success:
                    display_dro_results_v2(
                        result,
                        all_returns_df,
                        cdi_returns,
                        fund_categories,
                        fund_subcategories
                    )
                else:
                    st.error(f"❌ Optimization failed: {result.solver_status}")
                    with st.expander("📋 Optimization Log"):
                        for log_entry in result.optimization_log:
                            st.text(log_entry)

                    metrics = result['metrics']
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col4:
                        sharpe = (metrics['annual_return'] / metrics['annual_volatility']) if metrics['annual_volatility'] > 0 else 0
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        st.metric("CVaR (95%)", f"{metrics['cvar_95']*100:.2f}%")
                    
                    st.info("📌 Scroll down to see detailed portfolio analysis including returns, risk metrics, and benchmark exposures.")
                    
                    # Show optimization log
                    if 'optimization_log' in result:
                        with st.expander("📋 Optimization Log", expanded=True):
                            for log_entry in result['optimization_log']:
                                st.text(log_entry)
                
            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        if 'portfolio_result' in st.session_state and st.session_state['portfolio_result']:
            
            result = st.session_state['portfolio_result']
            portfolio_returns = result['portfolio_returns']
            weights = result['final_weights']
            all_returns_df = st.session_state['portfolio_returns_df']
            cdi_returns = st.session_state['portfolio_cdi']
            fund_categories = st.session_state['fund_categories']
            fund_subcategories = st.session_state['fund_subcategories']
            
            st.markdown("---")
            st.markdown("## 📈 PORTFOLIO ANALYSIS")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # WEIGHT VISUALIZATION - PIE CHARTS WITH BUTTONS
            # ═══════════════════════════════════════════════════════════════════════════
            
            st.markdown("### 🎨 Portfolio Composition")
            
            # Buttons to switch view
            view_col1, view_col2, view_col3 = st.columns(3)
            
            with view_col1:
                if st.button("📊 By Investment Fund", use_container_width=True):
                    st.session_state['portfolio_view'] = 'fund'
            
            with view_col2:
                if st.button("📁 By Category", use_container_width=True):
                    st.session_state['portfolio_view'] = 'category'
            
            with view_col3:
                if st.button("📂 By Subcategory", use_container_width=True):
                    st.session_state['portfolio_view'] = 'subcategory'
            
            # Default view
            if 'portfolio_view' not in st.session_state:
                st.session_state['portfolio_view'] = 'fund'
            
            # Display selected view
            fig_pie = create_portfolio_pie_chart(
                weights,
                st.session_state['portfolio_view'],
                fund_categories,
                fund_subcategories
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Summary tables
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("#### Category Breakdown")
                category_weights = {}
                for fund, weight in weights.items():
                    cat = fund_categories.get(fund, 'Unknown')
                    category_weights[cat] = category_weights.get(cat, 0) + weight
                
                cat_df = pd.DataFrame({
                    'Category': list(category_weights.keys()),
                    'Weight %': [w*100 for w in category_weights.values()]
                }).sort_values('Weight %', ascending=False)
                
                st.dataframe(
                    cat_df.style.format({'Weight %': '{:.2f}%'}),
                    use_container_width=True,
                    hide_index=True
                )
            
            with summary_col2:
                st.markdown("#### Subcategory Breakdown")
                subcat_weights = {}
                for fund, weight in weights.items():
                    subcat = fund_subcategories.get(fund, 'Unknown')
                    subcat_weights[subcat] = subcat_weights.get(subcat, 0) + weight
                
                subcat_df = pd.DataFrame({
                    'Subcategory': list(subcat_weights.keys()),
                    'Weight %': [w*100 for w in subcat_weights.values()]
                }).sort_values('Weight %', ascending=False)
                
                st.dataframe(
                    subcat_df.style.format({'Weight %': '{:.2f}%'}),
                    use_container_width=True,
                    hide_index=True
                )
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # RETURNS ANALYSIS
            # ═══════════════════════════════════════════════════════════════════════════
            
            st.markdown("### 📈 Returns Analysis")
            
            # Period selection
            period_options = {'3M': 3, '6M': 6, '12M': 12, '24M': 24, '36M': 36, 'All': None}
            selected_period = st.selectbox("Select Period:", list(period_options.keys()), index=5, key="portfolio_period")
            
            period_months = period_options[selected_period]
            
            if period_months:
                cutoff_date = portfolio_returns.index[-1] - pd.DateOffset(months=period_months)
                period_returns = portfolio_returns[portfolio_returns.index >= cutoff_date]
                period_cdi = cdi_returns[cdi_returns.index >= cutoff_date]
            else:
                period_returns = portfolio_returns
                period_cdi = cdi_returns
            
            # Cumulative returns chart
            portfolio_cum = calculate_cumulative_returns(period_returns) * 100
            cdi_cum = calculate_cumulative_returns(period_cdi) * 100
            
            fig_returns = go.Figure()
            
            fig_returns.add_trace(go.Scatter(
                x=portfolio_cum.index,
                y=portfolio_cum.values,
                name='Portfolio',
                line=dict(color='#D4AF37', width=3),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
            
            fig_returns.add_trace(go.Scatter(
                x=cdi_cum.index,
                y=cdi_cum.values,
                name='CDI',
                line=dict(color='#00CED1', width=2),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
            
            fig_returns.update_layout(
                title=f'Cumulative Returns - {selected_period}',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                template=PLOTLY_TEMPLATE,
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_returns, use_container_width=True)
            
            # Calculate and display metrics for different periods
            periods_list = ['3M', '6M', '12M', '24M', '36M', 'Total']
            metrics_data = {'Period': [], 'Portfolio Return': [], 'CDI Return': [], 'Excess Return': []}
            
            for period_name in periods_list:
                if period_name == 'Total':
                    p_ret = (1 + portfolio_returns).prod() - 1
                    c_ret = (1 + cdi_returns).prod() - 1
                else:
                    months = int(period_name.replace('M', ''))
                    cutoff = portfolio_returns.index[-1] - pd.DateOffset(months=months)
                    
                    if cutoff < portfolio_returns.index[0]:
                        continue
                    
                    p_ret = (1 + portfolio_returns[portfolio_returns.index >= cutoff]).prod() - 1
                    c_ret = (1 + cdi_returns[cdi_returns.index >= cutoff]).prod() - 1
                
                metrics_data['Period'].append(period_name)
                metrics_data['Portfolio Return'].append(f"{p_ret*100:.2f}%")
                metrics_data['CDI Return'].append(f"{c_ret*100:.2f}%")
                metrics_data['Excess Return'].append(f"{(p_ret / c_ret)*100:.2f}%")
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # MONTHLY RETURNS CALENDAR
            # ═══════════════════════════════════════════════════════════════════════════
            
            st.markdown("### 📅 Monthly Returns Calendar")
            
            # Comparison method selection
            comparison_method = st.selectbox(
                "Comparison Method:",
                options=['Relative Performance', 'Percentage Points', 'Benchmark Performance'],
                index=0,
                key="portfolio_comparison_method"
            )
            
            # Create monthly returns table
            monthly_table = create_monthly_returns_table(
                portfolio_returns,
                cdi_returns,
                comparison_method
            )
            
            # Style the table as HTML
            styled_html = style_monthly_returns_table(monthly_table, comparison_method)
            
            # Display HTML table
            st.markdown(styled_html, unsafe_allow_html=True)
            
            # Add explanation
            with st.expander("ℹ️ Understanding the Monthly Returns Calendar"):
                explanation_text = f"""
                **How to read this table:**
                
                For each year, {"two" if comparison_method == "Benchmark Performance" else "three"} rows are displayed:
                - **Investment Fund**: Monthly returns of the portfolio
                """
                
                if comparison_method != 'Benchmark Performance':
                    explanation_text += f"- **Benchmark**: Monthly returns of CDI\n"
                
                explanation_text += f"""- **{comparison_method}**: The comparison metric between portfolio and CDI
                
                **Comparison Methods:**
                - **Relative Performance**: Ratio showing portfolio performance relative to benchmark
                - **Percentage Points**: Portfolio return minus CDI return in absolute terms
                - **Benchmark Performance**: Displays CDI's monthly returns for reference
                
                **Columns:**
                - **Year**: The calendar year
                - **Type**: Portfolio, Benchmark (if shown), or Comparison metric
                - **Jan-Dec**: Monthly returns/comparison for each month
                - **YTD**: Year-to-date accumulated performance
                - **Total**: Cumulative performance since the beginning
                
                **Visual Guide:**
                - Negative values are displayed in **red** for easy identification
                - Bold gold borders separate year groups and column sections
                """
                
                st.markdown(explanation_text)
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # RISK-ADJUSTED PERFORMANCE
            # ═══════════════════════════════════════════════════════════════════════════
            
            st.markdown("### ⚖️ Risk-Adjusted Performance")
            
            # Frequency selection
            frequency_choice = st.radio(
                "Select frequency for Omega, Rachev, VaR and CVaR analysis:",
                options=['Daily', 'Weekly', 'Monthly'],
                horizontal=True,
                key="portfolio_freq"
            )
            
            if frequency_choice == 'Daily':
                analysis_returns = portfolio_returns
            elif frequency_choice == 'Weekly':
                analysis_returns = portfolio_returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
            else:
                analysis_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            st.markdown("---")
            
            # Omega Section
            st.markdown("#### Omega Ratio")
            
            omega_chart_col, omega_gauge_col = st.columns([2, 1])
            
            with omega_chart_col:
                fig_omega = create_omega_cdf_chart(analysis_returns, threshold=0, frequency=frequency_choice.lower())
                st.plotly_chart(fig_omega, use_container_width=True)
            
            with omega_gauge_col:
                omega_val = PortfolioMetrics.omega_ratio(analysis_returns)
                fig_omega_gauge = create_omega_gauge(omega_val, frequency=frequency_choice)
                st.plotly_chart(fig_omega_gauge, use_container_width=True)
            
            st.markdown("---")
            
            # Rachev / VaR / CVaR Section
            st.markdown("#### Rachev Ratio & Tail Risk")
            
            rachev_chart_col, rachev_metrics_col = st.columns([2, 1])
            
            with rachev_chart_col:
                var_val = PortfolioMetrics.var(analysis_returns, 0.95)
                cvar_val = PortfolioMetrics.cvar(analysis_returns, 0.95)
                
                fig_rachev = create_combined_rachev_var_chart(
                    analysis_returns, var_val, cvar_val, frequency=frequency_choice.lower()
                )
                st.plotly_chart(fig_rachev, use_container_width=True)
            
            with rachev_metrics_col:
                # Calculate Rachev ratio
                returns_pct = analysis_returns * 100
                lower_threshold = np.percentile(returns_pct, 5)
                upper_threshold = np.percentile(returns_pct, 95)
                
                expected_loss = abs(returns_pct[returns_pct <= lower_threshold].mean())
                expected_gain = returns_pct[returns_pct >= upper_threshold].mean()
                
                rachev_val = expected_gain / expected_loss if expected_loss > 0 else np.inf
                
                fig_rachev_gauge = create_rachev_gauge(rachev_val, frequency=frequency_choice)
                st.plotly_chart(fig_rachev_gauge, use_container_width=True)
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # SHARPE RATIO ANALYSIS
            # ═══════════════════════════════════════════════════════════════════════════
            
            st.markdown("### 📊 Sharpe Ratio Analysis")
            
            sharpe_chart_col, sharpe_metrics_col = st.columns([3, 1])
            
            with sharpe_chart_col:
                fig_sharpe = create_rolling_sharpe_chart(portfolio_returns, window_months=12)
                st.plotly_chart(fig_sharpe, use_container_width=True)
            
            with sharpe_metrics_col:
                # Calculate Sharpe for different periods
                for period_name, months in [('12M', 12), ('24M', 24), ('36M', 36), ('Total', None)]:
                    if months:
                        cutoff = portfolio_returns.index[-1] - pd.DateOffset(months=months)
                        if cutoff >= portfolio_returns.index[0]:
                            period_ret = portfolio_returns[portfolio_returns.index >= cutoff]
                        else:
                            continue
                    else:
                        period_ret = portfolio_returns
                    
                    sharpe = PortfolioMetrics.sharpe_ratio(period_ret)
                    st.metric(f"Sharpe {period_name}", f"{sharpe:.2f}")
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # RISK METRICS DASHBOARD
            # ═══════════════════════════════════════════════════════════════════════════
            
            st.markdown("### 🎯 Risk Metrics Dashboard")
            
            # Volatility section
            st.markdown("#### Volatility Analysis")
            
            vol_chart_col, vol_metrics_col = st.columns([3, 1])
            
            with vol_chart_col:
                fig_vol = create_rolling_vol_chart(portfolio_returns, window_months=12)
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with vol_metrics_col:
                for period_name, months in [('12M', 12), ('24M', 24), ('36M', 36), ('Total', None)]:
                    if months:
                        cutoff = portfolio_returns.index[-1] - pd.DateOffset(months=months)
                        if cutoff >= portfolio_returns.index[0]:
                            period_ret = portfolio_returns[portfolio_returns.index >= cutoff]
                        else:
                            continue
                    else:
                        period_ret = portfolio_returns
                    
                    vol = PortfolioMetrics.annualized_volatility(period_ret)
                    st.metric(f"Vol {period_name}", f"{vol*100:.2f}%")
            
            st.markdown("---")
            
            # Drawdown section
            st.markdown("#### Drawdown Analysis")
            
            dd_chart_col, dd_metrics_col = st.columns([3, 1])
            
            with dd_chart_col:
                fig_underwater, max_dd_info = create_underwater_plot(portfolio_returns)
                st.plotly_chart(fig_underwater, use_container_width=True)
            
            with dd_metrics_col:
                mdd = PortfolioMetrics.max_drawdown(portfolio_returns)
                st.metric("Max Drawdown", f"{mdd*100:.2f}%")
                
                # Display CDaR metric
                if max_dd_info and 'cdar_95' in max_dd_info:
                    cdar_95 = max_dd_info['cdar_95']
                    st.metric("CDaR (95%)", f"{cdar_95:.2f}%",
                             help="Conditional Drawdown at Risk: Average of worst 5% drawdowns")
                
                # Calculate MDD duration - find the LONGEST drawdown period
                drawdowns = PortfolioMetrics.drawdown_series(portfolio_returns)
                
                if len(drawdowns) > 0 and drawdowns.min() < -0.01:
                    cumulative = (1 + portfolio_returns).cumprod()
                    running_max = cumulative.expanding().max()
                    
                    # Identify all drawdown periods
                    in_drawdown = drawdowns < -0.001  # Consider drawdowns > 0.1%
                    
                    if in_drawdown.any():
                        # Find all drawdown periods (contiguous sequences)
                        drawdown_periods = []
                        start_idx = None
                        
                        for i in range(len(in_drawdown)):
                            if in_drawdown.iloc[i] and start_idx is None:
                                # Start of a new drawdown
                                start_idx = i
                            elif not in_drawdown.iloc[i] and start_idx is not None:
                                # End of drawdown (recovered)
                                end_idx = i - 1
                                duration = (in_drawdown.index[end_idx] - in_drawdown.index[start_idx]).days
                                max_dd_in_period = drawdowns.iloc[start_idx:end_idx+1].min()
                                
                                drawdown_periods.append({
                                    'start': in_drawdown.index[start_idx],
                                    'end': in_drawdown.index[end_idx],
                                    'duration': duration,
                                    'max_dd': max_dd_in_period
                                })
                                start_idx = None
                        
                        # Check if still in drawdown at the end
                        if start_idx is not None:
                            duration = (in_drawdown.index[-1] - in_drawdown.index[start_idx]).days
                            max_dd_in_period = drawdowns.iloc[start_idx:].min()
                            drawdown_periods.append({
                                'start': in_drawdown.index[start_idx],
                                'end': in_drawdown.index[-1],
                                'duration': duration,
                                'max_dd': max_dd_in_period,
                                'ongoing': True
                            })
                        
                        if drawdown_periods:
                            # Find the drawdown with longest duration
                            longest_dd = max(drawdown_periods, key=lambda x: x['duration'])
                            
                            # Also identify the deepest drawdown
                            deepest_dd = min(drawdown_periods, key=lambda x: x['max_dd'])
                            
                            # Display the longest duration drawdown
                            if longest_dd.get('ongoing', False):
                                st.metric(
                                    "Longest DD Duration", 
                                    f"{longest_dd['duration']} days",
                                    help="Longest drawdown period (still ongoing)"
                                )
                                st.caption(f"From {longest_dd['start'].date()} (ongoing)")
                            else:
                                st.metric(
                                    "Longest DD Duration", 
                                    f"{longest_dd['duration']} days",
                                    help="Longest time to recover from a drawdown"
                                )
                                st.caption(f"From {longest_dd['start'].date()} to {longest_dd['end'].date()}")
                            
                            # Show if longest != deepest
                            if longest_dd['start'] != deepest_dd['start']:
                                st.caption(f"⚠️ Note: Deepest DD ({deepest_dd['max_dd']*100:.2f}%) occurred from {deepest_dd['start'].date()}")
                        else:
                            st.metric("Longest DD Duration", "N/A")
                    else:
                        st.metric("Longest DD Duration", "0 days")
                else:
                    st.metric("Longest DD Duration", "0 days")
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # BENCHMARK EXPOSURES
            # ═══════════════════════════════════════════════════════════════════════════
            
            st.markdown("### 🌐 Benchmark Exposures")
            
            available_benchmarks = ['CDI', 'USDBRL', 'GOLD', 'IBOVESPA', 'SP500', 'BITCOIN']
            default_exposure = ['CDI', 'IBOVESPA'] if all(b in benchmarks.columns for b in ['CDI', 'IBOVESPA']) else ['CDI']
            
            selected_exposure_benches = st.multiselect(
                "Select Benchmarks for Exposure Analysis:",
                options=[b for b in available_benchmarks if b in benchmarks.columns],
                default=default_exposure,
                key="portfolio_exposure"
            )
            
            if selected_exposure_benches:
                exposure_data = []
                
                for bench in selected_exposure_benches:
                    # Calculate rolling copula metrics to match time series analysis
                    with st.spinner(f'Calculating exposure for {bench}...'):
                        copula_results = estimate_rolling_copula_for_chart(
                            portfolio_returns,
                            benchmarks[bench],
                            window=250
                        )
                        
                        if copula_results is not None:
                            # Last window values (most recent)
                            last_kendall = copula_results['kendall_tau'].iloc[-1]
                            last_tail_lower = copula_results['tail_lower'].iloc[-1]
                            last_tail_upper = copula_results['tail_upper'].iloc[-1]
                            last_asymmetry = copula_results['asymmetry_index'].iloc[-1]
                            
                            # Average values across all windows
                            avg_kendall = copula_results['kendall_tau'].mean()
                            avg_tail_lower = copula_results['tail_lower'].mean()
                            avg_tail_upper = copula_results['tail_upper'].mean()
                            avg_asymmetry = copula_results['asymmetry_index'].mean()
                            
                            # Add last window row
                            exposure_data.append({
                                'Benchmark': f'{bench} - Last Window',
                                'Kendall Tau': last_kendall,
                                'Tail Lower': last_tail_lower,
                                'Tail Upper': last_tail_upper,
                                'Asymmetry': last_asymmetry
                            })
                            
                            # Add average row
                            exposure_data.append({
                                'Benchmark': f'{bench} - Average',
                                'Kendall Tau': avg_kendall,
                                'Tail Lower': avg_tail_lower,
                                'Tail Upper': avg_tail_upper,
                                'Asymmetry': avg_asymmetry
                            })
                        else:
                            # Insufficient data - use full data calculation as fallback
                            bench_returns = benchmarks[bench].reindex(portfolio_returns.index, method='ffill').fillna(0)
                            
                            u = to_empirical_cdf(portfolio_returns)
                            v = to_empirical_cdf(bench_returns)
                            
                            tau = stats.kendalltau(u.values, v.values)[0]
                            
                            theta_lower, _ = estimate_gumbel_270_parameter(u.values, v.values)
                            lambda_lower, _ = gumbel_270_tail_dependence(theta_lower)
                            
                            theta_upper, _ = estimate_gumbel_180_parameter(u.values, v.values)
                            _, lambda_upper = gumbel_180_tail_dependence(theta_upper)
                            
                            asymmetry = (lambda_lower - lambda_upper) / (lambda_lower + lambda_upper) if (lambda_lower + lambda_upper) > 0 else 0
                            
                            exposure_data.append({
                                'Benchmark': f'{bench} - Full Period',
                                'Kendall Tau': tau,
                                'Tail Lower': lambda_lower,
                                'Tail Upper': lambda_upper,
                                'Asymmetry': asymmetry
                            })
                
                exposure_df = pd.DataFrame(exposure_data)
                
                st.dataframe(
                    exposure_df.style.format(
                        {col: "{:.4f}" for col in exposure_df.columns if col != 'Benchmark'}
                    ).background_gradient(
                        cmap='RdYlGn',
                        subset=[col for col in exposure_df.columns if col != 'Benchmark'],
                        vmin=-1, vmax=1
                    ),
                    use_container_width=True,
                    hide_index=True
                )
                
                with st.expander("📚 Exposure Metrics Guide"):
                    st.markdown("""
                    **Kendall Tau**: Overall correlation (-1 to +1)
                    - Positive: moves together | Zero: independent | Negative: moves opposite
                    
                    **Tail Lower**: Crash correlation (0 to 1)
                    - High: portfolio crashes together with benchmark
                    
                    **Tail Upper**: Boom correlation (0 to 1)
                    - High: portfolio rallies together with benchmark
                    
                    **Asymmetry**: Crash vs Boom bias (-1 to +1)
                    - Positive: stronger crash correlation | Negative: stronger boom correlation
                    """)
            else:
                st.info("Select at least one benchmark to view exposures")

            # ═══════════════════════════════════════════════════════════════════════════
            # EXPOSURE TIME SERIES ANALYSIS FOR PORTFOLIO
            # ═══════════════════════════════════════════════════════════════════════════

            st.markdown("---")
            st.markdown("### 📈 Portfolio Exposure Time Series Analysis")
            st.info("💡 Select a benchmark below to visualize the evolution of portfolio exposure metrics over time")

            # Benchmark selection for time series
            portfolio_ts_benchmarks = ['None'] + [b for b in available_benchmarks if b in benchmarks.columns]
            selected_portfolio_ts_benchmark = st.selectbox(
                "Select Benchmark for Time Series:",
                options=portfolio_ts_benchmarks,
                index=0,
                key="portfolio_ts_benchmark_selector"
            )

            if selected_portfolio_ts_benchmark != 'None' and selected_portfolio_ts_benchmark in benchmarks.columns:
                with st.spinner(f'Calculating portfolio exposure time series for {selected_portfolio_ts_benchmark}...'):
                    # Calculate rolling copula metrics for portfolio
                    copula_results = estimate_rolling_copula_for_chart(
                        portfolio_returns,
                        benchmarks[selected_portfolio_ts_benchmark],
                        window=250
                    )
                    
                    if copula_results is not None:
                        # Calculate current and average values
                        current_kendall = copula_results['kendall_tau'].iloc[-1]
                        avg_kendall = copula_results['kendall_tau'].mean()
                        
                        current_tail_lower = copula_results['tail_lower'].iloc[-1]
                        avg_tail_lower = copula_results['tail_lower'].mean()
                        
                        current_tail_upper = copula_results['tail_upper'].iloc[-1]
                        avg_tail_upper = copula_results['tail_upper'].mean()
                        
                        current_asymmetry = copula_results['asymmetry_index'].iloc[-1]
                        avg_asymmetry = copula_results['asymmetry_index'].mean()
                        
                        # Create 2x2 grid of charts
                        st.markdown(f"##### Portfolio Exposure Evolution - {selected_portfolio_ts_benchmark}")
                        
                        # Row 1: Kendall Tau and Asymmetry
                        row1_col1, row1_col2 = st.columns(2)
                        
                        with row1_col1:
                            fig_kendall = create_exposure_time_series_chart(
                                copula_results,
                                'kendall_tau',
                                current_kendall,
                                avg_kendall,
                                selected_portfolio_ts_benchmark
                            )
                            st.plotly_chart(fig_kendall, use_container_width=True)
                        
                        with row1_col2:
                            fig_asymmetry = create_exposure_time_series_chart(
                                copula_results,
                                'asymmetry_index',
                                current_asymmetry,
                                avg_asymmetry,
                                selected_portfolio_ts_benchmark
                            )
                            st.plotly_chart(fig_asymmetry, use_container_width=True)
                        
                        # Row 2: Lower Tail and Upper Tail
                        row2_col1, row2_col2 = st.columns(2)
                        
                        with row2_col1:
                            fig_tail_lower = create_exposure_time_series_chart(
                                copula_results,
                                'tail_lower',
                                current_tail_lower,
                                avg_tail_lower,
                                selected_portfolio_ts_benchmark
                            )
                            st.plotly_chart(fig_tail_lower, use_container_width=True)
                        
                        with row2_col2:
                            fig_tail_upper = create_exposure_time_series_chart(
                                copula_results,
                                'tail_upper',
                                current_tail_upper,
                                avg_tail_upper,
                                selected_portfolio_ts_benchmark
                            )
                            st.plotly_chart(fig_tail_upper, use_container_width=True)
                        
                        # Summary metrics
                        st.markdown("##### Summary Statistics")
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        
                        with summary_col1:
                            st.metric(
                                "Kendall Tau (Last)",
                                f"{current_kendall:.4f}",
                                delta=f"{(current_kendall - avg_kendall):.4f}"
                            )
                        
                        with summary_col2:
                            st.metric(
                                "Lower Tail (Last)",
                                f"{current_tail_lower:.4f}",
                                delta=f"{(current_tail_lower - avg_tail_lower):.4f}"
                            )
                        
                        with summary_col3:
                            st.metric(
                                "Upper Tail (Last)",
                                f"{current_tail_upper:.4f}",
                                delta=f"{(current_tail_upper - avg_tail_upper):.4f}"
                            )
                        
                        with summary_col4:
                            st.metric(
                                "Asymmetry (Last)",
                                f"{current_asymmetry:.4f}",
                                delta=f"{(current_asymmetry - avg_asymmetry):.4f}"
                            )
                        
                        # Interpretation guide
                        with st.expander("📖 How to Read These Charts"):
                            st.markdown("""
                            **Chart Elements:**
                            - **Yellow Line**: Time series of the exposure metric across all rolling windows
                            - **Red Dot**: Most recent value (last window)
                            - **Blue Line**: Average value across all windows (horizontal reference)
                            
                            **What This Shows:**
                            - The evolution of the portfolio's relationship with the selected benchmark over time
                            - Whether current exposure is above or below historical average
                            - Trends and changes in the nature of the dependence
                            
                            **Interpreting Trends:**
                            - **Kendall Tau**: Overall correlation strength - higher means more synchronized movements
                            - **Lower Tail**: Crash correlation - higher means portfolio tends to fall when benchmark crashes
                            - **Upper Tail**: Boom correlation - higher means portfolio tends to rise when benchmark rallies
                            - **Asymmetry**: Positive = stronger crash link, Negative = stronger boom link
                            
                            **Rolling Window**: The calculations use a 250-day (≈1 year) rolling window, providing a dynamic view of how the relationship evolves over time.
                            """)
                    else:
                        st.warning("⚠️ Insufficient data to calculate exposure time series for this benchmark")
            else:
                st.info("📊 Select a benchmark to enable portfolio exposure time series analysis")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 5: RECOMMENDED PORTFOLIO
    # ═══════════════════════════════════════════════════════════════════════════════
    
    with tab5:
        st.title("💼 RECOMMENDED PORTFOLIO")
        st.markdown("### Create and analyze your recommended investment fund portfolio")
        st.markdown("---")
        
        if fund_metrics is None or fund_details is None or benchmarks is None:
            st.warning("⚠️ Please upload all required data files to use Recommended Portfolio")
        else:
            if 'recommended_portfolio' not in st.session_state:
                st.session_state['recommended_portfolio'] = {}
            if 'recommended_portfolio_saved' not in st.session_state:
                st.session_state['recommended_portfolio_saved'] = False
            if 'temp_portfolio' not in st.session_state:
                st.session_state['temp_portfolio'] = {}
            
            rec_tab1, rec_tab2, rec_tab3 = st.tabs(["📊 Portfolio Analysis", "📈 Investment Fund Analysis", "📚 Book Analysis"])
            
            # ═══════════════════════════════════════════════════════════════════════════
            # SECONDARY TAB 1: PORTFOLIO ANALYSIS
            # ═══════════════════════════════════════════════════════════════════════════
            
            with rec_tab1:
                st.markdown("### 📊 Portfolio Analysis")
                st.markdown("---")
                st.markdown("#### 📝 Create Your Portfolio")
                
                creation_method = st.radio("Choose method:", ["📤 Upload Excel File", "🔍 Search and Select Funds"], horizontal=True, key="rec_method")
                
                if creation_method == "📤 Upload Excel File":
                    st.markdown("---")
                    template_df = create_portfolio_template()
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        template_df.to_excel(writer, index=False)
                    buffer.seek(0)
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.download_button("📥 Download Template", buffer, "portfolio_template.xlsx", use_container_width=True)
                    with c2:
                        st.info("💡 Fill template with fund names and allocations, then upload.")
                    
                    uploaded = st.file_uploader("Upload portfolio", type=['xlsx'], key="rec_upload")
                    if uploaded:
                        try:
                            pdf = pd.read_excel(uploaded)
                            if 'Fund Name' in pdf.columns and 'Allocation (%)' in pdf.columns:
                                avail = fund_metrics['FUNDO DE INVESTIMENTO'].tolist()
                                valid, invalid = {}, []
                                for _, r in pdf.iterrows():
                                    if r['Fund Name'] in avail:
                                        valid[r['Fund Name']] = r['Allocation (%)']
                                    else:
                                        invalid.append(r['Fund Name'])
                                if invalid:
                                    st.warning(f"Not found: {', '.join(invalid)}")
                                if valid:
                                    st.success(f"✅ {len(valid)} valid funds")
                                    if st.button("💾 Save Portfolio", key="save_up"):
                                        st.session_state['recommended_portfolio'] = valid
                                        st.session_state['recommended_portfolio_saved'] = True
                                        st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.markdown("---")
                    avail = fund_metrics['FUNDO DE INVESTIMENTO'].tolist()
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        sel = st.selectbox("Fund:", [f for f in avail if f not in st.session_state['temp_portfolio']], key="rec_sel")
                    with c2:
                        alloc = st.number_input("Alloc (%)", 0.1, 100.0, 10.0, 0.5, key="rec_alloc")
                    with c3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("➕ Add"):
                            st.session_state['temp_portfolio'][sel] = alloc
                            st.rerun()
                    
                    if st.session_state['temp_portfolio']:
                        for fn in list(st.session_state['temp_portfolio'].keys()):
                            c1, c2, c3 = st.columns([3, 1, 1])
                            with c1:
                                st.text(fn)
                            with c2:
                                st.session_state['temp_portfolio'][fn] = st.number_input("", 0.1, 100.0, float(st.session_state['temp_portfolio'][fn]), 0.5, key=f"a_{fn}", label_visibility="collapsed")
                            with c3:
                                if st.button("❌", key=f"r_{fn}"):
                                    del st.session_state['temp_portfolio'][fn]
                                    st.rerun()
                        
                        st.metric("Total", f"{sum(st.session_state['temp_portfolio'].values()):.1f}%")
                        if st.button("💾 Save Portfolio", key="save_sel"):
                            st.session_state['recommended_portfolio'] = st.session_state['temp_portfolio'].copy()
                            st.session_state['recommended_portfolio_saved'] = True
                            st.rerun()
                
                st.markdown("---")
                
                # ═══════════════════════════════════════════════════════════════════════════
                # SUPABASE PORTFOLIO STORAGE
                # ═══════════════════════════════════════════════════════════════════════════
                
                with st.expander("☁️ Cloud Portfolio Storage (Supabase)", expanded=False):
                    supabase_client = get_supabase_client()
                    
                    if not SUPABASE_AVAILABLE:
                        st.warning("⚠️ Supabase library not installed. Run: `pip install supabase`")
                    elif not supabase_client:
                        st.info("""
                        **Configure Supabase to save portfolios to the cloud:**
                        
                        1. Create a Supabase account at https://supabase.com
                        2. Create a new project
                        3. Run the SQL below to create the table
                        4. Set `SUPABASE_URL` and `SUPABASE_KEY` in the app config or use Streamlit secrets
                        """)
                        
                        st.code("""
-- SQL to create the recommended_portfolios table in Supabase
CREATE TABLE recommended_portfolios (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    portfolio_name TEXT NOT NULL,
    portfolio_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, portfolio_name)
);

-- Create index for faster lookups
CREATE INDEX idx_portfolios_user_id ON recommended_portfolios(user_id);

-- Enable Row Level Security (optional, for multi-user)
ALTER TABLE recommended_portfolios ENABLE ROW LEVEL SECURITY;

-- Policy to allow all operations (adjust for production)
CREATE POLICY "Allow all operations" ON recommended_portfolios
    FOR ALL USING (true) WITH CHECK (true);
                        """, language="sql")
                    else:
                        st.success("✅ Connected to Supabase")
                        
                        # Get current user (from session or default)
                        current_user = st.session_state.get('username', 'default')
                        
                        save_col, load_col = st.columns(2)
                        
                        with save_col:
                            st.markdown("##### 💾 Save Current Portfolio")
                            
                            if st.session_state.get('recommended_portfolio'):
                                save_name = st.text_input(
                                    "Portfolio Name:", 
                                    value=f"Portfolio_{datetime.now().strftime('%Y%m%d')}",
                                    key="supabase_save_name"
                                )
                                
                                if st.button("☁️ Save to Supabase", key="supabase_save_btn", use_container_width=True):
                                    if save_portfolio_to_supabase(save_name, st.session_state['recommended_portfolio'], current_user):
                                        st.success(f"✅ Portfolio '{save_name}' saved!")
                                        st.rerun()
                            else:
                                st.info("Create a portfolio first to save it")
                        
                        with load_col:
                            st.markdown("##### 📂 Load Saved Portfolio")
                            
                            saved_portfolios = list_portfolios_from_supabase(current_user)
                            
                            if saved_portfolios:
                                portfolio_options = [p['portfolio_name'] for p in saved_portfolios]
                                selected_portfolio = st.selectbox(
                                    "Select Portfolio:",
                                    portfolio_options,
                                    key="supabase_load_select"
                                )
                                
                                btn_col1, btn_col2 = st.columns(2)
                                
                                with btn_col1:
                                    if st.button("📥 Load", key="supabase_load_btn", use_container_width=True):
                                        loaded = load_portfolio_from_supabase(selected_portfolio, current_user)
                                        if loaded:
                                            st.session_state['recommended_portfolio'] = loaded
                                            st.session_state['recommended_portfolio_saved'] = True
                                            st.session_state['temp_portfolio'] = loaded.copy()
                                            # Clear cached tables to force recomputation
                                            if 'inv_fund_tables_computed' in st.session_state:
                                                del st.session_state['inv_fund_tables_computed']
                                            if 'inv_fund_returns_cache' in st.session_state:
                                                del st.session_state['inv_fund_returns_cache']
                                            st.success(f"✅ Portfolio '{selected_portfolio}' loaded!")
                                            st.rerun()
                                
                                with btn_col2:
                                    if st.button("🗑️ Delete", key="supabase_delete_btn", use_container_width=True):
                                        if delete_portfolio_from_supabase(selected_portfolio, current_user):
                                            st.success(f"✅ Portfolio '{selected_portfolio}' deleted!")
                                            st.rerun()
                                
                                # Show saved portfolios info
                                st.markdown("##### 📋 Saved Portfolios")
                                for p in saved_portfolios:
                                    updated = p.get('updated_at', '')[:10] if p.get('updated_at') else 'N/A'
                                    st.text(f"• {p['portfolio_name']} (Updated: {updated})")
                            else:
                                st.info("No saved portfolios found")
                
                st.markdown("---")
                
                # ═══════════════════════════════════════════════════════════════════════════
                # PORTFOLIO ANALYSIS - REPLICATE DETAILED ANALYSIS EXACTLY
                # ═══════════════════════════════════════════════════════════════════════════
                
                if st.session_state['recommended_portfolio_saved'] and st.session_state['recommended_portfolio']:
                    portfolio = st.session_state['recommended_portfolio']
                    st.markdown("### 📊 Saved Portfolio")
                    
                    # Get fund categories and subcategories
                    rec_fund_categories = {}
                    rec_fund_subcategories = {}
                    for fund_name in portfolio.keys():
                        fund_row = fund_metrics[fund_metrics['FUNDO DE INVESTIMENTO'] == fund_name]
                        if len(fund_row) > 0:
                            rec_fund_categories[fund_name] = fund_row['CATEGORIA BTG'].iloc[0] if 'CATEGORIA BTG' in fund_row.columns else 'Unknown'
                            rec_fund_subcategories[fund_name] = fund_row['SUBCATEGORIA BTG'].iloc[0] if 'SUBCATEGORIA BTG' in fund_row.columns else 'Unknown'
                        else:
                            rec_fund_categories[fund_name] = 'Unknown'
                            rec_fund_subcategories[fund_name] = 'Unknown'
                    
                    # Normalize weights
                    total_alloc = sum(portfolio.values())
                    weights_series = pd.Series({k: v / total_alloc for k, v in portfolio.items()})
                    
                    # Pie chart view selector buttons
                    view_col1, view_col2, view_col3 = st.columns(3)
                    
                    with view_col1:
                        if st.button("📊 By Investment Fund", use_container_width=True, key="rec_view_fund"):
                            st.session_state['rec_portfolio_view'] = 'fund'
                    
                    with view_col2:
                        if st.button("📁 By Category", use_container_width=True, key="rec_view_cat"):
                            st.session_state['rec_portfolio_view'] = 'category'
                    
                    with view_col3:
                        if st.button("📂 By Subcategory", use_container_width=True, key="rec_view_subcat"):
                            st.session_state['rec_portfolio_view'] = 'subcategory'
                    
                    # Default view
                    if 'rec_portfolio_view' not in st.session_state:
                        st.session_state['rec_portfolio_view'] = 'fund'
                    
                    # Display pie chart
                    fig_pie = create_portfolio_pie_chart(
                        weights_series,
                        st.session_state['rec_portfolio_view'],
                        rec_fund_categories,
                        rec_fund_subcategories
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Summary tables
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        st.markdown("#### Category Breakdown")
                        category_weights = {}
                        for fund, weight in weights_series.items():
                            cat = rec_fund_categories.get(fund, 'Unknown')
                            category_weights[cat] = category_weights.get(cat, 0) + weight
                        
                        cat_df = pd.DataFrame({
                            'Category': list(category_weights.keys()),
                            'Weight %': [w*100 for w in category_weights.values()]
                        }).sort_values('Weight %', ascending=False)
                        
                        st.dataframe(
                            cat_df.style.format({'Weight %': '{:.2f}%'}),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with summary_col2:
                        st.markdown("#### Subcategory Breakdown")
                        subcat_weights = {}
                        for fund, weight in weights_series.items():
                            subcat = rec_fund_subcategories.get(fund, 'Unknown')
                            subcat_weights[subcat] = subcat_weights.get(subcat, 0) + weight
                        
                        subcat_df = pd.DataFrame({
                            'Subcategory': list(subcat_weights.keys()),
                            'Weight %': [w*100 for w in subcat_weights.values()]
                        }).sort_values('Weight %', ascending=False)
                        
                        st.dataframe(
                            subcat_df.style.format({'Weight %': '{:.2f}%'}),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # Load fund returns
                    fund_returns_dict = {fn: get_fund_returns_by_name(fn, fund_metrics, fund_details) for fn in portfolio.keys()}
                    fund_returns_dict = {k: v for k, v in fund_returns_dict.items() if v is not None}
                    
                    if fund_returns_dict:
                        portfolio_returns = calculate_portfolio_returns(fund_returns_dict, portfolio)
                        
                        if portfolio_returns is not None and len(portfolio_returns) > 0:
                            st.markdown("---")
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # RETURNS ANALYSIS SECTION
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("### 📈 Returns Analysis")
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                period_map = {'3M': 3, '6M': 6, '12M': 12, '24M': 24, '36M': 36, 'All': None}
                                period_list = list(period_map.keys())
                                selected_period = st.selectbox("Select Period:", period_list, index=5, key="rec_port_period")
                            
                            with col2:
                                benchmark_cols = benchmarks.columns.tolist()
                                default_benches = []
                                if 'CDI' in benchmark_cols:
                                    default_benches.append('CDI')
                                if 'IBOVESPA' in benchmark_cols:
                                    default_benches.append('IBOVESPA')
                                selected_benchmarks = st.multiselect("Select Benchmarks:", benchmark_cols, default=default_benches, key="rec_port_bench")
                            
                            # Filter returns by period
                            if period_map[selected_period] is not None:
                                cutoff = portfolio_returns.index[-1] - pd.DateOffset(months=period_map[selected_period])
                                port_ret_filtered = portfolio_returns[portfolio_returns.index >= cutoff]
                            else:
                                port_ret_filtered = portfolio_returns
                            
                            # Benchmark dict
                            benchmark_dict = {b: benchmarks[b] for b in selected_benchmarks if b in benchmarks.columns}
                            
                            # Cumulative returns chart
                            fig_returns = create_returns_chart(port_ret_filtered, benchmark_dict, "Recommended Portfolio", selected_period)
                            st.plotly_chart(fig_returns, use_container_width=True)
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # RETURNS COMPARISON TABLE
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("#### Returns Comparison")
                            
                            periods = ['MTD', '3M', '6M', 'YTD', '12M', '24M', '36M', 'TOTAL']
                            returns_data = []
                            
                            # Portfolio row
                            port_row = {'Asset': 'Recommended Portfolio'}
                            for period in periods:
                                if period == 'MTD':
                                    pr = portfolio_returns[portfolio_returns.index >= portfolio_returns.index[-1].replace(day=1)]
                                elif period == 'YTD':
                                    pr = portfolio_returns[portfolio_returns.index >= portfolio_returns.index[-1].replace(month=1, day=1)]
                                elif period == 'TOTAL':
                                    pr = portfolio_returns
                                else:
                                    months = int(period.replace('M', ''))
                                    pr = portfolio_returns.tail(months * 21)
                                port_row[f'Return {period}'] = (1 + pr).prod() - 1 if len(pr) > 0 else np.nan
                            returns_data.append(port_row)
                            
                            # Benchmark rows
                            for bench_name in selected_benchmarks:
                                if bench_name in benchmarks.columns:
                                    bench_row = {'Asset': bench_name}
                                    bench_data = benchmarks[bench_name].reindex(portfolio_returns.index, method='ffill').fillna(0)
                                    for period in periods:
                                        if period == 'MTD':
                                            br = bench_data[bench_data.index >= bench_data.index[-1].replace(day=1)]
                                        elif period == 'YTD':
                                            br = bench_data[bench_data.index >= bench_data.index[-1].replace(month=1, day=1)]
                                        elif period == 'TOTAL':
                                            br = bench_data
                                        else:
                                            months = int(period.replace('M', ''))
                                            br = bench_data.tail(months * 21)
                                        bench_row[f'Return {period}'] = (1 + br).prod() - 1 if len(br) > 0 else np.nan
                                    returns_data.append(bench_row)
                            
                            returns_df = pd.DataFrame(returns_data)
                            for col in returns_df.columns:
                                if col != 'Asset':
                                    returns_df[col] = returns_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                            st.dataframe(returns_df, use_container_width=True, hide_index=True)
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # MONTHLY STATISTICS
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("#### Monthly Statistics")
                            
                            monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                            total_months = len(monthly_returns)
                            
                            months_above_zero = (monthly_returns > 0).sum()
                            months_below_zero = (monthly_returns <= 0).sum()
                            pct_above_zero = months_above_zero / total_months if total_months > 0 else 0
                            pct_below_zero = months_below_zero / total_months if total_months > 0 else 0
                            
                            # Calculate vs benchmark
                            if selected_benchmarks and benchmarks is not None:
                                bench_name = selected_benchmarks[0]
                                if bench_name in benchmarks.columns:
                                    aligned_bench = benchmarks[bench_name].reindex(portfolio_returns.index, method='ffill').fillna(0)
                                    monthly_bench = aligned_bench.resample('M').apply(lambda x: (1 + x).prod() - 1)
                                    months_above_bench = (monthly_returns > monthly_bench).sum()
                                    months_below_bench = (monthly_returns <= monthly_bench).sum()
                                    pct_above_bench = months_above_bench / total_months if total_months > 0 else 0
                                    pct_below_bench = months_below_bench / total_months if total_months > 0 else 0
                                else:
                                    months_above_bench = months_below_bench = pct_above_bench = pct_below_bench = np.nan
                            else:
                                months_above_bench = months_below_bench = pct_above_bench = pct_below_bench = np.nan
                            
                            best_month = monthly_returns.max()
                            worst_month = monthly_returns.min()
                            
                            monthly_data = {
                                'Months Above Zero': f"{months_above_zero} ({pct_above_zero*100:.1f}%)",
                                'Months Below Zero': f"{months_below_zero} ({pct_below_zero*100:.1f}%)",
                            }
                            if pd.notna(months_above_bench):
                                monthly_data['Months Above Benchmark'] = f"{int(months_above_bench)} ({pct_above_bench*100:.1f}%)"
                                monthly_data['Months Below Benchmark'] = f"{int(months_below_bench)} ({pct_below_bench*100:.1f}%)"
                            monthly_data['Best Month'] = f"{best_month*100:.2f}%"
                            monthly_data['Worst Month'] = f"{worst_month*100:.2f}%"
                            
                            st.dataframe(pd.DataFrame([monthly_data]), use_container_width=True, hide_index=True)
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # MONTHLY RETURNS CALENDAR
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("---")
                            st.markdown("#### Monthly Returns Calendar")
                            
                            cal_col1, cal_col2 = st.columns([1, 1])
                            with cal_col1:
                                default_bench_idx = benchmark_cols.index('CDI') if 'CDI' in benchmark_cols else 0
                                selected_calendar_benchmark = st.selectbox("Select Benchmark for Comparison:", benchmark_cols, index=default_bench_idx, key="rec_calendar_bench")
                            with cal_col2:
                                comparison_method = st.selectbox("Comparison Method:", ['Relative Performance', 'Percentage Points', 'Benchmark Performance'], index=0, key="rec_comp_method")
                            
                            if selected_calendar_benchmark in benchmarks.columns:
                                monthly_table = create_monthly_returns_table(portfolio_returns, benchmarks[selected_calendar_benchmark], comparison_method)
                                styled_html = style_monthly_returns_table(monthly_table, comparison_method)
                                st.markdown(styled_html, unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # RISK-ADJUSTED PERFORMANCE SECTION
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("### ⚖️ Risk-Adjusted Performance")
                            
                            # Frequency selection
                            st.markdown("#### Data Frequency Selection")
                            frequency_choice = st.radio(
                                "Select frequency for Omega, Rachev, VaR and CVaR analysis:",
                                options=['Daily', 'Weekly', 'Monthly'],
                                horizontal=True,
                                help="Choose whether to analyze daily, weekly, or monthly returns data",
                                key="rec_freq_choice"
                            )
                            
                            freq_label = frequency_choice.lower()
                            
                            if frequency_choice == 'Daily':
                                returns_data = portfolio_returns
                            elif frequency_choice == 'Weekly':
                                returns_data = portfolio_returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
                            else:
                                returns_data = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                            
                            st.markdown("---")
                            
                            # === OMEGA SECTION ===
                            st.markdown("#### Omega Ratio")
                            
                            omega_chart_col, omega_gauge_col = st.columns([2, 1])
                            
                            with omega_chart_col:
                                fig_omega = create_omega_cdf_chart(returns_data, threshold=0, frequency=freq_label)
                                st.plotly_chart(fig_omega, use_container_width=True)
                            
                            with omega_gauge_col:
                                omega_val = PortfolioMetrics.omega_ratio(returns_data)
                                if pd.notna(omega_val) and not np.isinf(omega_val):
                                    fig_omega_gauge = create_omega_gauge(omega_val, frequency=frequency_choice)
                                    st.plotly_chart(fig_omega_gauge, use_container_width=True)
                                else:
                                    st.metric(f"Omega Ratio ({frequency_choice})", "N/A")
                            
                            st.markdown("---")
                            
                            # === RACHEV / VAR / CVAR SECTION ===
                            st.markdown("#### Rachev Ratio & Tail Risk")
                            
                            rachev_chart_col, rachev_metrics_col = st.columns([2, 1])
                            
                            with rachev_chart_col:
                                var_val = PortfolioMetrics.var(returns_data, confidence=0.95)
                                cvar_val = PortfolioMetrics.cvar(returns_data, confidence=0.95)
                                fig_rachev = create_combined_rachev_var_chart(returns_data, var_val, cvar_val, frequency=freq_label)
                                st.plotly_chart(fig_rachev, use_container_width=True)
                            
                            with rachev_metrics_col:
                                rachev_val = PortfolioMetrics.rachev_ratio(returns_data)
                                if pd.notna(rachev_val) and not np.isinf(rachev_val):
                                    fig_rachev_gauge = create_rachev_gauge(rachev_val, frequency=frequency_choice)
                                    st.plotly_chart(fig_rachev_gauge, use_container_width=True)
                                else:
                                    st.metric(f"Rachev Ratio ({frequency_choice})", "N/A")
                            
                            st.markdown("---")
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # SHARPE RATIO ANALYSIS
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("### 📊 Sharpe Ratio Analysis")
                            
                            sharpe_chart_col, sharpe_metrics_col = st.columns([3, 1])
                            
                            with sharpe_chart_col:
                                fig_sharpe = create_rolling_sharpe_chart(portfolio_returns, window_months=12)
                                st.plotly_chart(fig_sharpe, use_container_width=True)
                            
                            with sharpe_metrics_col:
                                sharpe_12m = PortfolioMetrics.sharpe_ratio(portfolio_returns.tail(252))
                                st.metric("Sharpe 12M", f"{sharpe_12m:.2f}" if pd.notna(sharpe_12m) else "N/A")
                                
                                sharpe_24m = PortfolioMetrics.sharpe_ratio(portfolio_returns.tail(504))
                                st.metric("Sharpe 24M", f"{sharpe_24m:.2f}" if pd.notna(sharpe_24m) else "N/A")
                                
                                sharpe_36m = PortfolioMetrics.sharpe_ratio(portfolio_returns.tail(756))
                                st.metric("Sharpe 36M", f"{sharpe_36m:.2f}" if pd.notna(sharpe_36m) else "N/A")
                                
                                sharpe_total = PortfolioMetrics.sharpe_ratio(portfolio_returns)
                                st.metric("Sharpe Total", f"{sharpe_total:.2f}" if pd.notna(sharpe_total) else "N/A")
                            
                            st.markdown("---")
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # RISK METRICS SECTION
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("### 🎯 Risk Metrics Dashboard")
                            
                            # Volatility section
                            st.markdown("#### Volatility Analysis")
                            
                            vol_chart_col, vol_metrics_col = st.columns([3, 1])
                            
                            with vol_chart_col:
                                fig_vol = create_rolling_vol_chart(portfolio_returns, window_months=12)
                                st.plotly_chart(fig_vol, use_container_width=True)
                            
                            with vol_metrics_col:
                                vol_12m = PortfolioMetrics.annualized_volatility(portfolio_returns.tail(252))
                                st.metric("Vol 12M", f"{vol_12m*100:.2f}%" if pd.notna(vol_12m) else "N/A")
                                
                                vol_24m = PortfolioMetrics.annualized_volatility(portfolio_returns.tail(504))
                                st.metric("Vol 24M", f"{vol_24m*100:.2f}%" if pd.notna(vol_24m) else "N/A")
                                
                                vol_36m = PortfolioMetrics.annualized_volatility(portfolio_returns.tail(756))
                                st.metric("Vol 36M", f"{vol_36m*100:.2f}%" if pd.notna(vol_36m) else "N/A")
                                
                                vol_total = PortfolioMetrics.annualized_volatility(portfolio_returns)
                                st.metric("Vol Total", f"{vol_total*100:.2f}%" if pd.notna(vol_total) else "N/A")
                            
                            st.markdown("---")
                            
                            # Drawdown section
                            st.markdown("#### Drawdown Analysis")
                            
                            dd_chart_col, dd_metrics_col = st.columns([3, 1])
                            
                            with dd_chart_col:
                                fig_underwater, max_dd_info = create_underwater_plot(portfolio_returns)
                                st.plotly_chart(fig_underwater, use_container_width=True)
                            
                            with dd_metrics_col:
                                mdd = PortfolioMetrics.max_drawdown(portfolio_returns)
                                st.metric("Max Drawdown", f"{mdd*100:.2f}%" if pd.notna(mdd) else "N/A")
                                
                                if max_dd_info and 'duration' in max_dd_info:
                                    st.metric("MDD Duration", f"{max_dd_info['duration']} days")
                                
                                if max_dd_info and 'cdar_95' in max_dd_info:
                                    st.metric("CDaR (95%)", f"{max_dd_info['cdar_95']:.2f}%", help="Conditional Drawdown at Risk: Average of worst 5% drawdowns")
                            
                            st.markdown("---")
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # BENCHMARK EXPOSURES SECTION
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("### 🌍 Benchmark Exposures")
                            
                            available_benchmarks = [b for b in ['CDI', 'USDBRL', 'GOLD', 'IBOVESPA', 'SP500', 'BITCOIN'] if b in benchmarks.columns]
                            default_exposure = []
                            if 'CDI' in available_benchmarks:
                                default_exposure.append('CDI')
                            if 'IBOVESPA' in available_benchmarks:
                                default_exposure.append('IBOVESPA')
                            
                            selected_exposure_benches = st.multiselect(
                                "Select Benchmarks for Exposure Analysis:",
                                options=available_benchmarks,
                                default=default_exposure,
                                key="rec_exposure_select"
                            )
                            
                            if selected_exposure_benches:
                                exposure_data = []
                                
                                for bench in selected_exposure_benches:
                                    with st.spinner(f'Calculating exposure for {bench}...'):
                                        copula_results = estimate_rolling_copula_for_chart(
                                            portfolio_returns,
                                            benchmarks[bench],
                                            window=250
                                        )
                                        
                                        if copula_results is not None:
                                            last_kendall = copula_results['kendall_tau'].iloc[-1]
                                            last_tail_lower = copula_results['tail_lower'].iloc[-1]
                                            last_tail_upper = copula_results['tail_upper'].iloc[-1]
                                            last_asymmetry = copula_results['asymmetry_index'].iloc[-1]
                                            
                                            avg_kendall = copula_results['kendall_tau'].mean()
                                            avg_tail_lower = copula_results['tail_lower'].mean()
                                            avg_tail_upper = copula_results['tail_upper'].mean()
                                            avg_asymmetry = copula_results['asymmetry_index'].mean()
                                            
                                            exposure_data.append({
                                                'Benchmark': f'{bench} - Last Window',
                                                'Kendall Tau': last_kendall,
                                                'Tail Lower': last_tail_lower,
                                                'Tail Upper': last_tail_upper,
                                                'Asymmetry': last_asymmetry
                                            })
                                            exposure_data.append({
                                                'Benchmark': f'{bench} - Average',
                                                'Kendall Tau': avg_kendall,
                                                'Tail Lower': avg_tail_lower,
                                                'Tail Upper': avg_tail_upper,
                                                'Asymmetry': avg_asymmetry
                                            })
                                        else:
                                            bench_returns = benchmarks[bench].reindex(portfolio_returns.index, method='ffill').fillna(0)
                                            u = to_empirical_cdf(portfolio_returns)
                                            v = to_empirical_cdf(bench_returns)
                                            tau = stats.kendalltau(u.values, v.values)[0]
                                            theta_lower, _ = estimate_gumbel_270_parameter(u.values, v.values)
                                            lambda_lower, _ = gumbel_270_tail_dependence(theta_lower)
                                            theta_upper, _ = estimate_gumbel_180_parameter(u.values, v.values)
                                            _, lambda_upper = gumbel_180_tail_dependence(theta_upper)
                                            asymmetry = (lambda_lower - lambda_upper) / (lambda_lower + lambda_upper) if (lambda_lower + lambda_upper) > 0 else 0
                                            
                                            exposure_data.append({
                                                'Benchmark': f'{bench} - Full Period',
                                                'Kendall Tau': tau,
                                                'Tail Lower': lambda_lower,
                                                'Tail Upper': lambda_upper,
                                                'Asymmetry': asymmetry
                                            })
                                
                                exposure_df = pd.DataFrame(exposure_data)
                                st.dataframe(
                                    exposure_df.style.format({col: "{:.4f}" for col in exposure_df.columns if col != 'Benchmark'})
                                    .background_gradient(cmap='RdYlGn', subset=[col for col in exposure_df.columns if col != 'Benchmark'], vmin=-1, vmax=1),
                                    use_container_width=True, hide_index=True
                                )
                            else:
                                st.info("Select at least one benchmark to view exposures")
                            
                            st.markdown("---")
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # EXPOSURE TIME SERIES ANALYSIS
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("### 📈 Portfolio Exposure Time Series Analysis")
                            st.info("💡 Select a benchmark below to visualize the evolution of portfolio exposure metrics over time")
                            
                            available_ts_benchmarks = ['None'] + [b for b in ['CDI', 'USDBRL', 'GOLD', 'IBOVESPA', 'SP500', 'BITCOIN'] if b in benchmarks.columns]
                            selected_ts_benchmark = st.selectbox("Select Benchmark for Time Series:", available_ts_benchmarks, index=0, key="rec_ts_bench")
                            
                            if selected_ts_benchmark != 'None' and selected_ts_benchmark in benchmarks.columns:
                                with st.spinner(f'Calculating exposure time series for {selected_ts_benchmark}...'):
                                    copula_results = estimate_rolling_copula_for_chart(
                                        portfolio_returns,
                                        benchmarks[selected_ts_benchmark],
                                        window=250
                                    )
                                    
                                    if copula_results is not None:
                                        current_kendall = copula_results['kendall_tau'].iloc[-1]
                                        avg_kendall = copula_results['kendall_tau'].mean()
                                        current_tail_lower = copula_results['tail_lower'].iloc[-1]
                                        avg_tail_lower = copula_results['tail_lower'].mean()
                                        current_tail_upper = copula_results['tail_upper'].iloc[-1]
                                        avg_tail_upper = copula_results['tail_upper'].mean()
                                        current_asymmetry = copula_results['asymmetry_index'].iloc[-1]
                                        avg_asymmetry = copula_results['asymmetry_index'].mean()
                                        
                                        st.markdown(f"##### Portfolio Exposure Evolution - {selected_ts_benchmark}")
                                        
                                        row1_col1, row1_col2 = st.columns(2)
                                        with row1_col1:
                                            fig_kendall = create_exposure_time_series_chart(copula_results, 'kendall_tau', current_kendall, avg_kendall, selected_ts_benchmark)
                                            st.plotly_chart(fig_kendall, use_container_width=True)
                                        with row1_col2:
                                            fig_asymmetry = create_exposure_time_series_chart(copula_results, 'asymmetry_index', current_asymmetry, avg_asymmetry, selected_ts_benchmark)
                                            st.plotly_chart(fig_asymmetry, use_container_width=True)
                                        
                                        row2_col1, row2_col2 = st.columns(2)
                                        with row2_col1:
                                            fig_tail_lower = create_exposure_time_series_chart(copula_results, 'tail_lower', current_tail_lower, avg_tail_lower, selected_ts_benchmark)
                                            st.plotly_chart(fig_tail_lower, use_container_width=True)
                                        with row2_col2:
                                            fig_tail_upper = create_exposure_time_series_chart(copula_results, 'tail_upper', current_tail_upper, avg_tail_upper, selected_ts_benchmark)
                                            st.plotly_chart(fig_tail_upper, use_container_width=True)
                                        
                                        st.markdown("##### Summary Statistics")
                                        mc1, mc2, mc3, mc4 = st.columns(4)
                                        with mc1:
                                            st.metric("Kendall Tau", f"{current_kendall:.4f}", delta=f"Avg: {avg_kendall:.4f}")
                                        with mc2:
                                            st.metric("Tail Lower", f"{current_tail_lower:.4f}", delta=f"Avg: {avg_tail_lower:.4f}")
                                        with mc3:
                                            st.metric("Tail Upper", f"{current_tail_upper:.4f}", delta=f"Avg: {avg_tail_upper:.4f}")
                                        with mc4:
                                            st.metric("Asymmetry", f"{current_asymmetry:.4f}", delta=f"Avg: {avg_asymmetry:.4f}")
                                    else:
                                        st.warning("Insufficient data for time series analysis (need at least 275 observations)")
                        else:
                            st.error("❌ Could not calculate portfolio returns")
                    else:
                        st.error("❌ No fund returns data available")
                else:
                    st.info("👆 Create and save a portfolio above to see the analysis")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # SECONDARY TAB 2: INVESTMENT FUND ANALYSIS
            # ═══════════════════════════════════════════════════════════════════════════
            
            with rec_tab2:
                st.markdown("### 📈 Investment Fund Analysis")
                st.markdown("---")
                
                if not st.session_state['recommended_portfolio_saved'] or not st.session_state['recommended_portfolio']:
                    st.info("👆 Create portfolio in 'Portfolio Analysis' tab first")
                else:
                    portfolio = st.session_state['recommended_portfolio']
                    
                    # Cache fund returns in session state to avoid recomputation
                    portfolio_key = tuple(sorted(portfolio.keys()))
                    if 'inv_fund_returns_cache' not in st.session_state or st.session_state.get('inv_fund_portfolio_key') != portfolio_key:
                        with st.spinner("Loading fund returns..."):
                            fund_returns_dict = {fn: get_fund_returns_by_name(fn, fund_metrics, fund_details) for fn in portfolio.keys()}
                            fund_returns_dict = {k: v for k, v in fund_returns_dict.items() if v is not None}
                            st.session_state['inv_fund_returns_cache'] = fund_returns_dict
                            st.session_state['inv_fund_portfolio_key'] = portfolio_key
                    else:
                        fund_returns_dict = st.session_state['inv_fund_returns_cache']
                    
                    if fund_returns_dict and 'CDI' in benchmarks.columns:
                        cdi_returns = benchmarks['CDI']
                        
                        # Pre-compute tables once and store in session state
                        if 'inv_fund_tables_computed' not in st.session_state or st.session_state.get('inv_fund_portfolio_key_tables') != portfolio_key:
                            with st.spinner("Computing return tables..."):
                                # Compute cumulative returns table
                                cdf, ccdi = create_cumulative_returns_comparison_table(fund_returns_dict, cdi_returns)
                                if cdf is not None:
                                    # Add Last Day column
                                    for idx, row in cdf.iterrows():
                                        fund_name = row['Fund']
                                        if fund_name == 'CDI':
                                            last_day_ret = cdi_returns.iloc[-1] if len(cdi_returns) > 0 else np.nan
                                        else:
                                            if fund_name in fund_returns_dict:
                                                last_day_ret = fund_returns_dict[fund_name].iloc[-1] if len(fund_returns_dict[fund_name]) > 0 else np.nan
                                            else:
                                                last_day_ret = np.nan
                                        cdf.loc[idx, 'Last Day'] = last_day_ret
                                    cols = ['Fund', 'Last Day'] + [c for c in cdf.columns if c not in ['Fund', 'Last Day']]
                                    cdf = cdf[cols]
                                    ccdi['Last Day'] = cdi_returns.iloc[-1] if len(cdi_returns) > 0 else 0
                                
                                # Compute monthly returns table
                                mdf, mcdi = create_monthly_returns_comparison_table(fund_returns_dict, cdi_returns, 12)
                                
                                # Store in session state
                                st.session_state['inv_fund_cdf'] = cdf
                                st.session_state['inv_fund_ccdi'] = ccdi
                                st.session_state['inv_fund_mdf'] = mdf
                                st.session_state['inv_fund_mcdi'] = mcdi
                                st.session_state['inv_fund_tables_computed'] = True
                                st.session_state['inv_fund_portfolio_key_tables'] = portfolio_key
                        
                        # Retrieve cached tables
                        cdf = st.session_state.get('inv_fund_cdf')
                        ccdi = st.session_state.get('inv_fund_ccdi')
                        mdf = st.session_state.get('inv_fund_mdf')
                        mcdi = st.session_state.get('inv_fund_mcdi')
                        
                        # Display mode selection
                        display_mode = st.radio(
                            "Display Mode:",
                            ["Absolute Returns", "Relative Performance (vs CDI)"],
                            horizontal=True,
                            key="inv_fund_display_mode"
                        )
                        
                        st.markdown("---")
                        
                        # Initialize sorting state
                        if 'inv_sort_col_cum' not in st.session_state:
                            st.session_state['inv_sort_col_cum'] = None
                        if 'inv_sort_asc_cum' not in st.session_state:
                            st.session_state['inv_sort_asc_cum'] = False
                        if 'inv_sort_col_monthly' not in st.session_state:
                            st.session_state['inv_sort_col_monthly'] = None
                        if 'inv_sort_asc_monthly' not in st.session_state:
                            st.session_state['inv_sort_asc_monthly'] = False
                        
                        if display_mode == "Absolute Returns":
                            st.markdown("""<p style='color: #888; font-size: 12px;'>
                            Color Legend: <span style='color: #FFF;'>■ White = Return > CDI</span> | 
                            <span style='color: #48F;'>■ Blue = 0 ≤ Return ≤ CDI</span> | 
                            <span style='color: #F44;'>■ Red = Return < 0</span></p>""", unsafe_allow_html=True)
                            
                            # Cumulative Returns Table with Last Day column (use cached)
                            st.markdown("#### Cumulative Returns")
                            st.markdown("<p style='color: #888; font-size: 11px;'>Click column buttons to sort ↑↓</p>", unsafe_allow_html=True)
                            
                            if cdf is not None:
                                # Sort buttons for cumulative table
                                sort_cols_cum = st.columns(len(cdf.columns[1:]))
                                for i, col in enumerate(cdf.columns[1:]):
                                    with sort_cols_cum[i]:
                                        sort_indicator = ""
                                        if st.session_state['inv_sort_col_cum'] == col:
                                            sort_indicator = " ↑" if st.session_state['inv_sort_asc_cum'] else " ↓"
                                        if st.button(f"{col}{sort_indicator}", key=f"sort_cum_{col}", use_container_width=True):
                                            if st.session_state['inv_sort_col_cum'] == col:
                                                st.session_state['inv_sort_asc_cum'] = not st.session_state['inv_sort_asc_cum']
                                            else:
                                                st.session_state['inv_sort_col_cum'] = col
                                                st.session_state['inv_sort_asc_cum'] = False
                                            st.rerun()
                                
                                st.markdown(style_sortable_returns_table(
                                    cdf, ccdi, 
                                    st.session_state['inv_sort_col_cum'], 
                                    st.session_state['inv_sort_asc_cum']
                                ), unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # Monthly Returns Table (use cached)
                            st.markdown("#### Monthly Returns (Last 12 Months)")
                            st.markdown("<p style='color: #888; font-size: 11px;'>Click column buttons to sort ↑↓</p>", unsafe_allow_html=True)
                            
                            if mdf is not None:
                                # Sort buttons for monthly table
                                sort_cols_monthly = st.columns(min(len(mdf.columns[1:]), 12))
                                for i, col in enumerate(mdf.columns[1:]):
                                    if i < 12:  # Limit columns shown
                                        with sort_cols_monthly[i]:
                                            sort_indicator = ""
                                            if st.session_state['inv_sort_col_monthly'] == col:
                                                sort_indicator = " ↑" if st.session_state['inv_sort_asc_monthly'] else " ↓"
                                            if st.button(f"{col}{sort_indicator}", key=f"sort_monthly_{col}", use_container_width=True):
                                                if st.session_state['inv_sort_col_monthly'] == col:
                                                    st.session_state['inv_sort_asc_monthly'] = not st.session_state['inv_sort_asc_monthly']
                                                else:
                                                    st.session_state['inv_sort_col_monthly'] = col
                                                    st.session_state['inv_sort_asc_monthly'] = False
                                                st.rerun()
                                
                                st.markdown(style_sortable_returns_table(
                                    mdf, mcdi,
                                    st.session_state['inv_sort_col_monthly'],
                                    st.session_state['inv_sort_asc_monthly']
                                ), unsafe_allow_html=True)
                        
                        else:  # Relative Performance
                            st.markdown("""<p style='color: #888; font-size: 12px;'>
                            Color Legend: <span style='color: #FFF;'>■ White = Outperformed CDI (>100%)</span> | 
                            <span style='color: #48F;'>■ Blue = 0-100% of CDI</span> | 
                            <span style='color: #F44;'>■ Red = Negative relative performance</span></p>""", unsafe_allow_html=True)
                            
                            # Cumulative Returns - Relative Performance (use cached)
                            st.markdown("#### Cumulative Returns (Relative to CDI)")
                            st.markdown("<p style='color: #888; font-size: 11px;'>Click column buttons to sort ↑↓</p>", unsafe_allow_html=True)
                            
                            if cdf is not None:
                                # Convert to relative performance
                                rel_cdf = cdf.copy()
                                for col in rel_cdf.columns:
                                    if col != 'Fund':
                                        cdi_val = ccdi.get(col, 0)
                                        for idx in rel_cdf.index:
                                            fund_val = rel_cdf.loc[idx, col]
                                            if rel_cdf.loc[idx, 'Fund'] != 'CDI' and pd.notna(fund_val) and cdi_val != 0:
                                                rel_cdf.loc[idx, col] = fund_val / cdi_val if cdi_val > 0 else (1 + fund_val) / (1 + cdi_val) if cdi_val < 0 else np.nan
                                            elif rel_cdf.loc[idx, 'Fund'] == 'CDI':
                                                rel_cdf.loc[idx, col] = 1.0
                                
                                # Sort buttons for cumulative table
                                sort_cols_cum = st.columns(len(rel_cdf.columns[1:]))
                                for i, col in enumerate(rel_cdf.columns[1:]):
                                    with sort_cols_cum[i]:
                                        sort_indicator = ""
                                        if st.session_state['inv_sort_col_cum'] == col:
                                            sort_indicator = " ↑" if st.session_state['inv_sort_asc_cum'] else " ↓"
                                        if st.button(f"{col}{sort_indicator}", key=f"sort_rel_cum_{col}", use_container_width=True):
                                            if st.session_state['inv_sort_col_cum'] == col:
                                                st.session_state['inv_sort_asc_cum'] = not st.session_state['inv_sort_asc_cum']
                                            else:
                                                st.session_state['inv_sort_col_cum'] = col
                                                st.session_state['inv_sort_asc_cum'] = False
                                            st.rerun()
                                
                                st.markdown(style_sortable_relative_table(
                                    rel_cdf,
                                    st.session_state['inv_sort_col_cum'],
                                    st.session_state['inv_sort_asc_cum']
                                ), unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # Monthly Returns - Relative Performance (use cached)
                            st.markdown("#### Monthly Returns (Relative to CDI - Last 12 Months)")
                            st.markdown("<p style='color: #888; font-size: 11px;'>Click column buttons to sort ↑↓</p>", unsafe_allow_html=True)
                            
                            if mdf is not None:
                                # Convert to relative performance
                                rel_mdf = mdf.copy()
                                for col in rel_mdf.columns:
                                    if col != 'Fund':
                                        cdi_val = mcdi.get(col, 0)
                                        for idx in rel_mdf.index:
                                            fund_val = rel_mdf.loc[idx, col]
                                            if rel_mdf.loc[idx, 'Fund'] != 'CDI' and pd.notna(fund_val) and cdi_val != 0:
                                                rel_mdf.loc[idx, col] = fund_val / cdi_val if cdi_val > 0 else (1 + fund_val) / (1 + cdi_val) if cdi_val < 0 else np.nan
                                            elif rel_mdf.loc[idx, 'Fund'] == 'CDI':
                                                rel_mdf.loc[idx, col] = 1.0
                                
                                # Sort buttons for monthly table
                                sort_cols_monthly = st.columns(min(len(rel_mdf.columns[1:]), 12))
                                for i, col in enumerate(rel_mdf.columns[1:]):
                                    if i < 12:
                                        with sort_cols_monthly[i]:
                                            sort_indicator = ""
                                            if st.session_state['inv_sort_col_monthly'] == col:
                                                sort_indicator = " ↑" if st.session_state['inv_sort_asc_monthly'] else " ↓"
                                            if st.button(f"{col}{sort_indicator}", key=f"sort_rel_monthly_{col}", use_container_width=True):
                                                if st.session_state['inv_sort_col_monthly'] == col:
                                                    st.session_state['inv_sort_asc_monthly'] = not st.session_state['inv_sort_asc_monthly']
                                                else:
                                                    st.session_state['inv_sort_col_monthly'] = col
                                                    st.session_state['inv_sort_asc_monthly'] = False
                                                st.rerun()
                                
                                st.markdown(style_sortable_relative_table(
                                    rel_mdf,
                                    st.session_state['inv_sort_col_monthly'],
                                    st.session_state['inv_sort_asc_monthly']
                                ), unsafe_allow_html=True)
                    else:
                        st.error("❌ CDI benchmark data not available or no fund returns")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # SECONDARY TAB 3: BOOK ANALYSIS
            # ═══════════════════════════════════════════════════════════════════════════
            
            with rec_tab3:
                st.markdown("### 📚 Book Analysis")
                st.markdown("#### Portfolio Contribution Analysis by Category")
                st.markdown("---")
                
                if not st.session_state['recommended_portfolio_saved'] or not st.session_state['recommended_portfolio']:
                    st.info("👆 Create portfolio in 'Portfolio Analysis' tab first")
                else:
                    portfolio = st.session_state['recommended_portfolio']
                    
                    # Use cached fund returns from Investment Fund Analysis
                    portfolio_key = tuple(sorted(portfolio.keys()))
                    if 'inv_fund_returns_cache' in st.session_state and st.session_state.get('inv_fund_portfolio_key') == portfolio_key:
                        fund_returns_dict = st.session_state['inv_fund_returns_cache']
                    else:
                        with st.spinner("Loading fund returns..."):
                            fund_returns_dict = {fn: get_fund_returns_by_name(fn, fund_metrics, fund_details) for fn in portfolio.keys()}
                            fund_returns_dict = {k: v for k, v in fund_returns_dict.items() if v is not None}
                            st.session_state['inv_fund_returns_cache'] = fund_returns_dict
                            st.session_state['inv_fund_portfolio_key'] = portfolio_key
                    
                    if fund_returns_dict and 'CDI' in benchmarks.columns:
                        cdi_returns = benchmarks['CDI']
                        
                        # Normalize allocations
                        total_alloc = sum(portfolio.values())
                        weights = {k: v / total_alloc for k, v in portfolio.items()}
                        
                        # Get fund categories
                        fund_categories = {}
                        for fund_name in portfolio.keys():
                            fund_row = fund_metrics[fund_metrics['FUNDO DE INVESTIMENTO'] == fund_name]
                            if len(fund_row) > 0 and 'CATEGORIA BTG' in fund_row.columns:
                                fund_categories[fund_name] = fund_row['CATEGORIA BTG'].iloc[0]
                            else:
                                fund_categories[fund_name] = 'Unknown'
                        
                        # Group funds by category
                        categories = {}
                        for fund_name, category in fund_categories.items():
                            if category not in categories:
                                categories[category] = []
                            categories[category].append(fund_name)
                        
                        # Calculate category weights (within category and global)
                        category_global_weights = {}
                        category_internal_weights = {}
                        for category, funds in categories.items():
                            cat_total = sum(weights.get(f, 0) for f in funds)
                            category_global_weights[category] = cat_total
                            category_internal_weights[category] = {f: weights.get(f, 0) / cat_total if cat_total > 0 else 0 for f in funds}
                        
                        # Get max date and months for time series
                        max_date = None
                        for returns in fund_returns_dict.values():
                            if len(returns) > 0:
                                if max_date is None or returns.index.max() > max_date:
                                    max_date = returns.index.max()
                        
                        if max_date is not None:
                            current_month_end = max_date + pd.offsets.MonthEnd(0)
                            start_date = (current_month_end - pd.DateOffset(months=11)).replace(day=1)
                            months = pd.date_range(start=start_date, end=current_month_end, freq='M')
                            month_labels = [m.strftime('%b-%y') for m in months]
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # CHART 1: CATEGORY PERFORMANCE (WEIGHTED AVERAGE WITHIN CATEGORY)
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("### 📊 Category Performance")
                            st.markdown("<p style='color: #888; font-size: 12px;'>Weighted average return of funds within each category (normalized to category allocation)</p>", unsafe_allow_html=True)
                            
                            chart1_mode = st.radio(
                                "View Mode:",
                                ["Cumulative Returns", "Monthly Returns"],
                                horizontal=True,
                                key="book_chart1_mode"
                            )
                            
                            # Calculate category returns (weighted average within category)
                            category_monthly_returns = {cat: [] for cat in categories.keys()}
                            cdi_monthly = []
                            
                            for month_end in months:
                                month_start = month_end.replace(day=1)
                                
                                # CDI return for this month
                                cr = cdi_returns[(cdi_returns.index >= month_start) & (cdi_returns.index <= month_end)]
                                cdi_ret = (1 + cr).prod() - 1 if len(cr) > 0 else 0
                                cdi_monthly.append(cdi_ret)
                                
                                # Category returns
                                for category, funds in categories.items():
                                    cat_return = 0
                                    for fund_name in funds:
                                        if fund_name in fund_returns_dict:
                                            fund_ret = fund_returns_dict[fund_name]
                                            mr = fund_ret[(fund_ret.index >= month_start) & (fund_ret.index <= month_end)]
                                            ret = (1 + mr).prod() - 1 if len(mr) > 0 else 0
                                            internal_weight = category_internal_weights[category].get(fund_name, 0)
                                            cat_return += ret * internal_weight
                                    category_monthly_returns[category].append(cat_return)
                            
                            # Create chart
                            fig_cat = go.Figure()
                            colors = px.colors.qualitative.Bold
                            
                            if chart1_mode == "Cumulative Returns":
                                # Calculate cumulative returns
                                for i, (category, monthly_rets) in enumerate(category_monthly_returns.items()):
                                    cum_ret = [(1 + r) for r in monthly_rets]
                                    cum_ret = [np.prod(cum_ret[:j+1]) - 1 for j in range(len(cum_ret))]
                                    fig_cat.add_trace(go.Scatter(
                                        x=month_labels,
                                        y=[r * 100 for r in cum_ret],
                                        name=category,
                                        line=dict(color=colors[i % len(colors)], width=2),
                                        hovertemplate='%{y:.2f}%<extra></extra>'
                                    ))
                                
                                # CDI cumulative
                                cdi_cum = [(1 + r) for r in cdi_monthly]
                                cdi_cum = [np.prod(cdi_cum[:j+1]) - 1 for j in range(len(cdi_cum))]
                                fig_cat.add_trace(go.Scatter(
                                    x=month_labels,
                                    y=[r * 100 for r in cdi_cum],
                                    name='CDI',
                                    line=dict(color='#00CED1', width=2, dash='dash'),
                                    hovertemplate='%{y:.2f}%<extra></extra>'
                                ))
                                title = "Category Performance - Cumulative Returns (Last 12 Months)"
                            else:
                                # Monthly returns
                                for i, (category, monthly_rets) in enumerate(category_monthly_returns.items()):
                                    fig_cat.add_trace(go.Bar(
                                        x=month_labels,
                                        y=[r * 100 for r in monthly_rets],
                                        name=category,
                                        marker_color=colors[i % len(colors)],
                                        hovertemplate='%{y:.2f}%<extra></extra>'
                                    ))
                                
                                fig_cat.add_trace(go.Scatter(
                                    x=month_labels,
                                    y=[r * 100 for r in cdi_monthly],
                                    name='CDI',
                                    line=dict(color='#00CED1', width=2),
                                    hovertemplate='%{y:.2f}%<extra></extra>'
                                ))
                                title = "Category Performance - Monthly Returns (Last 12 Months)"
                                fig_cat.update_layout(barmode='group')
                            
                            fig_cat.update_layout(
                                title=title,
                                xaxis_title='Month',
                                yaxis_title='Return (%)',
                                template=PLOTLY_TEMPLATE,
                                height=450,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)
                            
                            st.markdown("---")
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # CHART 2: STACKED CONTRIBUTION (GLOBAL PORTFOLIO WEIGHTS)
                            # ═══════════════════════════════════════════════════════════════════
                            
                            st.markdown("### 📈 Portfolio Contribution by Category")
                            st.markdown("<p style='color: #888; font-size: 12px;'>Sum of weighted contributions from each category to total portfolio return</p>", unsafe_allow_html=True)
                            
                            chart2_mode = st.radio(
                                "View Mode:",
                                ["Cumulative Returns", "Monthly Returns"],
                                horizontal=True,
                                key="book_chart2_mode"
                            )
                            
                            # Calculate category contributions (global weights)
                            category_contributions = {cat: [] for cat in categories.keys()}
                            portfolio_monthly = []
                            
                            for month_end in months:
                                month_start = month_end.replace(day=1)
                                port_ret = 0
                                
                                for category, funds in categories.items():
                                    cat_contrib = 0
                                    for fund_name in funds:
                                        if fund_name in fund_returns_dict:
                                            fund_ret = fund_returns_dict[fund_name]
                                            mr = fund_ret[(fund_ret.index >= month_start) & (fund_ret.index <= month_end)]
                                            ret = (1 + mr).prod() - 1 if len(mr) > 0 else 0
                                            global_weight = weights.get(fund_name, 0)
                                            cat_contrib += ret * global_weight
                                    category_contributions[category].append(cat_contrib)
                                    port_ret += cat_contrib
                                
                                portfolio_monthly.append(port_ret)
                            
                            # Create stacked chart
                            fig_stack = go.Figure()
                            
                            if chart2_mode == "Cumulative Returns":
                                # Calculate cumulative contributions
                                cum_contributions = {}
                                for category, contribs in category_contributions.items():
                                    cum = []
                                    running = 0
                                    for c in contribs:
                                        running += c
                                        cum.append(running)
                                    cum_contributions[category] = cum
                                
                                for i, (category, cum_contribs) in enumerate(cum_contributions.items()):
                                    fig_stack.add_trace(go.Scatter(
                                        x=month_labels,
                                        y=[c * 100 for c in cum_contribs],
                                        name=category,
                                        stackgroup='one',
                                        fillcolor=colors[i % len(colors)],
                                        line=dict(width=0.5, color=colors[i % len(colors)]),
                                        hovertemplate='%{y:.2f}%<extra></extra>'
                                    ))
                                
                                # CDI cumulative
                                cdi_cum = []
                                running = 0
                                for r in cdi_monthly:
                                    running += r
                                    cdi_cum.append(running)
                                
                                fig_stack.add_trace(go.Scatter(
                                    x=month_labels,
                                    y=[r * 100 for r in cdi_cum],
                                    name='CDI',
                                    line=dict(color='#00CED1', width=2, dash='dash'),
                                    hovertemplate='%{y:.2f}%<extra></extra>'
                                ))
                                title = "Portfolio Contribution by Category - Cumulative (Last 12 Months)"
                            else:
                                # Monthly bar chart with Portfolio Total line
                                for i, (category, contribs) in enumerate(category_contributions.items()):
                                    fig_stack.add_trace(go.Bar(
                                        x=month_labels,
                                        y=[c * 100 for c in contribs],
                                        name=category,
                                        marker_color=colors[i % len(colors)],
                                        hovertemplate='%{y:.2f}%<extra></extra>'
                                    ))
                                
                                # Add Portfolio Total line
                                fig_stack.add_trace(go.Scatter(
                                    x=month_labels,
                                    y=[r * 100 for r in portfolio_monthly],
                                    name='Portfolio Total',
                                    line=dict(color='#D4AF37', width=3),
                                    hovertemplate='%{y:.2f}%<extra></extra>'
                                ))
                                
                                # Add CDI line
                                fig_stack.add_trace(go.Scatter(
                                    x=month_labels,
                                    y=[r * 100 for r in cdi_monthly],
                                    name='CDI',
                                    line=dict(color='#00CED1', width=2, dash='dash'),
                                    hovertemplate='%{y:.2f}%<extra></extra>'
                                ))
                                title = "Portfolio Contribution by Category - Monthly (Last 12 Months)"
                                fig_stack.update_layout(barmode='group')
                            
                            fig_stack.update_layout(
                                title=title,
                                xaxis_title='Month',
                                yaxis_title='Return Contribution (%)',
                                template=PLOTLY_TEMPLATE,
                                height=450,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_stack, use_container_width=True)
                        
                        st.markdown("---")
                        
                        st.markdown("""<p style='color: #888; font-size: 12px;'>
                        Shows weighted contribution of each fund to portfolio returns (Fund Return × Allocation Weight).
                        Category totals show sum of contributions from all funds in that category.</p>""", unsafe_allow_html=True)
                        
                        # ═══════════════════════════════════════════════════════════════════
                        # CUMULATIVE RETURNS CONTRIBUTION TABLE
                        # ═══════════════════════════════════════════════════════════════════
                        
                        st.markdown("#### Cumulative Returns Contribution")
                        
                        # Calculate cumulative returns for each period
                        periods = ['Last Day', 'MTD', '3M', '6M', '12M']
                        book_data = []
                        category_totals = {cat: {p: 0.0 for p in periods} for cat in categories.keys()}
                        portfolio_total = {p: 0.0 for p in periods}
                        
                        for category in sorted(categories.keys()):
                            for fund_name in categories[category]:
                                if fund_name not in fund_returns_dict:
                                    continue
                                
                                fund_ret = fund_returns_dict[fund_name]
                                weight = weights.get(fund_name, 0)
                                
                                row = {'Fund': fund_name, 'Category': category, 'Allocation': weight}
                                
                                for period in periods:
                                    if period == 'Last Day':
                                        ret = fund_ret.iloc[-1] if len(fund_ret) > 0 else 0
                                    elif period == 'MTD':
                                        pr = fund_ret[fund_ret.index >= fund_ret.index[-1].replace(day=1)]
                                        ret = (1 + pr).prod() - 1 if len(pr) > 0 else 0
                                    else:
                                        months_num = int(period.replace('M', ''))
                                        pr = fund_ret.tail(months_num * 21)
                                        ret = (1 + pr).prod() - 1 if len(pr) > 0 else 0
                                    
                                    contribution = ret * weight
                                    row[period] = contribution
                                    category_totals[category][period] += contribution
                                    portfolio_total[period] += contribution
                                
                                book_data.append(row)
                            
                            # Add category total row
                            cat_row = {'Fund': f'📁 {category} TOTAL', 'Category': category, 'Allocation': sum(weights.get(f, 0) for f in categories[category])}
                            for period in periods:
                                cat_row[period] = category_totals[category][period]
                            book_data.append(cat_row)
                        
                        # Add portfolio total row
                        port_row = {'Fund': '📊 PORTFOLIO TOTAL', 'Category': '', 'Allocation': 1.0}
                        for period in periods:
                            port_row[period] = portfolio_total[period]
                        book_data.append(port_row)
                        
                        # Add CDI row
                        cdi_row = {'Fund': '📈 CDI', 'Category': '', 'Allocation': ''}
                        for period in periods:
                            if period == 'Last Day':
                                cdi_row[period] = cdi_returns.iloc[-1] if len(cdi_returns) > 0 else 0
                            elif period == 'MTD':
                                cr = cdi_returns[cdi_returns.index >= cdi_returns.index[-1].replace(day=1)]
                                cdi_row[period] = (1 + cr).prod() - 1 if len(cr) > 0 else 0
                            else:
                                months_num = int(period.replace('M', ''))
                                cr = cdi_returns.tail(months_num * 21)
                                cdi_row[period] = (1 + cr).prod() - 1 if len(cr) > 0 else 0
                        book_data.append(cdi_row)
                        
                        book_df = pd.DataFrame(book_data)
                        st.markdown(style_book_analysis_table(book_df, periods), unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # ═══════════════════════════════════════════════════════════════════
                        # MONTHLY RETURNS CONTRIBUTION TABLE
                        # ═══════════════════════════════════════════════════════════════════
                        
                        st.markdown("#### Monthly Returns Contribution (Last 12 Months)")
                        
                        if max_date is not None:
                            monthly_book_data = []
                            monthly_cat_totals = {cat: {m: 0.0 for m in month_labels} for cat in categories.keys()}
                            monthly_port_total = {m: 0.0 for m in month_labels}
                            
                            for category in sorted(categories.keys()):
                                for fund_name in categories[category]:
                                    if fund_name not in fund_returns_dict:
                                        continue
                                    
                                    fund_ret = fund_returns_dict[fund_name]
                                    weight = weights.get(fund_name, 0)
                                    
                                    row = {'Fund': fund_name, 'Category': category}
                                    
                                    for month_end, month_label in zip(months, month_labels):
                                        month_start = month_end.replace(day=1)
                                        mr = fund_ret[(fund_ret.index >= month_start) & (fund_ret.index <= month_end)]
                                        ret = (1 + mr).prod() - 1 if len(mr) > 0 else 0
                                        contribution = ret * weight
                                        row[month_label] = contribution
                                        monthly_cat_totals[category][month_label] += contribution
                                        monthly_port_total[month_label] += contribution
                                    
                                    monthly_book_data.append(row)
                                
                                # Category total
                                cat_row = {'Fund': f'📁 {category} TOTAL', 'Category': category}
                                for month_label in month_labels:
                                    cat_row[month_label] = monthly_cat_totals[category][month_label]
                                monthly_book_data.append(cat_row)
                            
                            # Portfolio total
                            port_row = {'Fund': '📊 PORTFOLIO TOTAL', 'Category': ''}
                            for month_label in month_labels:
                                port_row[month_label] = monthly_port_total[month_label]
                            monthly_book_data.append(port_row)
                            
                            # CDI row
                            cdi_row = {'Fund': '📈 CDI', 'Category': ''}
                            for month_end, month_label in zip(months, month_labels):
                                month_start = month_end.replace(day=1)
                                cr = cdi_returns[(cdi_returns.index >= month_start) & (cdi_returns.index <= month_end)]
                                cdi_row[month_label] = (1 + cr).prod() - 1 if len(cr) > 0 else 0
                            monthly_book_data.append(cdi_row)
                            
                            monthly_book_df = pd.DataFrame(monthly_book_data)
                            st.markdown(style_book_analysis_table(monthly_book_df, month_labels), unsafe_allow_html=True)
                    else:
                        st.error("❌ CDI benchmark data not available or no fund returns")


if __name__ == "__main__":
    main()


