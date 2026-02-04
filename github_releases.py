"""
═══════════════════════════════════════════════════════════════════════════════
GITHUB RELEASES STORAGE - Data file management for Fund Analytics Platform
═══════════════════════════════════════════════════════════════════════════════

This module handles uploading and downloading data files using GitHub Releases.

FEATURES:
- Upload xlsx/pkl files from the app as release assets
- Automatically compresses pkl files to zip for GitHub compatibility
- Download and decompress files on startup
- List available files in the release
- No file size limit issues (2GB per file on free tier)
- No credit card required

SETUP:
1. Create a GitHub Personal Access Token (PAT):
   - Go to: github.com > Settings > Developer settings > Personal access tokens > Tokens (classic)
   - Generate new token with 'repo' scope
2. Create a release in your repository (tag: "data" or similar)
3. Add token to Streamlit secrets

FREE TIER LIMITS:
- 2 GB per file
- Unlimited storage for public repos
- 1 GB storage for private repos (free tier)

NOTE: pkl files are automatically compressed to .zip for upload and decompressed on download.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import joblib
import zipfile
from datetime import datetime
from typing import Optional, Any, Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Expected file names in the release (pkl files stored as .zip)
RELEASE_FILE_NAMES = {
    'fund_metrics': 'fund_metrics.xlsx',
    'funds_info': 'funds_info.zip',      # pkl compressed as zip
    'benchmarks': 'benchmarks_data.xlsx',
    # Assets (ETF) files
    'assets_metrics': 'assets_metrics.xlsx',
    'assets_prices': 'assets_prices.zip',  # pkl compressed as zip
}

# Original pkl names (inside zip files)
PKL_ORIGINAL_NAMES = {
    'funds_info': 'funds_info.pkl',
    'assets_prices': 'assets_prices.pkl',  # pkl inside zip
}

# Session state cache keys
CACHE_KEYS = {
    'fund_metrics': '_gh_fund_metrics',
    'funds_info': '_gh_funds_info',
    'benchmarks': '_gh_benchmarks',
    'assets_metrics': '_gh_assets_metrics',
    'assets_prices': '_gh_assets_prices',
    'release_info': '_gh_release_info',
    'last_sync': '_gh_last_sync',
}

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_github_config() -> Dict[str, str]:
    """Get GitHub configuration from Streamlit secrets."""
    config = {
        'token': '',
        'owner': '',
        'repo': '',
        'release_tag': 'data',  # Default release tag
    }
    
    # Try Streamlit secrets
    try:
        if hasattr(st, 'secrets') and 'github' in st.secrets:
            gh_secrets = st.secrets['github']
            config['token'] = gh_secrets.get('token', '')
            config['owner'] = gh_secrets.get('owner', '')
            config['repo'] = gh_secrets.get('repo', '')
            config['release_tag'] = gh_secrets.get('release_tag', 'data')
    except Exception:
        pass
    
    return config


def is_github_configured() -> bool:
    """Check if GitHub credentials are properly configured."""
    config = get_github_config()
    return all([
        config.get('token'),
        config.get('owner'),
        config.get('repo'),
    ])


def get_headers(with_auth: bool = True) -> Dict[str, str]:
    """Get headers for GitHub API requests."""
    headers = {
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    
    if with_auth:
        config = get_github_config()
        if config.get('token'):
            headers['Authorization'] = f"Bearer {config['token']}"
    
    return headers


# ═══════════════════════════════════════════════════════════════════════════════
# RELEASE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_release_by_tag(tag: str = None) -> Optional[Dict]:
    """Get release information by tag."""
    config = get_github_config()
    tag = tag or config.get('release_tag', 'data')
    
    url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/releases/tags/{tag}"
    
    try:
        response = requests.get(url, headers=get_headers(), timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            st.error(f"Error fetching release: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error connecting to GitHub: {e}")
        return None


def get_latest_release() -> Optional[Dict]:
    """Get the latest release."""
    config = get_github_config()
    
    url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/releases/latest"
    
    try:
        response = requests.get(url, headers=get_headers(), timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            st.error(f"Error fetching latest release: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error connecting to GitHub: {e}")
        return None


def create_release(tag: str, name: str = None, body: str = None) -> Optional[Dict]:
    """Create a new release."""
    config = get_github_config()
    
    url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/releases"
    
    data = {
        'tag_name': tag,
        'name': name or f"Data Release {tag}",
        'body': body or f"Data files uploaded at {datetime.now().isoformat()}",
        'draft': False,
        'prerelease': False,
    }
    
    try:
        response = requests.post(url, headers=get_headers(), json=data, timeout=30)
        
        if response.status_code == 201:
            return response.json()
        else:
            st.error(f"Error creating release: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error creating release: {e}")
        return None


def get_or_create_release(tag: str = None) -> Optional[Dict]:
    """Get existing release or create a new one."""
    config = get_github_config()
    tag = tag or config.get('release_tag', 'data')
    
    release = get_release_by_tag(tag)
    
    if release is None:
        st.info(f"Creating new release with tag '{tag}'...")
        release = create_release(tag)
    
    return release


# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSION FUNCTIONS (for pkl files)
# ═══════════════════════════════════════════════════════════════════════════════

def compress_pkl_to_zip(pkl_content: bytes, original_filename: str) -> bytes:
    """
    Compress pkl file content to zip format.
    
    Args:
        pkl_content: Raw pkl file bytes
        original_filename: Original filename (e.g., 'funds_info.pkl')
    
    Returns:
        Zip file bytes
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(original_filename, pkl_content)
    
    zip_buffer.seek(0)
    return zip_buffer.read()


def decompress_zip_to_pkl(zip_content: bytes) -> bytes:
    """
    Decompress zip file to get pkl content.
    
    Args:
        zip_content: Raw zip file bytes
    
    Returns:
        pkl file bytes
    """
    zip_buffer = io.BytesIO(zip_content)
    
    with zipfile.ZipFile(zip_buffer, 'r') as zf:
        # Get first file in zip (should be the pkl)
        file_list = zf.namelist()
        if not file_list:
            raise ValueError("Zip file is empty")
        
        pkl_content = zf.read(file_list[0])
    
    return pkl_content


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def find_asset_by_name(release: Dict, asset_name: str) -> Optional[Dict]:
    """Find an asset by name in a release."""
    if not release or 'assets' not in release:
        return None
    
    for asset in release['assets']:
        if asset['name'] == asset_name:
            return asset
    
    return None


def download_asset(asset: Dict) -> Optional[bytes]:
    """Download an asset from GitHub."""
    if not asset or 'url' not in asset:
        return None
    
    try:
        response = requests.get(
            asset['url'],
            headers={**get_headers(), 'Accept': 'application/octet-stream'},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Error downloading asset: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error downloading asset: {e}")
        return None


def delete_asset(asset_id: int) -> bool:
    """Delete an asset from GitHub Release."""
    config = get_github_config()
    
    url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/releases/assets/{asset_id}"
    
    try:
        response = requests.delete(url, headers=get_headers(), timeout=30)
        return response.status_code == 204
    except Exception:
        return False


def upload_asset(release: Dict, content: bytes, file_name: str, content_type: str) -> bool:
    """Upload an asset to a GitHub Release, replacing if it exists."""
    if not release or 'upload_url' not in release:
        st.error("Invalid release object")
        return False
    
    # Delete existing asset if present
    existing_asset = find_asset_by_name(release, file_name)
    if existing_asset:
        st.info(f"Deleting existing {file_name}...")
        delete_asset(existing_asset['id'])
    
    # Upload new asset
    upload_url = release['upload_url'].replace('{?name,label}', f'?name={file_name}')
    
    headers = get_headers()
    headers['Content-Type'] = content_type
    
    try:
        response = requests.post(
            upload_url,
            headers=headers,
            data=content,
            timeout=120
        )
        
        if response.status_code == 201:
            st.success(f"✅ {file_name} uploaded successfully!")
            return True
        else:
            st.error(f"Error uploading {file_name}: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        st.error(f"Error uploading {file_name}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_excel_from_github(asset_name: str, _release_id: str = None) -> Optional[pd.DataFrame]:
    """
    Load Excel file from GitHub Release.
    
    Args:
        asset_name: Name of the asset to load (e.g., 'fund_metrics.xlsx')
        _release_id: Release ID for cache invalidation (use release['id'])
    
    Returns:
        DataFrame or None
    """
    release = get_release_by_tag()
    if not release:
        return None
    
    asset = find_asset_by_name(release, asset_name)
    if not asset:
        return None
    
    content = download_asset(asset)
    if not content:
        return None
    
    try:
        df = pd.read_excel(io.BytesIO(content), index_col=0)
        return df
    except Exception as e:
        st.error(f"Error reading {asset_name}: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_pickle_from_github(asset_name: str, _release_id: str = None) -> Optional[Any]:
    """
    Load pickle file from GitHub Release (handles .zip compressed files).
    
    Args:
        asset_name: Name of the asset to load (e.g., 'funds_info.zip')
        _release_id: Release ID for cache invalidation (use release['id'])
    
    Returns:
        Unpickled object or None
    """
    release = get_release_by_tag()
    if not release:
        return None
    
    asset = find_asset_by_name(release, asset_name)
    if not asset:
        return None
    
    content = download_asset(asset)
    if not content:
        return None
    
    try:
        # If it's a zip file, decompress first
        if asset_name.endswith('.zip'):
            content = decompress_zip_to_pkl(content)
        
        # Load pickle
        obj = joblib.load(io.BytesIO(content))
        return obj
    except Exception as e:
        st.error(f"Error reading {asset_name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIFIC DATA LOADERS (INVESTMENT FUNDS)
# ═══════════════════════════════════════════════════════════════════════════════

def load_fund_metrics_from_github() -> Optional[pd.DataFrame]:
    """Load fund metrics from GitHub Release."""
    if CACHE_KEYS['fund_metrics'] in st.session_state:
        return st.session_state[CACHE_KEYS['fund_metrics']]
    
    release = get_release_by_tag()
    if not release:
        return None
    
    with st.spinner("📥 Loading fund metrics from GitHub..."):
        df = load_excel_from_github(
            RELEASE_FILE_NAMES['fund_metrics'],
            _release_id=str(release.get('id'))
        )
        
        if df is not None:
            # Validate that this is Investment Funds data
            expected_cols = ['FUNDO DE INVESTIMENTO', 'CNPJ', 'GESTOR']
            has_expected = any(col in df.columns for col in expected_cols)
            
            if not has_expected:
                st.error(f"❌ File '{RELEASE_FILE_NAMES['fund_metrics']}' doesn't appear to be Investment Funds data")
                st.warning("It might be ETF/Assets data. Please check your GitHub Release files.")
                st.info(f"Found columns: {', '.join(df.columns.tolist()[:10])}")
                return None
            
            st.session_state[CACHE_KEYS['fund_metrics']] = df
        
        return df


def load_fund_details_from_github() -> Optional[pd.DataFrame]:
    """Load fund details from GitHub Release."""
    if CACHE_KEYS['funds_info'] in st.session_state:
        return st.session_state[CACHE_KEYS['funds_info']]
    
    release = get_release_by_tag()
    if not release:
        return None
    
    with st.spinner("📥 Loading fund details from GitHub..."):
        df = load_pickle_from_github(
            RELEASE_FILE_NAMES['funds_info'],
            _release_id=str(release.get('id'))
        )
        
        if df is not None:
            st.session_state[CACHE_KEYS['funds_info']] = df
        
        return df


def load_benchmarks_from_github() -> Optional[pd.DataFrame]:
    """Load benchmarks from GitHub Release."""
    if CACHE_KEYS['benchmarks'] in st.session_state:
        return st.session_state[CACHE_KEYS['benchmarks']]
    
    release = get_release_by_tag()
    if not release:
        return None
    
    with st.spinner("📥 Loading benchmarks from GitHub..."):
        # Try xlsx first
        df = load_excel_from_github(
            RELEASE_FILE_NAMES['benchmarks'],
            _release_id=str(release.get('id'))
        )
        
        if df is not None:
            st.session_state[CACHE_KEYS['benchmarks']] = df
        
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIFIC DATA LOADERS (ASSETS/ETFS)
# ═══════════════════════════════════════════════════════════════════════════════

def load_assets_metrics_from_github() -> Optional[pd.DataFrame]:
    """Load assets/ETF metrics from GitHub Release."""
    if CACHE_KEYS['assets_metrics'] in st.session_state:
        return st.session_state[CACHE_KEYS['assets_metrics']]
    
    release = get_release_by_tag()
    if not release:
        return None
    
    with st.spinner("📥 Loading assets metrics from GitHub..."):
        df = load_excel_from_github(
            RELEASE_FILE_NAMES['assets_metrics'],
            _release_id=str(release.get('id'))
        )
        
        if df is not None:
            # Validate that this is ETF/Assets data
            expected_cols = ['Name', 'Class', 'Category']  # ETF typical columns
            has_expected = any(col in df.columns for col in expected_cols)
            
            # Also check it's NOT Investment Funds data
            wrong_cols = ['FUNDO DE INVESTIMENTO', 'CNPJ', 'GESTOR']
            has_wrong = any(col in df.columns for col in wrong_cols)
            
            if has_wrong:
                st.error(f"❌ File '{RELEASE_FILE_NAMES['assets_metrics']}' appears to be Investment Funds data, not ETF/Assets data")
                st.warning("Please check your GitHub Release files - you may have uploaded the wrong file.")
                st.info(f"Found columns: {', '.join(df.columns.tolist()[:10])}")
                return None
            
            st.session_state[CACHE_KEYS['assets_metrics']] = df
        
        return df


def load_assets_prices_from_github() -> Optional[pd.DataFrame]:
    """Load assets/ETF prices from GitHub Release (pkl compressed as zip)."""
    if CACHE_KEYS['assets_prices'] in st.session_state:
        return st.session_state[CACHE_KEYS['assets_prices']]
    
    release = get_release_by_tag()
    if not release:
        return None
    
    with st.spinner("📥 Loading assets prices from GitHub..."):
        df = load_pickle_from_github(
            RELEASE_FILE_NAMES['assets_prices'],
            _release_id=str(release.get('id'))
        )
        
        if df is not None:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            st.session_state[CACHE_KEYS['assets_prices']] = df
        
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD FUNCTIONS (INVESTMENT FUNDS)
# ═══════════════════════════════════════════════════════════════════════════════

def upload_fund_metrics(uploaded_file) -> bool:
    """Upload fund metrics file to GitHub Release."""
    try:
        content = uploaded_file.read()
        original_name = uploaded_file.name if hasattr(uploaded_file, 'name') else 'fund_metrics.xlsx'
        
        # Determine file name and content type
        if uploaded_file.name.endswith('.pkl'):
            # Compress pkl to zip for GitHub compatibility
            st.info("Compressing pkl file for GitHub compatibility...")
            content = compress_pkl_to_zip(content, original_name)
            file_name = 'fund_metrics.zip'
            content_type = 'application/zip'
        else:
            file_name = 'fund_metrics.xlsx'
            content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
        release = get_or_create_release()
        if release is None:
            return False
        
        success = upload_asset(release, content, file_name, content_type)
        
        if success:
            # Clear cache
            if CACHE_KEYS['fund_metrics'] in st.session_state:
                del st.session_state[CACHE_KEYS['fund_metrics']]
            load_excel_from_github.clear()
            load_pickle_from_github.clear()
        
        return success
        
    except Exception as e:
        st.error(f"Error uploading fund metrics: {e}")
        return False


def upload_fund_details(uploaded_file) -> bool:
    """Upload fund details file to GitHub Release."""
    try:
        content = uploaded_file.read()
        original_name = uploaded_file.name if hasattr(uploaded_file, 'name') else 'funds_info.pkl'
        
        # Compress pkl to zip for GitHub compatibility
        st.info("Compressing pkl file for GitHub compatibility...")
        zip_content = compress_pkl_to_zip(content, original_name)
        
        file_name = 'funds_info.zip'
        content_type = 'application/zip'
        
        release = get_or_create_release()
        if release is None:
            return False
        
        success = upload_asset(release, zip_content, file_name, content_type)
        
        if success:
            # Clear cache
            if CACHE_KEYS['funds_info'] in st.session_state:
                del st.session_state[CACHE_KEYS['funds_info']]
            load_pickle_from_github.clear()
        
        return success
        
    except Exception as e:
        st.error(f"Error uploading fund details: {e}")
        return False


def upload_benchmarks(uploaded_file) -> bool:
    """Upload benchmarks file to GitHub Release."""
    try:
        content = uploaded_file.read()
        original_name = uploaded_file.name if hasattr(uploaded_file, 'name') else 'benchmarks_data.xlsx'
        
        # Determine file name and content type
        if uploaded_file.name.endswith('.pkl'):
            # Compress pkl to zip for GitHub compatibility
            st.info("Compressing pkl file for GitHub compatibility...")
            content = compress_pkl_to_zip(content, original_name)
            file_name = 'benchmarks_data.zip'
            content_type = 'application/zip'
        else:
            file_name = 'benchmarks_data.xlsx'
            content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
        release = get_or_create_release()
        if release is None:
            return False
        
        success = upload_asset(release, content, file_name, content_type)
        
        if success:
            # Clear cache
            if CACHE_KEYS['benchmarks'] in st.session_state:
                del st.session_state[CACHE_KEYS['benchmarks']]
            load_excel_from_github.clear()
            load_pickle_from_github.clear()
        
        return success
        
    except Exception as e:
        st.error(f"Error uploading benchmarks: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD FUNCTIONS (ASSETS/ETFS)
# ═══════════════════════════════════════════════════════════════════════════════

def upload_assets_metrics(uploaded_file) -> bool:
    """Upload assets/ETF metrics file to GitHub Release."""
    try:
        content = uploaded_file.read()
        file_name = 'assets_metrics.xlsx'
        content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
        release = get_or_create_release()
        if release is None:
            return False
        
        success = upload_asset(release, content, file_name, content_type)
        
        if success:
            # Clear cache
            if CACHE_KEYS['assets_metrics'] in st.session_state:
                del st.session_state[CACHE_KEYS['assets_metrics']]
            load_excel_from_github.clear()
        
        return success
        
    except Exception as e:
        st.error(f"Error uploading assets metrics: {e}")
        return False


def upload_assets_prices(uploaded_file) -> bool:
    """Upload assets/ETF prices file to GitHub Release (pkl will be compressed to zip)."""
    try:
        content = uploaded_file.read()
        original_name = uploaded_file.name if hasattr(uploaded_file, 'name') else 'assets_prices.pkl'
        
        # Compress pkl to zip for GitHub compatibility
        st.info("Compressing pkl file for GitHub compatibility...")
        zip_content = compress_pkl_to_zip(content, original_name)
        
        file_name = 'assets_prices.zip'
        content_type = 'application/zip'
        
        release = get_or_create_release()
        if release is None:
            return False
        
        success = upload_asset(release, zip_content, file_name, content_type)
        
        if success:
            # Clear cache
            if CACHE_KEYS['assets_prices'] in st.session_state:
                del st.session_state[CACHE_KEYS['assets_prices']]
            load_pickle_from_github.clear()
        
        return success
        
    except Exception as e:
        st.error(f"Error uploading assets prices: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def list_release_assets() -> List[Dict[str, Any]]:
    """List all assets in the data release."""
    release = get_release_by_tag()
    if release is None:
        return []
    
    assets = []
    for asset in release.get('assets', []):
        assets.append({
            'name': asset['name'],
            'size': asset['size'],
            'download_count': asset['download_count'],
            'created_at': asset['created_at'],
            'updated_at': asset['updated_at'],
        })
    
    return assets


def clear_github_cache():
    """Clear all GitHub-related caches."""
    for key in CACHE_KEYS.values():
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear function caches
    load_excel_from_github.clear()
    load_pickle_from_github.clear()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_datetime(dt_str: str) -> str:
    """Format ISO datetime string for display."""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return dt_str


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_github_data_panel(show_upload: bool = True):
    """Render the GitHub Releases data management panel for the sidebar (Investment Funds).
    
    Args:
        show_upload: If False, hides the upload section (for read-only users)
    """
    
    st.markdown("### 📦 GitHub Releases Storage")
    
    if not is_github_configured():
        st.warning("⚠️ GitHub not configured")
        with st.expander("Setup Instructions"):
            st.markdown("""
            **1. Create Personal Access Token**
            - Go to: [GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)](https://github.com/settings/tokens)
            - Click "Generate new token (classic)"
            - Select scope: `repo` (full control)
            - Copy the token
            
            **2. Create a Release**
            - Go to your repository
            - Click "Releases" → "Create a new release"
            - Tag: `data`
            - Title: "Data Files"
            - Publish release
            
            **3. Add to Streamlit Secrets**
            ```toml
            [github]
            token = "ghp_your_token_here"
            owner = "your_username"
            repo = "your_repo_name"
            release_tag = "data"
            ```
            """)
        return None, None, None
    
    # Show connection status
    config = get_github_config()
    st.success(f"✅ Connected to {config['owner']}/{config['repo']}")
    
    # Check for release
    release = get_release_by_tag()
    if release:
        st.caption(f"Release: {release.get('tag_name', 'unknown')}")
    else:
        st.warning("No release found. Upload files to create one.")
    
    # Show current files
    with st.expander("📁 Files in Release", expanded=False):
        assets = list_release_assets()
        if assets:
            for asset in assets:
                size_str = format_file_size(asset['size'])
                date_str = format_datetime(asset['updated_at'])
                st.caption(f"**{asset['name']}** ({size_str}) - {date_str}")
        else:
            st.info("No files uploaded yet")
    
    # Upload section - only show if user has permission
    if show_upload:
        with st.expander("📤 Upload New Data", expanded=False):
            st.caption("Upload new versions of data files")
            st.caption("💡 *pkl files are auto-compressed to .zip for GitHub compatibility*")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metrics_file = st.file_uploader(
                    "Fund Metrics",
                    type=['xlsx', 'pkl'],
                    key='gh_upload_metrics',
                    help="fund_metrics.xlsx or .pkl (pkl will be compressed)"
                )
                
                if metrics_file and st.button("Upload Metrics", key='btn_gh_upload_metrics'):
                    with st.spinner("Uploading..."):
                        if upload_fund_metrics(metrics_file):
                            st.success("✅ Uploaded!")
                            st.rerun()
            
            with col2:
                details_file = st.file_uploader(
                    "Fund Details",
                    type=['pkl'],
                    key='gh_upload_details',
                    help="funds_info.pkl (will be compressed to .zip)"
                )
                
                if details_file and st.button("Upload Details", key='btn_gh_upload_details'):
                    with st.spinner("Uploading (compressing pkl → zip)..."):
                        if upload_fund_details(details_file):
                            st.success("✅ Uploaded!")
                            st.rerun()
            
            benchmarks_file = st.file_uploader(
                "Benchmarks",
                type=['xlsx', 'pkl'],
                key='gh_upload_benchmarks',
                help="benchmarks_data.xlsx or .pkl"
            )
            
            if benchmarks_file and st.button("Upload Benchmarks", key='btn_gh_upload_benchmarks'):
                with st.spinner("Uploading..."):
                    if upload_benchmarks(benchmarks_file):
                        st.success("✅ Uploaded!")
                        st.rerun()
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True, help="Clear cache and reload from GitHub"):
        clear_github_cache()
        st.rerun()
    
    # Load and return data
    fund_metrics = load_fund_metrics_from_github()
    fund_details = load_fund_details_from_github()
    benchmarks = load_benchmarks_from_github()
    
    return fund_metrics, fund_details, benchmarks


def render_github_assets_panel(show_upload: bool = True):
    """Render the GitHub Releases data management panel for Assets/ETFs.
    
    Args:
        show_upload: If False, hides the upload section (for read-only users)
    
    Returns:
        Tuple of (assets_metrics, assets_prices)
    """
    
    st.markdown("### 📦 GitHub Releases Storage")
    
    if not is_github_configured():
        st.warning("⚠️ GitHub not configured")
        with st.expander("Setup Instructions"):
            st.markdown("""
            **1. Create Personal Access Token**
            - Go to: [GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)](https://github.com/settings/tokens)
            - Click "Generate new token (classic)"
            - Select scope: `repo` (full control)
            - Copy the token
            
            **2. Create a Release**
            - Go to your repository
            - Click "Releases" → "Create a new release"
            - Tag: `data`
            - Title: "Data Files"
            - Publish release
            
            **3. Add to Streamlit Secrets**
            ```toml
            [github]
            token = "ghp_your_token_here"
            owner = "your_username"
            repo = "your_repo_name"
            release_tag = "data"
            ```
            """)
        return None, None
    
    # Show connection status
    config = get_github_config()
    st.success(f"✅ Connected to {config['owner']}/{config['repo']}")
    
    # Check for release
    release = get_release_by_tag()
    if release:
        st.caption(f"Release: {release.get('tag_name', 'unknown')}")
    else:
        st.warning("No release found. Upload files to create one.")
    
    # Show current files
    with st.expander("📁 Files in Release", expanded=False):
        assets = list_release_assets()
        if assets:
            for asset in assets:
                size_str = format_file_size(asset['size'])
                date_str = format_datetime(asset['updated_at'])
                st.caption(f"**{asset['name']}** ({size_str}) - {date_str}")
        else:
            st.info("No files uploaded yet")
    
    # Upload section - only show if user has permission
    if show_upload:
        with st.expander("📤 Upload New Data", expanded=False):
            st.caption("Upload new versions of ETF/Assets data files")
            st.caption("💡 *assets_prices.pkl is auto-compressed to .zip for GitHub compatibility*")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metrics_file = st.file_uploader(
                    "Assets Metrics",
                    type=['xlsx'],
                    key='gh_upload_assets_metrics',
                    help="assets_metrics.xlsx"
                )
                
                if metrics_file and st.button("Upload Metrics", key='btn_gh_upload_assets_metrics'):
                    with st.spinner("Uploading..."):
                        if upload_assets_metrics(metrics_file):
                            st.success("✅ Uploaded!")
                            st.rerun()
            
            with col2:
                prices_file = st.file_uploader(
                    "Assets Prices",
                    type=['pkl'],
                    key='gh_upload_assets_prices',
                    help="assets_prices.pkl (will be compressed to .zip)"
                )
                
                if prices_file and st.button("Upload Prices", key='btn_gh_upload_assets_prices'):
                    with st.spinner("Uploading (compressing pkl → zip)..."):
                        if upload_assets_prices(prices_file):
                            st.success("✅ Uploaded!")
                            st.rerun()
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True, help="Clear cache and reload from GitHub", key="refresh_assets_gh"):
        clear_github_cache()
        st.rerun()
    
    # Load and return data
    assets_metrics = load_assets_metrics_from_github()
    assets_prices = load_assets_prices_from_github()
    
    return assets_metrics, assets_prices
