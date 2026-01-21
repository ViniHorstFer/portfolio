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
}

# Original pkl names (inside zip files)
PKL_ORIGINAL_NAMES = {
    'funds_info': 'funds_info.pkl',
}

# Session state cache keys
CACHE_KEYS = {
    'fund_metrics': '_gh_fund_metrics',
    'funds_info': '_gh_funds_info',
    'benchmarks': '_gh_benchmarks',
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
    Decompress zip file and extract the pkl content.
    
    Args:
        zip_content: Zip file bytes
    
    Returns:
        Extracted pkl file bytes
    """
    zip_buffer = io.BytesIO(zip_content)
    
    with zipfile.ZipFile(zip_buffer, 'r') as zf:
        # Get the first file in the zip (should be the pkl)
        file_list = zf.namelist()
        if not file_list:
            raise ValueError("Zip file is empty")
        
        # Extract the pkl file
        pkl_filename = file_list[0]
        return zf.read(pkl_filename)


def is_zip_file(content: bytes) -> bool:
    """Check if content is a zip file by checking magic bytes."""
    return content[:4] == b'PK\x03\x04'


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_asset_download_url(release: Dict, asset_name: str) -> Optional[str]:
    """Get the download URL for a specific asset."""
    for asset in release.get('assets', []):
        if asset['name'] == asset_name:
            return asset['browser_download_url']
    return None


def get_asset_info(release: Dict, asset_name: str) -> Optional[Dict]:
    """Get information about a specific asset."""
    for asset in release.get('assets', []):
        if asset['name'] == asset_name:
            return {
                'id': asset['id'],
                'name': asset['name'],
                'size': asset['size'],
                'download_url': asset['browser_download_url'],
                'created_at': asset['created_at'],
                'updated_at': asset['updated_at'],
            }
    return None


def download_asset(download_url: str) -> Optional[bytes]:
    """Download an asset from GitHub Releases."""
    try:
        # For public repos, browser_download_url works without auth
        # For private repos, we need to use the API endpoint with auth
        response = requests.get(download_url, headers=get_headers(), timeout=120, allow_redirects=True)
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Error downloading asset: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error downloading asset: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_excel_from_github(download_url: str) -> Optional[pd.DataFrame]:
    """Load Excel file from GitHub with caching."""
    content = download_asset(download_url)
    if content is None:
        return None
    
    try:
        df = pd.read_excel(io.BytesIO(content))
        return df
    except Exception as e:
        st.error(f"Failed to parse Excel file: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_pickle_from_github(download_url: str, is_zipped: bool = True) -> Optional[Any]:
    """
    Load Pickle file from GitHub with caching.
    
    Args:
        download_url: URL to download the file
        is_zipped: If True, expects a zip file containing the pkl
    
    Returns:
        Loaded pickle object
    """
    content = download_asset(download_url)
    if content is None:
        return None
    
    try:
        # Check if content is zipped
        if is_zipped or is_zip_file(content):
            # Decompress the zip file first
            pkl_content = decompress_zip_to_pkl(content)
            data = joblib.load(io.BytesIO(pkl_content))
        else:
            # Load directly as pkl
            data = joblib.load(io.BytesIO(content))
        return data
    except zipfile.BadZipFile:
        # Not a zip file, try loading directly
        try:
            data = joblib.load(io.BytesIO(content))
            return data
        except Exception as e:
            st.error(f"Failed to parse Pickle file: {e}")
            return None
    except Exception as e:
        st.error(f"Failed to parse Pickle file: {e}")
        return None


def load_fund_metrics_from_github() -> Optional[pd.DataFrame]:
    """Load fund metrics from GitHub Releases."""
    # Check session cache first
    if CACHE_KEYS['fund_metrics'] in st.session_state:
        return st.session_state[CACHE_KEYS['fund_metrics']]
    
    release = get_release_by_tag()
    if release is None:
        return None
    
    # Try xlsx first (default)
    file_name = RELEASE_FILE_NAMES['fund_metrics']  # fund_metrics.xlsx
    download_url = get_asset_download_url(release, file_name)
    is_zipped = False
    
    # Try .pkl.zip if xlsx not found
    if download_url is None:
        file_name = 'fund_metrics.pkl.zip'
        download_url = get_asset_download_url(release, file_name)
        is_zipped = True
    
    # Try .pkl if .pkl.zip not found
    if download_url is None:
        file_name = 'fund_metrics.pkl'
        download_url = get_asset_download_url(release, file_name)
        is_zipped = False
    
    if download_url is None:
        st.warning(f"File not found in release: fund_metrics.xlsx, .pkl.zip, or .pkl")
        return None
    
    with st.spinner("Loading fund metrics from GitHub..."):
        if file_name.endswith('.pkl.zip'):
            df = load_pickle_from_github(download_url, is_zipped=True)
        elif file_name.endswith('.pkl'):
            df = load_pickle_from_github(download_url, is_zipped=False)
        else:
            df = load_excel_from_github(download_url)
    
    if df is not None:
        st.session_state[CACHE_KEYS['fund_metrics']] = df
    
    return df


def load_fund_details_from_github() -> Optional[Any]:
    """Load fund details (pkl) from GitHub Releases."""
    # Check session cache first
    if CACHE_KEYS['funds_info'] in st.session_state:
        return st.session_state[CACHE_KEYS['funds_info']]
    
    release = get_release_by_tag()
    if release is None:
        return None
    
    # Try .pkl.zip first (compressed format for GitHub compatibility)
    file_name = RELEASE_FILE_NAMES['funds_info']  # funds_info.pkl.zip
    download_url = get_asset_download_url(release, file_name)
    is_zipped = True
    
    # Fallback to .pkl if .pkl.zip not found
    if download_url is None:
        file_name = 'funds_info.pkl'
        download_url = get_asset_download_url(release, file_name)
        is_zipped = False
    
    if download_url is None:
        st.warning(f"File not found in release: funds_info.pkl.zip or funds_info.pkl")
        return None
    
    with st.spinner("Loading fund details from GitHub..."):
        data = load_pickle_from_github(download_url, is_zipped=is_zipped)
    
    if data is not None:
        st.session_state[CACHE_KEYS['funds_info']] = data
    
    return data


def load_benchmarks_from_github() -> Optional[pd.DataFrame]:
    """Load benchmarks from GitHub Releases."""
    # Check session cache first
    if CACHE_KEYS['benchmarks'] in st.session_state:
        return st.session_state[CACHE_KEYS['benchmarks']]
    
    release = get_release_by_tag()
    if release is None:
        return None
    
    # Try xlsx first (default)
    file_name = RELEASE_FILE_NAMES['benchmarks']  # benchmarks_data.xlsx
    download_url = get_asset_download_url(release, file_name)
    
    # Try .pkl.zip if xlsx not found
    if download_url is None:
        file_name = 'benchmarks_data.pkl.zip'
        download_url = get_asset_download_url(release, file_name)
    
    # Try .pkl if .pkl.zip not found
    if download_url is None:
        file_name = 'benchmarks_data.pkl'
        download_url = get_asset_download_url(release, file_name)
    
    if download_url is None:
        st.warning(f"File not found in release: benchmarks_data.xlsx, .pkl.zip, or .pkl")
        return None
    
    with st.spinner("Loading benchmarks from GitHub..."):
        if file_name.endswith('.pkl.zip'):
            df = load_pickle_from_github(download_url, is_zipped=True)
        elif file_name.endswith('.pkl'):
            df = load_pickle_from_github(download_url, is_zipped=False)
        else:
            df = load_excel_from_github(download_url)
    
    if df is not None:
        # Set date index if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
        
        st.session_state[CACHE_KEYS['benchmarks']] = df
    
    return df


def load_all_from_github() -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[pd.DataFrame]]:
    """Load all data files from GitHub Releases."""
    fund_metrics = load_fund_metrics_from_github()
    fund_details = load_fund_details_from_github()
    benchmarks = load_benchmarks_from_github()
    
    return fund_metrics, fund_details, benchmarks


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def delete_asset(asset_id: int) -> bool:
    """Delete an existing asset."""
    config = get_github_config()
    
    url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/releases/assets/{asset_id}"
    
    try:
        response = requests.delete(url, headers=get_headers(), timeout=30)
        return response.status_code == 204
    except Exception as e:
        st.error(f"Error deleting asset: {e}")
        return False


def upload_asset(release: Dict, file_content: bytes, file_name: str, content_type: str) -> bool:
    """Upload an asset to a release."""
    config = get_github_config()
    
    # Check if asset already exists and delete it
    for asset in release.get('assets', []):
        if asset['name'] == file_name:
            st.info(f"Replacing existing file: {file_name}")
            if not delete_asset(asset['id']):
                st.error("Failed to delete existing asset")
                return False
            break
    
    # Get upload URL from release
    upload_url = release.get('upload_url', '').replace('{?name,label}', '')
    if not upload_url:
        st.error("No upload URL found in release")
        return False
    
    upload_url = f"{upload_url}?name={file_name}"
    
    headers = get_headers()
    headers['Content-Type'] = content_type
    
    try:
        response = requests.post(
            upload_url,
            headers=headers,
            data=file_content,
            timeout=300  # 5 minutes for large files
        )
        
        if response.status_code == 201:
            return True
        else:
            st.error(f"Error uploading asset: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        st.error(f"Error uploading asset: {e}")
        return False


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
            file_name = 'fund_metrics.pkl.zip'
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
    """
    Upload fund details (pkl) file to GitHub Release.
    
    Note: pkl files are automatically compressed to .zip format
    for GitHub compatibility.
    """
    try:
        content = uploaded_file.read()
        original_name = uploaded_file.name if hasattr(uploaded_file, 'name') else 'funds_info.pkl'
        
        # Compress pkl to zip for GitHub compatibility
        st.info("Compressing pkl file for GitHub compatibility...")
        zip_content = compress_pkl_to_zip(content, original_name)
        
        file_name = 'funds_info.pkl.zip'
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
            file_name = 'benchmarks_data.pkl.zip'
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
# STREAMLIT UI COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

def render_github_data_panel(show_upload: bool = True):
    """Render the GitHub Releases data management panel for the sidebar.
    
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
