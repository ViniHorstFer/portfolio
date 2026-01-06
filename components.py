"""
═══════════════════════════════════════════════════════════════════════════════
COMPONENTS MODULE - Reusable UI Components and Metrics Functions
═══════════════════════════════════════════════════════════════════════════════
This module consolidates all duplicate code from the original app.py:
- PortfolioMetrics class (unified metric calculations)
- Chart creation functions (used across all tabs)
- Table styling functions (consolidated from 4 similar functions)
- Performance dashboard rendering (unified UI for all analysis tabs)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO METRICS CLASS - Unified Calculations
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioMetrics:
    """
    Unified class for all portfolio/fund metric calculations.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def calculate_all_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate all metrics at once for efficiency."""
        if returns is None or len(returns) < 2:
            return {}
        
        returns = returns.dropna()
        if len(returns) < 2:
            return {}
        
        metrics = {}
        
        # Basic stats
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
        metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        if metrics['annualized_volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility']
        else:
            metrics['sharpe_ratio'] = np.nan
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # VaR and CVaR
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # Omega and Rachev
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        metrics['omega_ratio'] = gains / losses if losses > 0 else np.inf
        
        gains_threshold = np.percentile(returns, 95)
        losses_threshold = np.percentile(returns, 5)
        cvar_gains = returns[returns >= gains_threshold].mean()
        cvar_losses = returns[returns <= losses_threshold].mean()
        if cvar_losses < 0:
            metrics['rachev_ratio'] = cvar_gains / abs(cvar_losses)
        else:
            metrics['rachev_ratio'] = np.nan
        
        return metrics
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if returns is None or len(returns) < 2:
            return np.nan
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return np.nan
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    
    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        if returns is None or len(returns) < 2:
            return np.nan
        gains = (returns[returns > threshold] - threshold).sum()
        losses = (threshold - returns[returns <= threshold]).sum()
        return gains / losses if losses > 0 else np.inf
    
    @staticmethod
    def rachev_ratio(returns: pd.Series, alpha: float = 0.05) -> float:
        """Calculate Rachev ratio (ETL ratio)."""
        if returns is None or len(returns) < 2:
            return np.nan
        gains_threshold = np.percentile(returns, 100 * (1 - alpha))
        losses_threshold = np.percentile(returns, 100 * alpha)
        cvar_gains = returns[returns >= gains_threshold].mean()
        cvar_losses = returns[returns <= losses_threshold].mean()
        if cvar_losses < 0:
            return cvar_gains / abs(cvar_losses)
        return np.nan
    
    @staticmethod
    def annualized_volatility(returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if returns is None or len(returns) < 2:
            return np.nan
        return returns.std() * np.sqrt(252)
    
    @staticmethod
    def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)."""
        if returns is None or len(returns) < 2:
            return np.nan
        var = np.percentile(returns, 100 * (1 - confidence))
        return returns[returns <= var].mean()
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if returns is None or len(returns) < 2:
            return np.nan
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART CREATION FUNCTIONS - Unified and Cached
# ═══════════════════════════════════════════════════════════════════════════════

def downsample_for_chart(data: pd.Series, max_points: int = 500) -> pd.Series:
    """Downsample data for smoother chart rendering."""
    if len(data) <= max_points:
        return data
    step = len(data) // max_points
    return data.iloc[::step]


@st.cache_data(ttl=3600, show_spinner=False)
def create_cumulative_returns_chart(
    fund_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    fund_name: str = "Fund",
    benchmark_name: str = "CDI",
    period_label: str = ""
) -> go.Figure:
    """
    Create cumulative returns comparison chart.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    # Calculate cumulative returns
    fund_cum = ((1 + fund_returns).cumprod() - 1) * 100
    fund_cum = downsample_for_chart(fund_cum)
    
    fig = go.Figure()
    
    # Fund line
    fig.add_trace(go.Scatter(
        x=fund_cum.index,
        y=fund_cum.values,
        mode='lines',
        name=fund_name,
        line=dict(color='#D4AF37', width=2),
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    
    # Benchmark line
    if benchmark_returns is not None:
        aligned_bench = benchmark_returns.reindex(fund_returns.index).fillna(0)
        bench_cum = ((1 + aligned_bench).cumprod() - 1) * 100
        bench_cum = downsample_for_chart(bench_cum)
        
        fig.add_trace(go.Scatter(
            x=bench_cum.index,
            y=bench_cum.values,
            mode='lines',
            name=benchmark_name,
            line=dict(color='#00CED1', width=2, dash='dash'),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
    
    title = f"Cumulative Returns - {fund_name}"
    if period_label:
        title += f" ({period_label})"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_rolling_sharpe_chart(returns: pd.Series, window_months: int = 12) -> go.Figure:
    """
    Create rolling Sharpe ratio chart.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    window_days = window_months * 21
    
    if len(returns) < window_days:
        return None
    
    rolling_mean = returns.rolling(window=window_days).mean() * 252
    rolling_std = returns.rolling(window=window_days).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    rolling_sharpe = rolling_sharpe.dropna()
    rolling_sharpe = downsample_for_chart(rolling_sharpe)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        name=f'{window_months}M Rolling Sharpe',
        line=dict(color='#D4AF37', width=2),
        hovertemplate='%{y:.3f}<extra></extra>'
    ))
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.5)
    fig.add_hline(y=-1, line_dash="dot", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title=f"Rolling Sharpe Ratio ({window_months} Months)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_rolling_volatility_chart(returns: pd.Series, window_months: int = 12) -> go.Figure:
    """
    Create rolling volatility chart.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    window_days = window_months * 21
    
    if len(returns) < window_days:
        return None
    
    rolling_vol = returns.rolling(window=window_days).std() * np.sqrt(252) * 100
    rolling_vol = rolling_vol.dropna()
    rolling_vol = downsample_for_chart(rolling_vol)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values,
        mode='lines',
        name=f'{window_months}M Rolling Volatility',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.2)',
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Rolling Volatility ({window_months} Months)",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_underwater_chart(returns: pd.Series) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Create underwater (drawdown) chart.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    Returns: (figure, max_drawdown_info)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = ((cumulative - running_max) / running_max) * 100
    drawdown = downsample_for_chart(drawdown)
    
    # Find max drawdown info
    min_idx = drawdown.idxmin()
    max_dd = drawdown.min()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        line=dict(color='#FF6B6B', width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.3)',
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    
    # Mark max drawdown point
    fig.add_trace(go.Scatter(
        x=[min_idx],
        y=[max_dd],
        mode='markers+text',
        name='Max Drawdown',
        marker=dict(color='red', size=10),
        text=[f'{max_dd:.2f}%'],
        textposition='bottom center',
        textfont=dict(color='red'),
        hovertemplate='Max DD: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Underwater Plot (Drawdown from Peak)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    max_dd_info = {
        'max_drawdown': max_dd,
        'max_drawdown_date': min_idx
    }
    
    return fig, max_dd_info


@st.cache_data(ttl=3600, show_spinner=False)
def create_omega_gauge(omega_value: float, frequency: str = 'Daily') -> go.Figure:
    """
    Create Omega ratio gauge chart.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    if np.isnan(omega_value) or np.isinf(omega_value):
        omega_value = 0
    
    omega_capped = min(omega_value, 3.0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=omega_capped,
        number={'suffix': '' if omega_value <= 3 else '+', 'valueformat': '.2f'},
        title={'text': f"Omega Ratio ({frequency})", 'font': {'size': 16, 'color': '#D4AF37'}},
        gauge={
            'axis': {'range': [0, 3], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#D4AF37"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#D4AF37",
            'steps': [
                {'range': [0, 1], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [1, 1.5], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [1.5, 3], 'color': 'rgba(0, 255, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': 1
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=250,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_rachev_gauge(rachev_value: float, frequency: str = 'Daily') -> go.Figure:
    """
    Create Rachev ratio gauge chart.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    if np.isnan(rachev_value) or np.isinf(rachev_value):
        rachev_value = 0
    
    rachev_capped = min(rachev_value, 3.0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rachev_capped,
        number={'suffix': '' if rachev_value <= 3 else '+', 'valueformat': '.2f'},
        title={'text': f"Rachev Ratio ({frequency})", 'font': {'size': 16, 'color': '#D4AF37'}},
        gauge={
            'axis': {'range': [0, 3], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00CED1"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#00CED1",
            'steps': [
                {'range': [0, 0.8], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [0.8, 1.2], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [1.2, 3], 'color': 'rgba(0, 255, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': 1
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=250,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_var_cvar_chart(
    returns: pd.Series,
    var_val: float,
    cvar_val: float,
    frequency: str = 'daily'
) -> go.Figure:
    """
    Create combined Rachev ratio and VaR/CVaR distribution chart.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    from scipy.stats import gaussian_kde
    
    returns_clean = returns.dropna()
    
    # Calculate Rachev ratio
    gains_threshold = np.percentile(returns_clean, 95)
    losses_threshold = np.percentile(returns_clean, 5)
    expected_gain = returns_clean[returns_clean >= gains_threshold].mean()
    expected_loss = returns_clean[returns_clean <= losses_threshold].mean()
    rachev_ratio = expected_gain / abs(expected_loss) if expected_loss < 0 else np.inf
    
    fig = go.Figure()
    
    # KDE for distribution
    kde = gaussian_kde(returns_clean)
    x_range = np.linspace(returns_clean.min(), returns_clean.max(), 200)
    y_kde = kde(x_range)
    
    fig.add_trace(go.Scatter(
        x=x_range * 100,
        y=y_kde,
        mode='lines',
        name='Distribution',
        line=dict(color='#D4AF37', width=2),
        fill='tozeroy',
        fillcolor='rgba(212, 175, 55, 0.2)'
    ))
    
    # VaR and CVaR lines
    fig.add_vline(
        x=var_val * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR 95%: {var_val*100:.2f}%",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=cvar_val * 100,
        line_dash="dot",
        line_color="darkred",
        annotation_text=f"CVaR 95%: {cvar_val*100:.2f}%",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        title=f'Rachev Ratio & VaR/CVaR - {frequency.title()} (R = {rachev_ratio:.2f})',
        xaxis_title=f'{frequency.title()} Return (%)',
        yaxis_title='Density',
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE STYLING FUNCTIONS - Consolidated
# ═══════════════════════════════════════════════════════════════════════════════

def style_returns_table(
    df: pd.DataFrame,
    cdi_returns_dict: Optional[Dict[str, float]] = None,
    mode: str = 'absolute',  # 'absolute' or 'relative'
    sortable: bool = False,
    sort_col: Optional[str] = None,
    sort_ascending: bool = True
) -> str:
    """
    Unified table styling function.
    Consolidates: style_returns_table_with_colors, style_returns_table_relative,
                  style_sortable_returns_table, style_sortable_relative_table
    
    Args:
        df: DataFrame with Fund column and period columns
        cdi_returns_dict: Dict of period -> CDI return value (for absolute mode)
        mode: 'absolute' for raw returns, 'relative' for % of CDI
        sortable: Whether to apply sorting
        sort_col: Column to sort by (if sortable)
        sort_ascending: Sort direction
    
    Returns:
        HTML string for the styled table
    """
    if df is None or len(df) == 0:
        return "<p>No data available</p>"
    
    # Sort if needed
    if sortable and sort_col and sort_col in df.columns:
        df = df.copy()
        df[sort_col] = pd.to_numeric(df[sort_col], errors='coerce')
        df = df.sort_values(by=sort_col, ascending=sort_ascending, na_position='last')
    
    period_cols = [c for c in df.columns if c != 'Fund']
    
    html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 14px; border: 2px solid #D4AF37;">'
    
    # Header
    html += '<thead><tr style="background-color: #D4AF37; color: #000; font-weight: 700;">'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; text-align: left; position: sticky; left: 0; background: #D4AF37; z-index: 1;">Fund</th>'
    
    for col in period_cols:
        sort_indicator = ""
        if sortable and sort_col == col:
            sort_indicator = " ▲" if sort_ascending else " ▼"
        html += f'<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">{col}{sort_indicator}</th>'
    html += '</tr></thead><tbody>'
    
    # Body
    for _, row in df.iterrows():
        fund_name = row['Fund']
        is_cdi = fund_name == 'CDI'
        
        html += '<tr>'
        font_weight = "700" if is_cdi else "400"
        html += f'<td style="padding: 10px; border: 1px solid #333; color: #D4AF37; font-weight: {font_weight}; position: sticky; left: 0; background: #1a1a1a; z-index: 1;">{fund_name}</td>'
        
        for col in period_cols:
            val = row.get(col, np.nan)
            
            if pd.isna(val):
                fv, color = '-', '#888'
            else:
                if mode == 'absolute':
                    fv = f"{val*100:.2f}%"
                    if is_cdi:
                        color = '#00CED1'
                    elif cdi_returns_dict and col in cdi_returns_dict:
                        cdi_val = cdi_returns_dict[col]
                        if val < 0:
                            color = '#F44'
                        elif val <= cdi_val:
                            color = '#48F'
                        else:
                            color = '#FFF'
                    else:
                        color = '#F44' if val < 0 else '#FFF'
                else:  # relative mode
                    fv = f"{val*100:.1f}%"
                    if is_cdi:
                        color = '#00CED1'
                    elif val < 100:
                        color = '#F44'
                    elif val >= 100:
                        color = '#4F4'
                    else:
                        color = '#FFF'
            
            html += f'<td style="padding: 10px; border: 1px solid #333; color: {color}; text-align: right; font-weight: {font_weight};">{fv}</td>'
        html += '</tr>'
    
    html += '</tbody></table></div>'
    return html


def style_book_analysis_table(df: pd.DataFrame, period_cols: List[str]) -> str:
    """
    Style table for Book Analysis with category grouping.
    Specifically handles: Category totals, Portfolio totals, CDI row
    """
    html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 14px; border: 2px solid #D4AF37;">'
    html += '<thead><tr style="background-color: #D4AF37; color: #000; font-weight: 700;">'
    html += '<th style="padding: 10px; border: 1px solid #D4AF37; text-align: left; position: sticky; left: 0; background: #D4AF37; z-index: 1;">Fund</th>'
    for col in period_cols:
        html += f'<th style="padding: 10px; border: 1px solid #D4AF37; text-align: center;">{col}</th>'
    html += '</tr></thead><tbody>'
    
    for _, row in df.iterrows():
        fund_name = row['Fund']
        is_cdi = fund_name == '📈 CDI'  # Exact match for CDI benchmark row
        is_category_total = fund_name.startswith('📁')
        is_portfolio_total = fund_name.startswith('📊')
        
        # Style based on row type
        if is_portfolio_total:
            bg_color, text_color, font_weight = '#2a2a2a', '#D4AF37', '700'
        elif is_cdi:
            bg_color, text_color, font_weight = '#1a1a1a', '#00CED1', '700'
        elif is_category_total:
            bg_color, text_color, font_weight = '#252525', '#FFA500', '600'
        else:
            bg_color, text_color, font_weight = '#1a1a1a', '#FFF', '400'
        
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
                    color = '#F44'
                else:
                    color = '#FFF'
            html += f'<td style="padding: 10px; border: 1px solid #333; color: {color}; text-align: right; font-weight: {font_weight};">{fv}</td>'
        html += '</tr>'
    
    html += '</tbody></table></div>'
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED PERFORMANCE DASHBOARD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_risk_adjusted_metrics(
    returns: pd.Series,
    frequency_key: str = "freq_selector"
) -> None:
    """
    Render Omega and Rachev ratio analysis section.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    st.markdown("#### Risk-Adjusted Performance (Omega & Rachev)")
    
    freq_choice = st.radio(
        "Select frequency:",
        ['Daily', 'Weekly', 'Monthly'],
        horizontal=True,
        key=frequency_key
    )
    
    # Resample returns based on frequency
    if freq_choice == 'Weekly':
        analysis_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
    elif freq_choice == 'Monthly':
        analysis_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    else:
        analysis_returns = returns
    
    analysis_returns = analysis_returns.dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        omega_val = PortfolioMetrics.omega_ratio(analysis_returns)
        fig_omega = create_omega_gauge(omega_val, frequency=freq_choice)
        if fig_omega:
            st.plotly_chart(fig_omega, use_container_width=True, key=f"{frequency_key}_omega")
    
    with col2:
        rachev_val = PortfolioMetrics.rachev_ratio(analysis_returns)
        var_val = np.percentile(analysis_returns, 5)
        cvar_val = analysis_returns[analysis_returns <= var_val].mean()
        fig_rachev = create_var_cvar_chart(analysis_returns, var_val, cvar_val, frequency=freq_choice.lower())
        if fig_rachev:
            st.plotly_chart(fig_rachev, use_container_width=True, key=f"{frequency_key}_rachev")
    
    # Display Rachev gauge below
    fig_rachev_gauge = create_rachev_gauge(rachev_val, frequency=freq_choice)
    if fig_rachev_gauge:
        st.plotly_chart(fig_rachev_gauge, use_container_width=True, key=f"{frequency_key}_rachev_gauge")


def render_sharpe_volatility_analysis(
    returns: pd.Series,
    key_prefix: str = "sharpe_vol"
) -> None:
    """
    Render rolling Sharpe and volatility analysis section.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    st.markdown("#### Sharpe & Volatility Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sharpe = create_rolling_sharpe_chart(returns, window_months=12)
        if fig_sharpe:
            st.plotly_chart(fig_sharpe, use_container_width=True, key=f"{key_prefix}_sharpe")
        
        # Period Sharpe metrics
        st.markdown("**Period Sharpe Ratios**")
        periods = [
            ('12M', 252),
            ('24M', 504),
            ('36M', 756)
        ]
        
        sharpe_cols = st.columns(3)
        for i, (label, days) in enumerate(periods):
            with sharpe_cols[i]:
                if len(returns) >= days:
                    sharpe = PortfolioMetrics.sharpe_ratio(returns.tail(days))
                    st.metric(label, f"{sharpe:.3f}" if not np.isnan(sharpe) else "N/A")
                else:
                    st.metric(label, "N/A")
    
    with col2:
        fig_vol = create_rolling_volatility_chart(returns, window_months=12)
        if fig_vol:
            st.plotly_chart(fig_vol, use_container_width=True, key=f"{key_prefix}_vol")
        
        # Period volatility metrics
        st.markdown("**Period Volatility (Annualized)**")
        
        vol_cols = st.columns(3)
        for i, (label, days) in enumerate(periods):
            with vol_cols[i]:
                if len(returns) >= days:
                    vol = PortfolioMetrics.annualized_volatility(returns.tail(days))
                    st.metric(label, f"{vol*100:.2f}%" if not np.isnan(vol) else "N/A")
                else:
                    st.metric(label, "N/A")


def render_drawdown_analysis(
    returns: pd.Series,
    key_prefix: str = "drawdown"
) -> None:
    """
    Render drawdown analysis section.
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    st.markdown("#### Drawdown Analysis")
    
    fig_underwater, max_dd_info = create_underwater_chart(returns)
    if fig_underwater:
        st.plotly_chart(fig_underwater, use_container_width=True, key=f"{key_prefix}_underwater")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mdd = PortfolioMetrics.max_drawdown(returns)
        st.metric("Max Drawdown", f"{mdd*100:.2f}%" if not np.isnan(mdd) else "N/A")
    
    with col2:
        if max_dd_info and max_dd_info.get('max_drawdown_date'):
            st.metric("MDD Date", max_dd_info['max_drawdown_date'].strftime('%Y-%m-%d'))
        else:
            st.metric("MDD Date", "N/A")
    
    with col3:
        # Recovery info (simplified)
        cumulative = (1 + returns).cumprod()
        is_recovered = cumulative.iloc[-1] >= cumulative.max()
        st.metric("Recovered", "✅ Yes" if is_recovered else "❌ No")


def render_full_performance_dashboard(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    name: str = "Portfolio",
    key_prefix: str = "dashboard"
) -> None:
    """
    Render complete performance dashboard with all analysis sections.
    This is the main unified function that replaces duplicate code in all tabs.
    
    Used by: Detailed Analysis, Portfolio Construction, Recommended Portfolio
    """
    if returns is None or len(returns) < 2:
        st.warning("Insufficient data for analysis")
        return
    
    # 1. Cumulative Returns Chart
    st.markdown("---")
    st.markdown("#### Cumulative Returns")
    fig_cum = create_cumulative_returns_chart(
        returns, 
        benchmark_returns, 
        fund_name=name,
        benchmark_name="CDI"
    )
    if fig_cum:
        st.plotly_chart(fig_cum, use_container_width=True, key=f"{key_prefix}_cumulative")
    
    # 2. Risk-Adjusted Metrics (Omega/Rachev)
    st.markdown("---")
    render_risk_adjusted_metrics(returns, frequency_key=f"{key_prefix}_freq")
    
    # 3. Sharpe & Volatility
    st.markdown("---")
    render_sharpe_volatility_analysis(returns, key_prefix=f"{key_prefix}_sv")
    
    # 4. Drawdown Analysis
    st.markdown("---")
    render_drawdown_analysis(returns, key_prefix=f"{key_prefix}_dd")


# ═══════════════════════════════════════════════════════════════════════════════
# PIE CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_portfolio_pie_chart(
    weights: pd.Series,
    view_type: str = 'fund',  # 'fund', 'category', 'subcategory'
    fund_categories: Optional[Dict[str, str]] = None,
    fund_subcategories: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Create portfolio allocation pie chart with different view modes.
    """
    if view_type == 'category' and fund_categories:
        # Aggregate by category
        cat_weights = {}
        for fund, weight in weights.items():
            cat = fund_categories.get(fund, 'Unknown')
            cat_weights[cat] = cat_weights.get(cat, 0) + weight
        labels = list(cat_weights.keys())
        values = list(cat_weights.values())
        title = "Portfolio Allocation by Category"
    elif view_type == 'subcategory' and fund_subcategories:
        # Aggregate by subcategory
        subcat_weights = {}
        for fund, weight in weights.items():
            subcat = fund_subcategories.get(fund, 'Unknown')
            subcat_weights[subcat] = subcat_weights.get(subcat, 0) + weight
        labels = list(subcat_weights.keys())
        values = list(subcat_weights.values())
        title = "Portfolio Allocation by Subcategory"
    else:
        # By fund
        labels = list(weights.index)
        values = list(weights.values)
        title = "Portfolio Allocation by Fund"
    
    colors = px.colors.qualitative.Bold
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='percent+label',
        textposition='outside',
        hovertemplate='%{label}<br>%{percent}<br>%{value:.2%}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='#D4AF37', size=18)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
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
