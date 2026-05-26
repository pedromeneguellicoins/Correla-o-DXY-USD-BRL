import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from fredapi import Fred

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="BRL Macro Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de cores consistente (tema dark)
COLORS = {
    'ptax': '#00D4AA',       # verde-água (destaque)
    'dxy': '#4A9EFF',        # azul
    'dtwexbgs': '#FF9F40',   # laranja
    'corr_pos': '#00D4AA',
    'corr_neg': '#FF4B6E',
    'neutral': '#8B92A8',
    'bg_card': '#1A1F2E',
    'text_dim': '#8B92A8',
}

# CSS customizado pra polish extra
st.markdown("""
<style>
    /* Header customizado */
    .main-header {
        padding: 1rem 0;
        border-bottom: 1px solid #2A3142;
        margin-bottom: 1.5rem;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.9rem;
        color: #8B92A8;
        margin-top: 0.3rem;
    }
    .header-meta {
        font-size: 0.8rem;
        color: #8B92A8;
        font-family: monospace;
    }
    
    /* KPI cards */
    .kpi-card {
        background-color: #1A1F2E;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 3px solid #00D4AA;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #8B92A8;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 0.3rem 0;
        font-family: monospace;
    }
    .kpi-delta-pos { color: #00D4AA; font-size: 0.85rem; font-family: monospace; }
    .kpi-delta-neg { color: #FF4B6E; font-size: 0.85rem; font-family: monospace; }
    .kpi-delta-neutral { color: #8B92A8; font-size: 0.85rem; font-family: monospace; }
    
    /* Esconder o "Made with Streamlit" */
    footer {visibility: hidden;}
    
    /* Customizar tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1F2E;
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00D4AA !important;
        color: #0E1117 !important;
    }
    
    /* Footer custom */
    .footer-custom {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2A3142;
        text-align: center;
        color: #8B92A8;
        font-size: 0.8rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# VALIDAÇÃO DO SECRET
# ============================================================
if "FRED_API_KEY" not in st.secrets:
    st.error("⚠️ FRED_API_KEY não configurada nos Secrets do Streamlit.")
    st.stop()

fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Parâmetros")
    
    st.markdown("**Período de análise**")
    anos = st.slider("Anos de histórico", 1, 10, 5, label_visibility="collapsed")
    
    st.markdown("**Janela de correlação**")
    janela_corr = st.slider("Dias", 10, 90, 30, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📊 Índices")
    indices_selecionados = st.multiselect(
        "Selecione",
        options=["DXY", "DTWEXBGS"],
        default=["DXY", "DTWEXBGS"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    debug_mode = st.checkbox("🔧 Modo debug", value=False)
    
    st.markdown("---")
    st.markdown(
        f"<p style='color:#8B92A8; font-size:0.75rem; font-family:monospace;'>"
        f"Última atualização<br>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
        unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
        <div>
            <p class="header-title">📈 BRL Macro Monitor</p>
            <p class="header-subtitle">Análise de correlação PTAX × Dollar Indices</p>
        </div>
        <div class="header-meta">
            <div>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC-3')}</div>
            <div>Sources: BCB SGS · Yahoo Finance · FRED</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DATAS
# ============================================================
end_date = datetime.today()
start_date = end_date - timedelta(days=anos * 365)

# ============================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================
@st.cache_data(ttl=3600)
def carrega_ptax(start, end):
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
        f"?formato=json"
        f"&dataInicial={start.strftime('%d/%m/%Y')}"
        f"&dataFinal={end.strftime('%d/%m/%Y')}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df['valor'] = df['valor'].astype(float)
    return df.set_index('data').rename(columns={'valor': 'PTAX'})

@st.cache_data(ttl=3600)
def carrega_dxy(start, end):
    df = yf.download('DX-Y.NYB', start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].copy()
    df.columns = ['DXY']
    return df

@st.cache_data(ttl=3600)
def carrega_fred(serie, start, end):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    try:
        s = fred.get_series(serie, observation_start=start_str, observation_end=end_str)
        df = pd.DataFrame(s, columns=[serie])
        df.index = pd.to_datetime(df.index)
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar {serie} do FRED: {e}")
        return pd.DataFrame()

# ============================================================
# CARREGAMENTO
# ============================================================
with st.spinner("Carregando dados de mercado..."):
    ptax = carrega_ptax(start_date, end_date)
    df = ptax.copy()
    diagnostico = {"PTAX": len(ptax)}

    if "DXY" in indices_selecionados:
        dxy = carrega_dxy(start_date, end_date)
        diagnostico["DXY"] = len(dxy)
        if not dxy.empty:
            df = df.join(dxy, how='left')

    if "DTWEXBGS" in indices_selecionados:
        twd = carrega_fred("DTWEXBGS", start_date, end_date)
        diagnostico["DTWEXBGS"] = len(twd)
        if not twd.empty:
            df = df.join(twd, how='left')

    cols_idx = [c for c in indices_selecionados if c in df.columns]
    df = df.sort_index()
    for col in cols_idx:
        df[col] = df[col].ffill()
    df = df.dropna(subset=['PTAX'] + cols_idx)
    diagnostico["DataFrame final"] = len(df)

if len(df) == 0:
    st.error("❌ DataFrame vazio após merge.")
    st.stop()
if len(df) < janela_corr + 5:
    st.error(f"❌ Dados insuficientes ({len(df)} linhas).")
    st.stop()

# Retornos e correlações
df['ret_ptax'] = df['PTAX'].pct_change()
for idx in cols_idx:
    df[f'ret_{idx}'] = df[idx].pct_change()
    df[f'corr_{idx}'] = df['ret_ptax'].rolling(janela_corr).corr(df[f'ret_{idx}'])

# ============================================================
# KPI CARDS CUSTOMIZADOS
# ============================================================
def render_kpi(label, value, delta_value=None, delta_format="pct"):
    """Renderiza KPI card customizado."""
    delta_html = ""
    if delta_value is not None:
        if delta_format == "pct":
            delta_str = f"{delta_value:+.2f}%"
        else:
            delta_str = f"{delta_value:+.3f}"
        css_class = "kpi-delta-pos" if delta_value > 0 else "kpi-delta-neg" if delta_value < 0 else "kpi-delta-neutral"
        arrow = "▲" if delta_value > 0 else "▼" if delta_value < 0 else "●"
        delta_html = f'<div class="{css_class}">{arrow} {delta_str}</div>'
    
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """

# Renderiza KPIs
ptax_now = df['PTAX'].iloc[-1]
ptax_chg = (df['PTAX'].iloc[-1]/df['PTAX'].iloc[-2]-1)*100

kpi_cols = st.columns(1 + len(cols_idx) + 1)
kpi_cols[0].markdown(render_kpi("PTAX", f"R$ {ptax_now:.4f}", ptax_chg), unsafe_allow_html=True)

for i, idx in enumerate(cols_idx):
    val = df[idx].iloc[-1]
    chg = (df[idx].iloc[-1]/df[idx].iloc[-2]-1)*100
    kpi_cols[i+1].markdown(render_kpi(idx, f"{val:.2f}", chg), unsafe_allow_html=True)

corr_now = df[f'corr_{cols_idx[0]}'].iloc[-1] if cols_idx else 0
kpi_cols[-1].markdown(
    render_kpi(f"Corr {janela_corr}d", f"{corr_now:.3f}", None),
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔗 Correlações", "🔍 Diferencial", "📋 Dados"])

# Layout Plotly comum (tema dark)
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(family="monospace", size=11, color="#FAFAFA"),
    hovermode='x unified',
    margin=dict(l=40, r=20, t=40, b=40),
)

# --- TAB 1: Overview ---
with tab1:
    fig = make_subplots(
        rows=1 + len(cols_idx), cols=1, shared_xaxes=True,
        subplot_titles=["PTAX (BRL/USD)"] + cols_idx,
        vertical_spacing=0.08
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['PTAX'], name='PTAX',
                   line=dict(color=COLORS['ptax'], width=2)),
        row=1, col=1
    )
    
    cores_idx = {'DXY': COLORS['dxy'], 'DTWEXBGS': COLORS['dtwexbgs']}
    for i, idx in enumerate(cols_idx):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[idx], name=idx,
                       line=dict(color=cores_idx.get(idx, '#888'), width=2)),
            row=i+2, col=1
        )
    
    fig.update_layout(**PLOTLY_LAYOUT, height=250*(1+len(cols_idx)), showlegend=False)
    fig.update_xaxes(gridcolor="#2A3142")
    fig.update_yaxes(gridcolor="#2A3142")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Correlações ---
with tab2:
    fig_corr = go.Figure()
    
    for idx in cols_idx:
        fig_corr.add_trace(go.Scatter(
            x=df.index, y=df[f'corr_{idx}'],
            name=f'PTAX × {idx}',
            line=dict(color=cores_idx.get(idx, '#888'), width=2)
        ))
    
    fig_corr.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'], opacity=0.5)
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=450,
                            title=f"Correlação Rolling {janela_corr} dias (retornos)")
    fig_corr.update_xaxes(gridcolor="#2A3142")
    fig_corr.update_yaxes(gridcolor="#2A3142", range=[-1, 1])
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Tabela de estatísticas
    st.markdown("##### 📈 Estatísticas comparativas")
    stats = []
    for idx in cols_idx:
        corr_serie = df[f'corr_{idx}'].dropna()
        stats.append({
            'Índice': idx,
            'Média': corr_serie.mean(),
            'Mediana': corr_serie.median(),
            'Mín': corr_serie.min(),
            'Máx': corr_serie.max(),
            '% > 0': (corr_serie > 0).mean() * 100,
            'Atual': corr_serie.iloc[-1]
        })
    st.dataframe(pd.DataFrame(stats).round(3), use_container_width=True, hide_index=True)

# --- TAB 3: Diferencial ---
with tab3:
    if "DXY" in cols_idx and "DTWEXBGS" in cols_idx:
        st.markdown(
            f"<p style='color:#8B92A8;'>"
            f"Diferencial = <code>corr(PTAX, DTWEXBGS) − corr(PTAX, DXY)</code><br>"
            f"<span style='color:{COLORS['corr_pos']};'>● Positivo:</span> BRL mais sensível à cesta ampla (regime EM-driven, inclui China)<br>"
            f"<span style='color:{COLORS['corr_neg']};'>● Negativo:</span> BRL mais G10-driven (carry ou risco idiossincrático local)"
            f"</p>",
            unsafe_allow_html=True
        )
        
        df['diff_corr'] = df['corr_DTWEXBGS'] - df['corr_DXY']
        
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(
            x=df.index, y=df['diff_corr'],
            fill='tozeroy',
            line=dict(color='#A78BFA', width=2),
            fillcolor='rgba(167, 139, 250, 0.2)',
            name='Diff'
        ))
        fig_diff.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'])
        fig_diff.update_layout(**PLOTLY_LAYOUT, height=500,
                                title="Diferencial de Correlação (regime indicator)")
        fig_diff.update_xaxes(gridcolor="#2A3142")
        fig_diff.update_yaxes(gridcolor="#2A3142")
        st.plotly_chart(fig_diff, use_container_width=True)
        
        # Stats do diferencial
        st.markdown("##### Estatísticas do diferencial")
        col_a, col_b, col_c, col_d = st.columns(4)
        diff_serie = df['diff_corr'].dropna()
        col_a.metric("Atual", f"{diff_serie.iloc[-1]:+.3f}")
        col_b.metric("Média 5y", f"{diff_serie.mean():+.3f}")
        col_c.metric("% tempo EM-driven", f"{(diff_serie > 0).mean()*100:.1f}%")
        col_d.metric("Mediana", f"{diff_serie.median():+.3f}")
    else:
        st.info("Selecione DXY e DTWEXBGS na sidebar para ver o diferencial.")

# --- TAB 4: Dados ---
with tab4:
    st.markdown("##### 📋 Dados brutos")
    st.dataframe(
        df[['PTAX'] + cols_idx + [f'corr_{i}' for i in cols_idx]].round(4),
        use_container_width=True,
        height=500
    )
    
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=f"brl_macro_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )
    
    if debug_mode:
        st.markdown("##### 🔧 Diagnóstico")
        diag_df = pd.DataFrame.from_dict(diagnostico, orient='index', columns=['Linhas'])
        st.dataframe(diag_df, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-custom">
    BRL Macro Monitor v1.0 · Built with Streamlit · Data: BCB SGS (PTAX 1) · Yahoo Finance (DXY) · FRED (DTWEXBGS)
</div>
""", unsafe_allow_html=True)
