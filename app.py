import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from fredapi import Fred

# --- Config da página ---
st.set_page_config(page_title="PTAX vs Dollar Indices", layout="wide")
st.title("📊 PTAX vs Dollar Indices — Análise Multi-Indicador")
st.caption("BRL/USD vs DXY (G10) + DTWEXBGS (Broad, inclui emergentes)")

# --- Validação do secret ---
if "FRED_API_KEY" not in st.secrets:
    st.error("⚠️ FRED_API_KEY não configurada nos Secrets do Streamlit.")
    st.info("Vai em Manage app → Settings → Secrets e adiciona: FRED_API_KEY = \"sua_chave\"")
    st.stop()

fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# --- Sidebar ---
st.sidebar.header("Parâmetros")
anos = st.sidebar.slider("Anos de histórico", 1, 10, 5)
janela_corr = st.sidebar.slider("Janela de correlação (dias)", 10, 90, 30)

indices_selecionados = st.sidebar.multiselect(
    "Índices de dólar para comparar com PTAX",
    options=["DXY", "DTWEXBGS"],
    default=["DXY", "DTWEXBGS"]
)

debug_mode = st.sidebar.checkbox("Modo debug (mostra diagnóstico)", value=True)

# --- Datas ---
end_date = datetime.today()
start_date = end_date - timedelta(days=anos * 365)

# --- Funções de carregamento ---
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

# --- Carregamento ---
with st.spinner("Carregando dados..."):
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

# --- Debug ---
if debug_mode:
    st.subheader("🔧 Diagnóstico")
    diag_df = pd.DataFrame.from_dict(diagnostico, orient='index', columns=['Linhas carregadas'])
    st.dataframe(diag_df)
    if len(df) > 0:
        st.write(f"Período: **{df.index.min().date()}** até **{df.index.max().date()}**")
        st.write("Primeiras linhas:")
        st.dataframe(df.head(3))
        st.write("Últimas linhas:")
        st.dataframe(df.tail(3))

# --- Validação ---
if len(df) == 0:
    st.error("❌ DataFrame vazio após merge. Confere o diagnóstico acima.")
    st.stop()

if len(df) < janela_corr + 5:
    st.error(f"❌ Dados insuficientes ({len(df)} linhas) para janela de correlação de {janela_corr} dias.")
    st.stop()

# --- Retornos e correlações ---
df['ret_ptax'] = df['PTAX'].pct_change()
for idx in cols_idx:
    df[f'ret_{idx}'] = df[idx].pct_change()
    df[f'corr_{idx}'] = df['ret_ptax'].rolling(janela_corr).corr(df[f'ret_{idx}'])

# --- Métricas no topo ---
cols = st.columns(1 + len(cols_idx) + 1)

cols[0].metric(
    "PTAX atual",
    f"R$ {df['PTAX'].iloc[-1]:.4f}",
    f"{(df['PTAX'].iloc[-1]/df['PTAX'].iloc[-2]-1)*100:.2f}%"
)

for i, idx in enumerate(cols_idx):
    cols[i+1].metric(
        f"{idx} atual",
        f"{df[idx].iloc[-1]:.2f}",
        f"{(df[idx].iloc[-1]/df[idx].iloc[-2]-1)*100:.2f}%"
    )

corr_text = " | ".join([f"{idx}: {df[f'corr_{idx}'].iloc[-1]:.2f}" for idx in cols_idx])
cols[-1].metric(f"Corr {janela_corr}d", corr_text)

# --- Gráficos ---
n_rows = 1 + len(cols_idx) + 1
titulos = ["PTAX (BRL/USD)"] + cols_idx + [f"Correlações Rolling {janela_corr}d"]

fig = make_subplots(
    rows=n_rows, cols=1, shared_xaxes=True,
    subplot_titles=titulos, vertical_spacing=0.05
)

fig.add_trace(
    go.Scatter(x=df.index, y=df['PTAX'], name='PTAX', line=dict(color='green')),
    row=1, col=1
)

cores_idx = {'DXY': 'blue', 'DTWEXBGS': 'darkorange'}
for i, idx in enumerate(cols_idx):
    fig.add_trace(
        go.Scatter(x=df.index, y=df[idx], name=idx, line=dict(color=cores_idx.get(idx, 'gray'))),
        row=i+2, col=1
    )

for idx in cols_idx:
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[f'corr_{idx}'],
            name=f'Corr PTAX-{idx}',
            line=dict(color=cores_idx.get(idx, 'gray'))
        ),
        row=n_rows, col=1
    )
fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=n_rows, col=1)

fig.update_layout(height=250*n_rows, showlegend=True, hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# --- Tabela comparativa ---
st.subheader("📈 Estatísticas comparativas")
stats = []
for idx in cols_idx:
    corr_serie = df[f'corr_{idx}'].dropna()
    stats.append({
        'Índice': idx,
        'Corr média': corr_serie.mean(),
        'Corr mediana': corr_serie.median(),
        'Corr mínima': corr_serie.min(),
        'Corr máxima': corr_serie.max(),
        '% tempo > 0': (corr_serie > 0).mean() * 100,
        'Corr atual': corr_serie.iloc[-1]
    })

st.dataframe(pd.DataFrame(stats).round(3), use_container_width=True, hide_index=True)

# --- Diferencial DTWEXBGS - DXY ---
if "DXY" in cols_idx and "DTWEXBGS" in cols_idx:
    st.subheader("🔍 Diferencial: corr(PTAX, DTWEXBGS) − corr(PTAX, DXY)")
    st.caption("Positivo: BRL mais sensível à cesta ampla (regime EM). Negativo: BRL mais G10-driven.")

    df['diff_corr'] = df['corr_DTWEXBGS'] - df['corr_DXY']

    fig_diff = go.Figure()
    fig_diff.add_trace(go.Scatter(
        x=df.index, y=df['diff_corr'],
        fill='tozeroy', name='Diff',
        line=dict(color='purple')
    ))
    fig_diff.add_hline(y=0, line_dash="dash", line_color="black")
    fig_diff.update_layout(height=300, hovermode='x unified')
    st.plotly_chart(fig_diff, use_container_width=True)

st.caption("BCB SGS · Yahoo Finance · FRED")
