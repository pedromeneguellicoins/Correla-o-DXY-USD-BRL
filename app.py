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
st.caption("BRL/USD vs DXY (G10) + DTWEXBGS (Broad, inclui emergentes) | Dados: BCB SGS + Yahoo Finance + FRED")

# --- Inicializa FRED ---
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

# --- Datas ---
end_date = datetime.today()
start_date = end_date - timedelta(days=anos * 365)

# --- Funções de carregamento (com cache) ---
@st.cache_data(ttl=3600)
def carrega_ptax(start, end):
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
        f"?formato=json"
        f"&dataInicial={start.strftime('%d/%m/%Y')}"
        f"&dataFinal={end.strftime('%d/%m/%Y')}"
    )
    df = pd.DataFrame(requests.get(url).json())
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df['valor'] = df['valor'].astype(float)
    return df.set_index('data').rename(columns={'valor': 'PTAX'})

@st.cache_data(ttl=3600)
def carrega_dxy(start, end):
    df = yf.download('DX-Y.NYB', start=start, end=end, progress=False)[['Close']]
    df.columns = ['DXY']
    return df

@st.cache_data(ttl=3600)
def carrega_fred(serie, start, end):
    """Carrega série do FRED. DTWEXBGS é semanal — fazemos forward-fill pra alinhar com PTAX diário."""
    s = fred.get_series(serie, observation_start=start, observation_end=end)
    df = pd.DataFrame(s, columns=[serie])
    df.index = pd.to_datetime(df.index)
    return df

# --- Carregamento ---
with st.spinner("Carregando dados..."):
    ptax = carrega_ptax(start_date, end_date)
    df = ptax.copy()

    if "DXY" in indices_selecionados:
        dxy = carrega_dxy(start_date, end_date)
        df = df.join(dxy, how='left')

    if "DTWEXBGS" in indices_selecionados:
        twd = carrega_fred("DTWEXBGS", start_date, end_date)
        # Forward-fill porque DTWEXBGS é semanal
        df = df.join(twd, how='left').ffill()

    df = df.dropna()

    # Retornos
    df['ret_ptax'] = df['PTAX'].pct_change()
    for idx in indices_selecionados:
        df[f'ret_{idx}'] = df[idx].pct_change()
        df[f'corr_{idx}'] = df['ret_ptax'].rolling(janela_corr).corr(df[f'ret_{idx}'])

# --- Métricas no topo ---
cols = st.columns(2 + len(indices_selecionados))
cols[0].metric("PTAX atual", f"R$ {df['PTAX'].iloc[-1]:.4f}",
               f"{(df['PTAX'].iloc[-1]/df['PTAX'].iloc[-2]-1)*100:.2f}%")

for i, idx in enumerate(indices_selecionados):
    cols[i+1].metric(
        f"{idx} atual",
        f"{df[idx].iloc[-1]:.2f}",
        f"{(df[idx].iloc[-1]/df[idx].iloc[-2]-1)*100:.2f}%"
    )

# Correlação atual de cada índice
last_col = len(indices_selecionados) + 1
corr_text = " | ".join([f"{idx}: {df[f'corr_{idx}'].iloc[-1]:.2f}" for idx in indices_selecionados])
cols[last_col].metric(f"Corr {janela_corr}d", corr_text)

# --- Gráficos ---
n_rows = 2 + len(indices_selecionados)  # PTAX + cada índice + correlações
titulos = ["PTAX (BRL/USD)"] + indices_selecionados + [f"Correlações Rolling {janela_corr}d"]

fig = make_subplots(
    rows=n_rows, cols=1, shared_xaxes=True,
    subplot_titles=titulos, vertical_spacing=0.05
)

# Linha 1: PTAX
fig.add_trace(
    go.Scatter(x=df.index, y=df['PTAX'], name='PTAX', line=dict(color='green')),
    row=1, col=1
)

# Linhas intermediárias: cada índice de dólar
cores_idx = {'DXY': 'blue', 'DTWEXBGS': 'darkorange'}
for i, idx in enumerate(indices_selecionados):
    fig.add_trace(
        go.Scatter(x=df.index, y=df[idx], name=idx, line=dict(color=cores_idx.get(idx, 'gray'))),
        row=i+2, col=1
    )

# Última linha: correlações sobrepostas
for idx in indices_selecionados:
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
st.subheader("📈 Estatísticas comparativas (correlação rolling)")

stats = []
for idx in indices_selecionados:
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

st.dataframe(
    pd.DataFrame(stats).round(3),
    use_container_width=True, hide_index=True
)

# --- Spread DXY vs DTWEXBGS ---
if "DXY" in indices_selecionados and "DTWEXBGS" in indices_selecionados:
    st.subheader("🔍 Diferencial de correlação: DTWEXBGS – DXY")
    st.caption("Quando positivo, BRL está mais sensível à cesta ampla (incluindo CNY/emergentes) do que ao G10. Sinal de regime emergente-driven.")

    df['diff_corr'] = df['corr_DTWEXBGS'] - df['corr_DXY']

    fig_diff = go.Figure()
    fig_diff.add_trace(go.Scatter(
        x=df.index, y=df['diff_corr'],
        fill='tozeroy', name='Diff (DTWEXBGS - DXY)',
        line=dict(color='purple')
    ))
    fig_diff.add_hline(y=0, line_dash="dash", line_color="black")
    fig_diff.update_layout(height=300, hovermode='x unified')
    st.plotly_chart(fig_diff, use_container_width=True)

st.caption("Construído por Pedro | Dados: BCB SGS (série 1) + Yahoo Finance (DX-Y.NYB) + FRED (DTWEXBGS)")
