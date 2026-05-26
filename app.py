import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- Config da página ---
st.set_page_config(page_title="PTAX vs DXY Dashboard", layout="wide")
st.title("📊 PTAX vs DXY — Análise de Correlação")
st.caption("Dashboard de monitoramento BRL/USD vs Dollar Index | Dados: BCB SGS + Yahoo Finance")

# --- Sidebar com controles ---
st.sidebar.header("Parâmetros")
anos = st.sidebar.slider("Anos de histórico", 1, 10, 5)
janela_corr = st.sidebar.slider("Janela de correlação (dias)", 10, 90, 30)

# --- Datas ---
end_date = datetime.today()
start_date = end_date - timedelta(days=anos * 365)

# --- Carregamento de dados (com cache pra não recarregar toda hora) ---
@st.cache_data(ttl=3600)  # cache de 1h
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

with st.spinner("Carregando dados..."):
    ptax = carrega_ptax(start_date, end_date)
    dxy = carrega_dxy(start_date, end_date)
    df = ptax.join(dxy, how='inner').dropna()
    df['ret_ptax'] = df['PTAX'].pct_change()
    df['ret_dxy'] = df['DXY'].pct_change()
    df['corr'] = df['ret_ptax'].rolling(janela_corr).corr(df['ret_dxy'])

# --- Métricas no topo ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("PTAX atual", f"R$ {df['PTAX'].iloc[-1]:.4f}",
            f"{(df['PTAX'].iloc[-1]/df['PTAX'].iloc[-2]-1)*100:.2f}%")
col2.metric("DXY atual", f"{df['DXY'].iloc[-1]:.2f}",
            f"{(df['DXY'].iloc[-1]/df['DXY'].iloc[-2]-1)*100:.2f}%")
col3.metric(f"Corr {janela_corr}d", f"{df['corr'].iloc[-1]:.3f}")
col4.metric(f"Corr média ({anos}a)", f"{df['corr'].mean():.3f}")

# --- Top 3 descorrelações ---
df_corr = df['corr'].dropna()
window_min = df_corr.rolling(60, center=True).min()
candidates = df_corr[df_corr == window_min].sort_values().head(10)
top3 = []
for date, val in candidates.items():
    if all(abs((date - d).days) > 90 for d, _ in top3):
        top3.append((date, val))
    if len(top3) == 3:
        break

# --- Gráfico interativo Plotly ---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=("PTAX (BRL/USD)", "DXY", f"Correlação Rolling {janela_corr}d"),
                    vertical_spacing=0.08)

fig.add_trace(go.Scatter(x=df.index, y=df['PTAX'], name='PTAX',
                         line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['DXY'], name='DXY',
                         line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['corr'], name=f'Corr {janela_corr}d',
                         line=dict(color='purple')), row=3, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)

for date, val in top3:
    fig.add_vline(x=date, line_dash="dot", line_color="orange", opacity=0.7)

fig.update_layout(height=800, showlegend=False, hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# --- Tabela dos top 3 ---
st.subheader("🎯 Top 3 períodos de maior descorrelação")
top3_df = pd.DataFrame(top3, columns=['Data', 'Correlação'])
top3_df['Data'] = top3_df['Data'].dt.strftime('%Y-%m-%d')
top3_df['Correlação'] = top3_df['Correlação'].round(3)
st.dataframe(top3_df, use_container_width=True, hide_index=True)

# --- Footer ---
st.caption("Construído por Pedro | Dados: BCB SGS (série 1) + Yahoo Finance (DX-Y.NYB)")
