# Streamlit – Mechanical Properties App (Tension/Compression)
# Author: ChatGPT (GPT-5 Thinking)
# Description:
#   Upload one or multiple datasets (CSV) from mechanical tests and compute
#   key properties: Young's modulus (manual window or auto-detect),
#   0.2% offset yield, ultimate strength, fracture strain, and toughness
#   (area under σ–ε). Works for tension or compression. Accepts either
#   stress–strain data directly or force–displacement + geometry to convert.

import io
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter

st.set_page_config(page_title="Mechanical Properties – σ–ε", page_icon="🧱", layout="wide")

# ----------------------------- Data Classes ------------------------------ #
@dataclass
class Metrics:
    sample: str
    E_GPa: float
    E_window: Tuple[float, float]
    E_R2: float
    yield_offset: float
    sig_y_MPa: float
    eps_y: float
    uts_MPa: float
    eps_uts: float
    eps_fracture: float
    toughness_MJm3: float

# ----------------------------- Conversions ------------------------------- #
FORCE_FACTORS = {"N": 1.0, "kN": 1e3}
LENGTH_FACTORS = {"mm": 1e-3, "m": 1.0}
AREA_FACTORS = {"mm²": 1e-6, "m²": 1.0}
STRESS_FACTORS = {"MPa": 1e6, "Pa": 1.0}

# --------------------------- Utility Functions --------------------------- #
def to_engineering_stress_strain(df: pd.DataFrame,
                                 mode: str,
                                 cols: Dict[str, str],
                                 geom: Dict,
                                 units: Dict,
                                 make_positive: bool = True) -> pd.DataFrame:
    """Return DataFrame with columns: strain (dimensionless), stress_MPa.
    mode: 'stress_strain' or 'force_disp'
    cols: column mapping
    geom: {'type': 'rect'|'cyl', 'w_mm', 't_mm', 'd_mm', 'L0_mm'}
    units: {'force': 'N'|'kN', 'disp': 'mm'|'m', 'stress': 'MPa'|'Pa', 'strain_in_percent': bool}
    """
    if mode == 'stress_strain':
        eps = df[cols['strain']].astype(float).to_numpy()
        if units.get('strain_in_percent', False):
            eps = eps / 100.0
        sig = df[cols['stress']].astype(float).to_numpy()
        # Convert stress to MPa
        sig_MPa = sig * (STRESS_FACTORS[units['stress']] / 1e6)
    else:
        F = df[cols['force']].astype(float).to_numpy() * FORCE_FACTORS[units['force']]
        disp = df[cols['disp']].astype(float).to_numpy() * LENGTH_FACTORS[units['disp']]
        # Geometry
        if geom['type'] == 'rect':
            w = geom['w_mm'] * 1e-3
            t = geom['t_mm'] * 1e-3
            A = w * t
        else:
            d = geom['d_mm'] * 1e-3
            A = np.pi * (d/2.0)**2
        L0 = geom['L0_mm'] * 1e-3
        eps = disp / max(L0, 1e-12)
        sig = F / max(A, 1e-12)  # Pa
        sig_MPa = sig / 1e6
    if make_positive:
        # For compression data sometimes negative; make positive for metrics
        sig_MPa = np.abs(sig_MPa)
        eps = np.abs(eps)
    df_out = pd.DataFrame({'strain': eps, 'stress_MPa': sig_MPa})
    df_out = df_out.replace([np.inf, -np.inf], np.nan).dropna()
    df_out = df_out.sort_values('strain')
    return df_out


def smooth_series(y: np.ndarray, enabled: bool, window: int, poly: int = 2) -> np.ndarray:
    if not enabled:
        return y
    window = max(5, window)
    if window % 2 == 0:
        window += 1
    try:
        return savgol_filter(y, window_length=window, polyorder=poly)
    except Exception:
        return y


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, intercept, R^2 for y ≈ m x + b."""
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = m * x + b
    ss_res = float(np.sum((y - y_pred)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(m), float(b), float(R2)


def auto_modulus_region(eps: np.ndarray, sig: np.ndarray, max_eps: float = 0.02, window: float = 0.004, step: float = 0.001) -> Tuple[Tuple[float, float], float, float]:
    """Scan windows up to max_eps to find the best linear region.
    Returns (eps_min, eps_max), slope(GPa), R^2.
    """
    mask = eps <= max_eps
    e = eps[mask]; s = sig[mask]
    if len(e) < 5:
        return (np.nan, np.nan), np.nan, np.nan
    best = (None, -np.inf, None)  # (window, R2, slope)
    e_min = float(np.min(e)); e_max = float(np.max(e))
    start = e_min
    while start + window <= e_max:
        end = start + window
        idx = (e >= start) & (e <= end)
        if idx.sum() >= 5:
            m, b, R2 = linear_fit(e[idx], s[idx])
            # prefer high R2 and non-crazy slope
            score = R2
            if score > best[1]:
                best = ((start, end), score, m)
        start += step
    if best[0] is None:
        return (np.nan, np.nan), np.nan, np.nan
    (a,bw), R2, m = best
    return (a, bw), float(m/1000.0), float(R2)  # slope in GPa (MPa/strain -> MPa => /1000)


def yield_by_offset(eps: np.ndarray, sig: np.ndarray, E_GPa: float, offset: float = 0.002) -> Tuple[float, float]:
    """Return (sigma_y_MPa, eps_y) by 0.2% offset using given modulus E (GPa).
    Solve σ(ε) = E*(ε - offset). Uses linear interpolation on residuals.
    """
    if not np.isfinite(E_GPa) or E_GPa <= 0:
        return np.nan, np.nan
    E = E_GPa * 1000.0  # MPa
    resid = sig - (E * (eps - offset))
    # find first crossing from negative to positive
    sign = np.sign(resid)
    for i in range(1, len(resid)):
        if sign[i-1] < 0 and sign[i] >= 0:
            # interpolate between i-1 and i
            x0, x1 = eps[i-1], eps[i]
            y0, y1 = resid[i-1], resid[i]
            if (y1 - y0) == 0:
                eps_y = x0
            else:
                eps_y = x0 - y0 * (x1 - x0) / (y1 - y0)
            sig_y = np.interp(eps_y, eps, sig)
            return float(sig_y), float(eps_y)
    return np.nan, np.nan


def toughness(eps: np.ndarray, sig: np.ndarray, eps_end: float = None) -> float:
    """Integrate area under σ–ε up to eps_end (or last point). Returns MJ/m^3."""
    if eps_end is None:
        area = np.trapz(sig, eps)  # MPa * strain = MPa
    else:
        emax = eps_end
        mask = eps <= emax
        area = np.trapz(sig[mask], eps[mask])
    # 1 MPa = 1e6 N/m^2, multiply by strain (dimensionless) => J/m^3; divide by 1e6 => MJ/m^3
    return float(area / 1e0) / 1e0  # MPa*strain ~ MJ/m^3 numerically (since 1 MPa = 1 N/mm^2)

# ------------------------------- Sidebar --------------------------------- #
st.sidebar.header("Entrada")
mode = st.sidebar.radio("Tipo de dados", ["Stress–Strain", "Force–Displacement + Geometria"], index=0)
files = st.sidebar.file_uploader("Arquivos CSV (1 ou mais)", type=["csv", "txt"], accept_multiple_files=True)

if not files:
    st.info("Envie 1+ arquivos CSV para começar. Cada arquivo será tratado como uma amostra.")
    st.stop()

# Column mapping & units
st.sidebar.subheader("Mapeamento de colunas & Unidades")
cols = {}
units = {}
geom = {}

if mode == "Stress–Strain":
    cols['strain'] = st.sidebar.text_input("Nome da coluna de deformação", value="strain")
    cols['stress'] = st.sidebar.text_input("Nome da coluna de tensão", value="stress")
    units['stress'] = st.sidebar.selectbox("Unidade de tensão", ["MPa", "Pa"], index=0)
    units['strain_in_percent'] = st.sidebar.checkbox("Deformação em % (dividir por 100)", value=False)
else:
    cols['force'] = st.sidebar.text_input("Nome da coluna de força", value="force")
    cols['disp'] = st.sidebar.text_input("Nome da coluna de deslocamento", value="displacement")
    units['force'] = st.sidebar.selectbox("Unidade de força", ["N", "kN"], index=0)
    units['disp'] = st.sidebar.selectbox("Unidade de deslocamento", ["mm", "m"], index=0)
    geom['type'] = st.sidebar.selectbox("Geometria da seção", ["rect", "cyl"], format_func=lambda x: "Retangular" if x=="rect" else "Cilíndrica")
    if geom['type'] == 'rect':
        geom['w_mm'] = st.sidebar.number_input("Largura (mm)", value=10.0, min_value=0.001)
        geom['t_mm'] = st.sidebar.number_input("Espessura (mm)", value=2.0, min_value=0.001)
    else:
        geom['d_mm'] = st.sidebar.number_input("Diâmetro (mm)", value=10.0, min_value=0.001)
    geom['L0_mm'] = st.sidebar.number_input("Comprimento útil L0 (mm)", value=50.0, min_value=0.001)

st.sidebar.subheader("Pré-processamento")
smooth = st.sidebar.checkbox("Suavizar (Savitzky–Golay)", value=False)
sg_window = st.sidebar.slider("Janela SG (pontos, ímpar)", 5, 201, 21, step=2)
make_positive = st.sidebar.checkbox("Forçar tensões/strains positivos (útil p/ compressão)", value=True)

st.sidebar.subheader("Módulo de Elasticidade (E)")
calc_mode = st.sidebar.radio("Como calcular E?", ["Janela manual", "Auto (buscar região linear)"])
max_eps_search = st.sidebar.number_input("Auto: ε máximo para busca", value=0.02, min_value=0.001, step=0.001)
win_auto = st.sidebar.number_input("Auto: largura da janela", value=0.004, min_value=0.001, step=0.001)
step_auto = st.sidebar.number_input("Auto: passo da janela", value=0.001, min_value=0.0005, step=0.0005)

st.sidebar.subheader("Limites de plotagem")
eps_crop = st.sidebar.slider("Limite de ε no gráfico", 0.0, 1.0, (0.0, 1.0))

st.sidebar.subheader("Prova por offset (escoamento)")
use_offset = st.sidebar.checkbox("Calcular escoamento por 0,2% offset", value=True)
offset_val = st.sidebar.number_input("Offset (ε)", value=0.002, min_value=0.0001, step=0.0001, format="%.4f")

# --------------------------- Processing & Plots --------------------------- #
st.title("🔧 Propriedades Mecânicas – Tração/Compressão")
st.caption("Cálculo de E (janela manual/auto), escoamento 0,2% offset, UTS, deformação máxima e tenacidade.")

# Manual window will be chosen after we load first file to get range
manual_window = [0.0005, 0.005]

all_curves = []
metrics: List[Metrics] = []

# First pass to derive strain range for manual control
first_df = None
for f in files:
    try:
        raw = pd.read_csv(f)
    except Exception:
        continue
    df = to_engineering_stress_strain(raw, 'stress_strain' if mode=="Stress–Strain" else 'force_disp', cols, geom, units, make_positive)
    if smooth:
        df['stress_MPa'] = smooth_series(df['stress_MPa'].to_numpy(), True, sg_window)
    if first_df is None and len(df) > 5:
        first_df = df.copy()

if first_df is not None:
    manual_window = [float(np.percentile(first_df['strain'], 0.5)), float(np.percentile(first_df['strain'], 5.0))]

if calc_mode == "Janela manual":
    st.info("Selecione a janela de ε para o cálculo de E (trecho linear inicial).")
    eps_min, eps_max = st.slider("Janela para E (ε)", 0.0, float(max(0.05, manual_window[1]*2)), (float(manual_window[0]), float(manual_window[1])), step=0.0005)
else:
    eps_min, eps_max = (np.nan, np.nan)

# Plot figure
fig = go.Figure()

for f in files:
    name = f.name
    try:
        raw = pd.read_csv(f)
    except Exception as e:
        st.warning(f"Falha ao ler {name}: {e}")
        continue

    df = to_engineering_stress_strain(raw, 'stress_strain' if mode=="Stress–Strain" else 'force_disp', cols, geom, units, make_positive)
    if df.empty:
        st.warning(f"Dados vazios após processamento: {name}")
        continue

    # crop for plot
    mask_plot = (df['strain'] >= eps_crop[0]) & (df['strain'] <= eps_crop[1])
    dfp = df.loc[mask_plot]

    # optional smoothing for metrics
    if smooth:
        dfp = dfp.copy()
        dfp['stress_MPa'] = smooth_series(dfp['stress_MPa'].to_numpy(), True, sg_window)

    eps = dfp['strain'].to_numpy()
    sig = dfp['stress_MPa'].to_numpy()

    # E determination
    if calc_mode == "Janela manual":
        idx = (eps >= eps_min) & (eps <= eps_max)
        E_GPa, R2 = (np.nan, np.nan)
        if idx.sum() >= 5:
            m, b, R2 = linear_fit(eps[idx], sig[idx])
            E_GPa = m / 1000.0  # MPa/strain -> GPa
        E_window = (float(eps_min), float(eps_max))
    else:
        (a,bw), E_GPa, R2 = auto_modulus_region(eps, sig, max_eps=max_eps_search, window=win_auto, step=step_auto)
        E_window = (float(a), float(bw)) if np.isfinite(E_GPa) else (np.nan, np.nan)

    # Yield by offset
    sig_y, eps_y = (np.nan, np.nan)
    if use_offset:
        sig_y, eps_y = yield_by_offset(eps, sig, E_GPa, offset=offset_val)

    # UTS and fracture
    i_max = int(np.argmax(sig))
    uts = float(sig[i_max]); eps_uts = float(eps[i_max])
    eps_frac = float(eps[-1])

    # Toughness (area under curve up to fracture)
    tough = float(np.trapz(sig, eps))  # MPa*strain ≈ MJ/m^3

    # Append metrics
    metrics.append(Metrics(name, float(E_GPa), E_window, float(R2) if np.isfinite(R2) else np.nan,
                           float(offset_val) if use_offset else np.nan,
                           float(sig_y) if np.isfinite(sig_y) else np.nan,
                           float(eps_y) if np.isfinite(eps_y) else np.nan,
                           float(uts), float(eps_uts), float(eps_frac), float(tough)))

    # Plot curve
    fig.add_trace(go.Scatter(x=eps, y=sig, mode='lines', name=name, line=dict(width=2)))

    # Plot modulus window and line
    if np.isfinite(E_window[0]) and np.isfinite(E_window[1]) and np.isfinite(E_GPa):
        e0, e1 = E_window
        # line anchored at 0: σ = E*ε (approx)
        e_line = np.linspace(e0, e1, 20)
        sig_line = (E_GPa*1000.0) * e_line
        fig.add_trace(go.Scatter(x=e_line, y=sig_line, mode='lines', name=f"E {name} ({E_GPa:.2f} GPa)", line=dict(dash='dash')))
        fig.add_vrect(x0=e0, x1=e1, fillcolor="LightSkyBlue", opacity=0.15, line_width=0)

    # Plot yield point
    if np.isfinite(eps_y) and np.isfinite(sig_y):
        fig.add_trace(go.Scatter(x=[eps_y], y=[sig_y], mode='markers', name=f"Yield {name}", marker=dict(size=9, symbol='x')))
        # offset line for visualization
        E = E_GPa * 1000.0
        ee = np.linspace(eps_y - (E_window[1]-E_window[0] if np.isfinite(E_window[0]) else 0.01), eps_y + 0.02, 20)
        sig_off = E * (ee - offset_val)
        fig.add_trace(go.Scatter(x=ee, y=sig_off, mode='lines', name=f"Offset {name}", line=dict(dash='dot')))

    # Mark UTS and fracture
    fig.add_trace(go.Scatter(x=[eps_uts], y=[uts], mode='markers', name=f"UTS {name}", marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[eps_frac], y=[sig[-1]], mode='markers', name=f"Fratura {name}", marker=dict(size=8)))

fig.update_layout(template='plotly_dark', height=600,
                  xaxis_title='Deformação, ε (–)', yaxis_title='Tensão, σ (MPa)', legend_title='Amostras')
st.plotly_chart(fig, use_container_width=True)

# ------------------------------- Results --------------------------------- #
if metrics:
    rows = []
    for m in metrics:
        rows.append({
            'Amostra': m.sample,
            'E (GPa)': m.E_GPa,
            'ε_min(E)': m.E_window[0],
            'ε_max(E)': m.E_window[1],
            'R² (E)': m.E_R2,
            'Offset ε': m.yield_offset,
            'σ_y (MPa)': m.sig_y_MPa,
            'ε_y': m.eps_y,
            'UTS (MPa)': m.uts_MPa,
            'ε_UTS': m.eps_uts,
            'ε_fratura': m.eps_fracture,
            'Tenacidade (MJ/m³)': m.toughness_MJm3,
        })
    tbl = pd.DataFrame(rows)
    st.subheader("Resultados")
    st.dataframe(tbl, use_container_width=True)

    # Downloads
    buf = io.StringIO(); tbl.to_csv(buf, index=False)
    st.download_button("⬇️ Baixar métricas (CSV)", buf.getvalue(), file_name="mechanical_properties_results.csv", mime="text/csv")

    # Also export combined curves for audit
    curves = []
    for f in files:
        try:
            raw = pd.read_csv(f)
        except Exception:
            continue
        df = to_engineering_stress_strain(raw, 'stress_strain' if mode=="Stress–Strain" else 'force_disp', cols, geom, units, make_positive)
        if smooth:
            df = df.copy(); df['stress_MPa'] = smooth_series(df['stress_MPa'].to_numpy(), True, sg_window)
        curves.append(df.assign(sample=f.name))
    if curves:
        comb = pd.concat(curves, ignore_index=True)
        cbuf = io.StringIO(); comb.to_csv(cbuf, index=False)
        st.download_button("⬇️ Baixar curvas processadas (CSV)", cbuf.getvalue(), file_name="mechanical_curves_processed.csv", mime="text/csv")

# ------------------------------- Help ------------------------------------ #
with st.expander("Notas e Boas Práticas"):
    st.markdown(
        """
        - **Entrada:**
          - *Stress–Strain*: forneça `strain` (– ou %) e `stress` (MPa/Pa).
          - *Force–Displacement*: forneça `force` (N/kN) e `displacement` (mm/m) + geometria e L0.
        - **Módulo (E):**
          - *Janela manual*: escolha uma faixa de ε linear (ex.: 0,0005–0,005).
          - *Auto*: o app varre janelas até `ε_max` e seleciona a de maior R².
        - **Yield 0,2% offset:** usa o E estimado e encontra a interseção de σ(ε) com σ = E(ε − 0,002).
        - **Tenacidade:** integral de σ–ε até a fratura, em unidades **MPa·ε ≈ MJ/m³**.
        - Em compressão, habilite "forçar positivos" para facilitar a leitura e mantenha consistência de unidades.
        """
    )
