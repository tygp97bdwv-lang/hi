import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from config import FIGHTERS_CSV, MODEL_PATH

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    background-color: #080808 !important;
    color: #e8e8e8;
    font-family: 'Rajdhani', sans-serif;
}
.stApp { background-color: #080808; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Title */
.ufc-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    text-align: center;
    letter-spacing: 6px;
    background: linear-gradient(90deg, #d4af37, #fff, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    margin-bottom: 0;
    line-height: 1;
}
.ufc-subtitle {
    text-align: center;
    color: #888;
    letter-spacing: 4px;
    font-size: 0.85rem;
    margin-top: 4px;
    text-transform: uppercase;
}

/* VS divider */
.vs-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: #d4af37;
    text-align: center;
    text-shadow: 0 0 20px #d4af3766;
    line-height: 1;
}

/* Corner labels */
.corner-red {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: #e74c3c;
    letter-spacing: 4px;
    text-align: center;
    text-transform: uppercase;
}
.corner-blue {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: #3498db;
    letter-spacing: 4px;
    text-align: center;
    text-transform: uppercase;
}

/* Fighter card */
.fighter-card {
    border-radius: 12px;
    padding: 24px 20px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.fighter-card-red {
    background: linear-gradient(145deg, #1a0808 0%, #120505 60%, #0a0a0a 100%);
    border: 1px solid #e74c3c44;
    box-shadow: 0 0 30px #e74c3c22, inset 0 0 60px #e74c3c08;
}
.fighter-card-blue {
    background: linear-gradient(145deg, #081018 0%, #050c12 60%, #0a0a0a 100%);
    border: 1px solid #3498db44;
    box-shadow: 0 0 30px #3498db22, inset 0 0 60px #3498db08;
}
.fighter-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    letter-spacing: 3px;
    line-height: 1.1;
    margin: 10px 0 4px;
}
.fighter-name-red  { color: #ff6b6b; text-shadow: 0 0 20px #e74c3c66; }
.fighter-name-blue { color: #5dade2; text-shadow: 0 0 20px #3498db66; }

.fighter-record {
    font-size: 1rem;
    color: #aaa;
    letter-spacing: 2px;
}

/* Silhouette */
.silhouette {
    font-size: 6rem;
    line-height: 1;
    margin: 8px 0;
    filter: drop-shadow(0 0 12px currentColor);
}

/* Stat pill */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #1a1a1a;
    font-size: 0.9rem;
}
.stat-label { color: #888; font-size: 0.8rem; letter-spacing: 1px; text-transform: uppercase; }
.stat-val-red  { color: #ff6b6b; font-weight: 700; font-size: 1.1rem; }
.stat-val-blue { color: #5dade2; font-weight: 700; font-size: 1.1rem; }

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #d4af37, #b8860b) !important;
    color: #000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.6rem !important;
    letter-spacing: 4px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px !important;
    cursor: pointer !important;
    box-shadow: 0 0 20px #d4af3744 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 40px #d4af3799 !important;
    transform: translateY(-2px) !important;
}

/* Result banner */
.winner-banner {
    text-align: center;
    padding: 28px;
    border-radius: 16px;
    margin: 20px 0;
    background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
    border: 1px solid #d4af3744;
    box-shadow: 0 0 40px #d4af3722;
}
.winner-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: #888;
    letter-spacing: 5px;
}
.winner-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    letter-spacing: 4px;
    background: linear-gradient(90deg, #d4af37, #fff 50%, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.winner-conf {
    font-size: 0.9rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* HP bar container */
.hp-wrapper {
    margin: 6px 0 16px;
}
.hp-label {
    font-size: 0.75rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.hp-bar-bg {
    background: #1a1a1a;
    border-radius: 4px;
    height: 22px;
    overflow: hidden;
    display: flex;
    border: 1px solid #2a2a2a;
}
.hp-seg-red  { background: linear-gradient(90deg, #c0392b, #e74c3c); height: 100%; transition: width 0.8s; }
.hp-seg-blue { background: linear-gradient(90deg, #2980b9, #3498db); height: 100%; transition: width 0.8s; }
.hp-pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 1px;
}

/* Section header */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 4px;
    color: #d4af37;
    border-bottom: 1px solid #d4af3733;
    padding-bottom: 6px;
    margin: 24px 0 14px;
}

/* Flag pills */
.flag-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    letter-spacing: 1px;
    margin: 4px 4px 4px 0;
    font-weight: 600;
}
.flag-danger  { background: #3d0a0a; color: #ff6b6b; border: 1px solid #e74c3c55; }
.flag-warning { background: #3d2a00; color: #f39c12; border: 1px solid #f39c1255; }
.flag-ok      { background: #0a1f0a; color: #2ecc71; border: 1px solid #2ecc7155; }

/* Dropdown */
.stSelectbox > div > div {
    background-color: #111 !important;
    border: 1px solid #333 !important;
    color: #fff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_fighters() -> pd.DataFrame:
    if not os.path.exists(FIGHTERS_CSV):
        return pd.DataFrame()
    df = pd.read_csv(FIGHTERS_CSV)
    return df[df["name"].notna()].sort_values("name").reset_index(drop=True)


def model_ready() -> bool:
    return os.path.exists(MODEL_PATH)


def _fmt(val, pct=False, decimals=1):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if pct:
        return f"{val*100:.0f}%"
    return f"{val:.{decimals}f}"


def radar_chart(f1: pd.Series, f2: pd.Series, name1: str, name2: str):
    cats = ["Striking\nOutput", "Accuracy", "Defense", "Takedowns", "Grappling", "Durability"]

    def norm(val, lo, hi):
        if val is None or pd.isna(val):
            return 0.5
        return max(0, min(1, (val - lo) / (hi - lo)))

    def scores(f):
        return [
            norm(f.get("slpm"), 1, 10),
            norm(f.get("str_acc"), 0.25, 0.75),
            norm(f.get("str_def"), 0.30, 0.85),
            norm(f.get("td_avg"), 0, 7),
            norm(f.get("sub_avg"), 0, 4),
            norm(1 / max(f.get("sapm", 4), 0.1), 0.1, 1),  # lower sapm = more durable
        ]

    s1, s2 = scores(f1), scores(f2)
    cats_closed = cats + [cats[0]]
    s1_closed   = s1  + [s1[0]]
    s2_closed   = s2  + [s2[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=s1_closed, theta=cats_closed, fill="toself",
        name=name1, line=dict(color="#e74c3c", width=2),
        fillcolor="rgba(231,76,60,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=s2_closed, theta=cats_closed, fill="toself",
        name=name2, line=dict(color="#3498db", width=2),
        fillcolor="rgba(52,152,219,0.15)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0f0f0f",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor="#2a2a2a", linecolor="#2a2a2a"),
            angularaxis=dict(gridcolor="#2a2a2a", linecolor="#333",
                             tickfont=dict(color="#ccc", size=11, family="Rajdhani")),
        ),
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font=dict(color="#ccc", family="Rajdhani"),
        legend=dict(font=dict(color="#ccc", size=12), bgcolor="#0f0f0f"),
        margin=dict(l=40, r=40, t=30, b=30),
        height=340,
    )
    return fig


def bar_chart(f1: pd.Series, f2: pd.Series, name1: str, name2: str):
    stats = [
        ("SLpM",    f1.get("slpm",    0), f2.get("slpm",    0)),
        ("Str Acc", (f1.get("str_acc", 0) or 0)*100, (f2.get("str_acc", 0) or 0)*100),
        ("Str Def", (f1.get("str_def", 0) or 0)*100, (f2.get("str_def", 0) or 0)*100),
        ("TD Avg",  f1.get("td_avg",  0), f2.get("td_avg",  0)),
        ("TD Def",  (f1.get("td_def", 0) or 0)*100, (f2.get("td_def", 0) or 0)*100),
        ("Sub Avg", f1.get("sub_avg", 0), f2.get("sub_avg", 0)),
    ]
    labels = [s[0] for s in stats]
    v1     = [s[1] for s in stats]
    v2     = [s[2] for s in stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=name1, x=labels, y=v1, marker_color="#e74c3c", opacity=0.85))
    fig.add_trace(go.Bar(name=name2, x=labels, y=v2, marker_color="#3498db", opacity=0.85))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font=dict(color="#ccc", family="Rajdhani", size=12),
        legend=dict(bgcolor="#0f0f0f", font=dict(color="#ccc")),
        xaxis=dict(gridcolor="#1a1a1a", linecolor="#333"),
        yaxis=dict(gridcolor="#1a1a1a", linecolor="#333"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
    )
    return fig


def hp_bar_html(pct_red: float, pct_blue: float, name1: str, name2: str) -> str:
    r = int(pct_red)
    b = int(pct_blue)
    return f"""
<div class="hp-wrapper">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <span class="hp-pct" style="color:#ff6b6b;">{name1.split()[-1].upper()}  {r}%</span>
    <span class="hp-label" style="align-self:center;">WIN PROBABILITY</span>
    <span class="hp-pct" style="color:#5dade2;">{b}%  {name2.split()[-1].upper()}</span>
  </div>
  <div class="hp-bar-bg">
    <div class="hp-seg-red"  style="width:{r}%;"></div>
    <div class="hp-seg-blue" style="width:{b}%;"></div>
  </div>
</div>"""


def fighter_card_html(f: pd.Series, color: str) -> str:
    card_cls = f"fighter-card fighter-card-{color}"
    name_cls = f"fighter-name fighter-name-{color}"
    val_cls  = f"stat-val-{color}"
    name = f.get("name", "Unknown")
    w, l, d = int(f.get("wins", 0) or 0), int(f.get("losses", 0) or 0), int(f.get("draws", 0) or 0)
    emoji = "🥊"
    reach = _fmt(f.get("reach_cm"))
    height = _fmt(f.get("height_cm"))
    stance = f.get("stance") or "—"
    age = _fmt(f.get("age"), decimals=0)
    slpm = _fmt(f.get("slpm"))
    acc  = _fmt(f.get("str_acc"), pct=True)
    def_ = _fmt(f.get("str_def"), pct=True)
    td   = _fmt(f.get("td_avg"))
    sub_ = _fmt(f.get("sub_avg"))
    streak = int(f.get("win_streak", 0) or 0)
    streak_html = f'<span style="color:#2ecc71">{"▲ " * min(streak,5)} {streak}W streak</span>' if streak > 0 else ""

    return f"""
<div class="{card_cls}">
  <div class="corner-{color}">{"🔴 RED CORNER" if color=="red" else "🔵 BLUE CORNER"}</div>
  <div class="silhouette">{emoji}</div>
  <div class="{name_cls}">{name}</div>
  <div class="fighter-record">{w} — {l} — {d}</div>
  <div style="font-size:0.8rem;color:#666;margin:2px 0 10px;letter-spacing:1px;">
    {stance} &nbsp;|&nbsp; {height}cm &nbsp;|&nbsp; Reach {reach}cm &nbsp;|&nbsp; Age {age}
  </div>
  {f'<div style="margin-bottom:8px;font-size:0.85rem;">{streak_html}</div>' if streak_html else ""}
  <div class="stat-row"><span class="stat-label">Strikes/min</span><span class="{val_cls}">{slpm}</span></div>
  <div class="stat-row"><span class="stat-label">Str Accuracy</span><span class="{val_cls}">{acc}</span></div>
  <div class="stat-row"><span class="stat-label">Str Defense</span><span class="{val_cls}">{def_}</span></div>
  <div class="stat-row"><span class="stat-label">TD Average</span><span class="{val_cls}">{td}</span></div>
  <div class="stat-row"><span class="stat-label">Sub Average</span><span class="{val_cls}">{sub_}</span></div>
</div>"""


# ── Main app ───────────────────────────────────────────────────────────────────

st.markdown('<div class="ufc-title">🥊 UFC FIGHT PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="ufc-subtitle">AI-Powered Matchup Analysis</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Setup check ───────────────────────────────────────────────────────────────
fighters_df = load_fighters()

if fighters_df.empty:
    st.error("⚠️  No fighter data found. Open CMD, go to your `hi` folder, and run:  `python demo.py`")
    st.stop()

if not model_ready():
    st.error("⚠️  Model not trained yet. Open CMD, go to your `hi` folder, and run:  `python demo.py`")
    st.stop()

fighter_names = fighters_df["name"].tolist()

# ── Fighter selectors ─────────────────────────────────────────────────────────
col_l, col_vs, col_r = st.columns([5, 1, 5])

with col_l:
    st.markdown('<div class="corner-red">🔴  RED CORNER</div>', unsafe_allow_html=True)
    sel_a = st.selectbox("Fighter A", fighter_names,
                         index=fighter_names.index("Ilia Topuria") if "Ilia Topuria" in fighter_names else 0,
                         label_visibility="collapsed", key="fa")

with col_vs:
    st.markdown('<div class="vs-badge" style="margin-top:28px">VS</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="corner-blue">🔵  BLUE CORNER</div>', unsafe_allow_html=True)
    default_b = "Arman Tsarukyan" if "Arman Tsarukyan" in fighter_names else (fighter_names[1] if len(fighter_names) > 1 else fighter_names[0])
    sel_b = st.selectbox("Fighter B", fighter_names,
                         index=fighter_names.index(default_b),
                         label_visibility="collapsed", key="fb")

st.markdown("<br>", unsafe_allow_html=True)

# ── Fighter cards ─────────────────────────────────────────────────────────────
f1 = fighters_df[fighters_df["name"] == sel_a].iloc[0]
f2 = fighters_df[fighters_df["name"] == sel_b].iloc[0]

col_card_l, col_card_r = st.columns(2)
with col_card_l:
    st.markdown(fighter_card_html(f1, "red"), unsafe_allow_html=True)
with col_card_r:
    st.markdown(fighter_card_html(f2, "blue"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ─────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    predict_clicked = st.button("⚡  PREDICT FIGHT OUTCOME")

# ── Results ────────────────────────────────────────────────────────────────────
if predict_clicked:
    if sel_a == sel_b:
        st.warning("Select two different fighters.")
        st.stop()

    with st.spinner("Analysing matchup..."):
        from model.predict import predict_matchup
        result = predict_matchup(sel_a, sel_b, fighters_df=fighters_df, fetch_news=False)

    prob_a = result["prob_a_wins"] * 100
    prob_b = result["prob_b_wins"] * 100
    winner = result["predicted_winner"]
    conf   = result["confidence"]
    conf_color = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}.get(conf, "#ccc")

    # Winner banner
    st.markdown(f"""
    <div class="winner-banner">
      <div class="winner-label">PREDICTED WINNER</div>
      <div class="winner-name">{winner}</div>
      <div class="winner-conf" style="color:{conf_color};">
        {"🟢" if conf=="High" else "🟡" if conf=="Medium" else "🔴"} {conf} Confidence
      </div>
    </div>""", unsafe_allow_html=True)

    # HP bar
    st.markdown(hp_bar_html(prob_a, prob_b, sel_a, sel_b), unsafe_allow_html=True)

    st.markdown('<div class="section-header">STAT BREAKDOWN</div>', unsafe_allow_html=True)
    chart_l, chart_r = st.columns(2)
    with chart_l:
        st.markdown("**Radar comparison**", help="Normalised 0-1 scale per category")
        st.plotly_chart(radar_chart(f1, f2, sel_a, sel_b), use_container_width=True, config={"displayModeBar": False})
    with chart_r:
        st.markdown("**Key stats head-to-head**")
        st.plotly_chart(bar_chart(f1, f2, sel_a, sel_b), use_container_width=True, config={"displayModeBar": False})

    # Detailed stat table
    st.markdown('<div class="section-header">DETAILED STATS</div>', unsafe_allow_html=True)
    feats = result.get("features", {})

    def diff_color(val):
        if val > 0:   return f'<span style="color:#2ecc71">▲ +{val:.2f}</span>'
        if val < 0:   return f'<span style="color:#e74c3c">▼ {val:.2f}</span>'
        return '<span style="color:#888">= 0</span>'

    rows = [
        ("Strikes / min",     _fmt(f1.get("slpm")),               _fmt(f2.get("slpm")),               feats.get("slpm_diff", 0)),
        ("Strike Accuracy",   _fmt(f1.get("str_acc"), pct=True),  _fmt(f2.get("str_acc"), pct=True),  feats.get("str_acc_diff", 0)),
        ("Strike Defense",    _fmt(f1.get("str_def"), pct=True),  _fmt(f2.get("str_def"), pct=True),  feats.get("str_def_diff", 0)),
        ("Strikes Absorbed",  _fmt(f1.get("sapm")),               _fmt(f2.get("sapm")),               feats.get("sapm_diff", 0)),
        ("TD / 15 min",       _fmt(f1.get("td_avg")),             _fmt(f2.get("td_avg")),             feats.get("td_avg_diff", 0)),
        ("TD Accuracy",       _fmt(f1.get("td_acc"), pct=True),   _fmt(f2.get("td_acc"), pct=True),   feats.get("td_acc_diff", 0)),
        ("TD Defense",        _fmt(f1.get("td_def"), pct=True),   _fmt(f2.get("td_def"), pct=True),   feats.get("td_def_diff", 0)),
        ("Sub Attempts / 15", _fmt(f1.get("sub_avg")),            _fmt(f2.get("sub_avg")),            feats.get("sub_avg_diff", 0)),
        ("Reach (cm)",        _fmt(f1.get("reach_cm")),           _fmt(f2.get("reach_cm")),           feats.get("reach_cm_diff", 0)),
        ("Age",               _fmt(f1.get("age"), decimals=0),    _fmt(f2.get("age"), decimals=0),    feats.get("age_diff", 0)),
        ("Win Streak",        str(int(f1.get("win_streak", 0) or 0)), str(int(f2.get("win_streak", 0) or 0)), feats.get("win_streak_diff", 0)),
    ]

    header = f"""<table style="width:100%;border-collapse:collapse;font-family:Rajdhani,sans-serif;font-size:0.95rem;">
<tr style="border-bottom:1px solid #d4af3744;">
  <th style="text-align:left;padding:8px;color:#d4af37;letter-spacing:2px;text-transform:uppercase;font-size:0.78rem;">Stat</th>
  <th style="text-align:center;padding:8px;color:#e74c3c;letter-spacing:1px;">{sel_a}</th>
  <th style="text-align:center;padding:8px;color:#3498db;letter-spacing:1px;">{sel_b}</th>
  <th style="text-align:center;padding:8px;color:#888;font-size:0.78rem;letter-spacing:1px;">EDGE</th>
</tr>"""
    body = ""
    for i, (label, v1, v2, diff) in enumerate(rows):
        bg = "#0f0f0f" if i % 2 == 0 else "#111"
        body += f"""<tr style="background:{bg};">
  <td style="padding:8px;color:#aaa;text-transform:uppercase;font-size:0.8rem;letter-spacing:1px;">{label}</td>
  <td style="text-align:center;padding:8px;color:#ff6b6b;font-weight:700;">{v1}</td>
  <td style="text-align:center;padding:8px;color:#5dade2;font-weight:700;">{v2}</td>
  <td style="text-align:center;padding:8px;">{diff_color(diff)}</td>
</tr>"""
    st.markdown(header + body + "</table>", unsafe_allow_html=True)

    # Risk flags
    na = result.get("news_a", {})
    nb = result.get("news_b", {})
    flags = []
    if na.get("injury_flag"):
        flags.append(f'<span class="flag-pill flag-danger">⚠ {sel_a} — Injury News</span>')
    if nb.get("injury_flag"):
        flags.append(f'<span class="flag-pill flag-danger">⚠ {sel_b} — Injury News</span>')
    if na.get("weight_issue_flag"):
        flags.append(f'<span class="flag-pill flag-warning">⚖ {sel_a} — Weight Concerns</span>')
    if nb.get("weight_issue_flag"):
        flags.append(f'<span class="flag-pill flag-warning">⚖ {sel_b} — Weight Concerns</span>')
    if not flags:
        flags.append('<span class="flag-pill flag-ok">✓ No risk flags detected</span>')

    st.markdown('<div class="section-header">RISK FLAGS</div>', unsafe_allow_html=True)
    st.markdown(" ".join(flags), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
