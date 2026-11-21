import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── 데이터 로딩 & 표준화 ────────────────────────────────────────────────────

@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순서로 읽고, 둘 다 실패하면 빈 DataFrame 반환."""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    """열 이름을 표준 이름으로 통일하고, 숫자 열은 숫자로 변환."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.extract(
                    r"(-?\d+\.?\d*)", expand=False
                ),
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed
    return df


def filter_growth(g_df, active, w_min, w_max, sh_min, sh_max, r_min, r_max):
    """사이드바 히스토그램 범위로 생육 데이터를 필터링."""
    if not active or g_df.empty:
        return g_df
    w_ok = pd.Series(True, index=g_df.index)
    sh_ok = pd.Series(True, index=g_df.index)
    r_ok = pd.Series(True, index=g_df.index)
    if w_min is not None and w_max is not None and "생중량" in g_df.columns:
        w = pd.to_numeric(g_df["생중량"], errors="coerce")
        w_ok = w.notna() & (w >= w_min) & (w <= w_max)
    if sh_min is not None and sh_max is not None and "지상부 길이" in g_df.columns:
        sh = pd.to_numeric(g_df["지상부 길이"], errors="coerce")
        sh_ok = sh.notna() & (sh >= sh_min) & (sh <= sh_max)
    if r_min is not None and r_max is not None and "지하부 길이" in g_df.columns:
        r = pd.to_numeric(g_df["지하부 길이"], errors="coerce")
        r_ok = r.notna() & (r >= r_min) & (r <= r_max)
    mask = w_ok & sh_ok & r_ok
    return g_df[mask] if mask.any() else g_df.iloc[:0]


def fmt_id(series):
    """개체번호를 원래 모습 그대로(정수면 정수, 문자면 문자) 표시."""
    if pd.api.types.is_numeric_dtype(series):
        return series.apply(
            lambda x: int(x) if pd.notna(x) and float(x).is_integer() else x
        )
    return series


def csv_bytes(df):
    """한글 깨짐 없이 다운로드용 바이트 반환."""
    return df.to_csv(index=False).encode("utf-8-sig")


def hist_fig(values, title, color, vmin, vmax):
    """사이드바용 컴팩트 히스토그램 + 범위 음영."""
    fig = go.Figure()
    if len(values) < 2:
        fig.add_annotation(text="데이터 부족", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, family="Malgun Gothic"))
        fig.update_layout(margin=dict(l=20, r=10, t=30, b=20),
                          font=dict(family="Malgun Gothic, sans-serif"),
                          height=140)
        return fig
    fig.add_trace(
        go.Histogram(x=values, nbinsx=12, marker_color=color, opacity=0.85, name="개체"),
    )
    if vmin is not None and vmax is not None and vmin < vmax:
        fig.add_vrect(x0=vmin, x1=vmax, fillcolor="orange", opacity=0.2,
                      line_width=0, layer="below")
    fig.update_layout(
        barmode="overlay",
        bargap=0.1,
        margin=dict(l=20, r=10, t=30, b=20),
        xaxis=dict(showticklabels=False,
                   title=title,
                   title_font=dict(size=10, family="Malgun Gothic, sans-serif")),
        yaxis=dict(showticklabels=False, title="개체 수",
                   title_font=dict(size=10, family="Malgun Gothic, sans-serif")),
        font=dict(family="Malgun Gothic, sans-serif"),
        showlegend=False,
        height=140,
    )
    return fig


# ── Streamlit 앱 ────────────────────────────────────────────────────────────

st.set_page_config(page_title="극지식물 생육 분석", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌱 극지식물, 똑같이 자랐을까?")
st.caption(
    "고등학생이 이해할 수 있도록 통계 용어(평균·표준편차·상관 등)에는 짧은 설명을 붙였습니다."
)

# ── 데이터 로딩 (사이드바보다 먼저, growth_df가 정의되도록) ────────────────

with st.spinner("CSV를 읽는 중..."):
    env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
    growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")
    growth_metrics = st.sidebar.multiselect(
        "📊 생육 측정 항목 선택 (Tab 3 막대그래프)",
        options=["잎 수", "지상부 길이", "지하부 길이", "생중량"],
        default=["생중량", "잎 수", "지상부 길이", "지하부 길이"],
    )

    env_df = read_csv(env_file) if env_file is not None else pd.DataFrame()
    growth_df = read_csv(growth_file) if growth_file is not None else pd.DataFrame()

env_df = std_cols(env_df) if not env_df.empty else env_df
growth_df = std_cols(growth_df) if not growth_df.empty else growth_df

school_name = ""
if env_file is not None:
    school_name = str(env_file.name).split("_")[0]
if not school_name and growth_file is not None:
    school_name = str(growth_file.name).split("_")[0]

if env_df.empty:
    st.sidebar.info("환경 CSV를 업로드하면 환경 탭이 표시됩니다.")
if growth_df.empty:
    st.sidebar.info("생육 CSV를 업로드하면 생육 탭이 표시됩니다.")

# ── 사이드바: 히스토그램 3종 + 범위 필터 (growth_df가 정의된 후 사용) ──────

if not growth_df.empty and "개체번호" in growth_df.columns:
    w_raw = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna()
    s_raw = pd.to_numeric(growth_df["지상부 길이"], errors="coerce").dropna()
    r_raw = pd.to_numeric(growth_df["지하부 길이"], errors="coerce").dropna()
else:
    w_raw = pd.Series(dtype=float)
    s_raw = pd.Series(dtype=float)
    r_raw = pd.Series(dtype=float)

w_min, w_max = (float(w_raw.min()), float(w_raw.max())) if len(w_raw) >= 2 else (None, None)
s_min, s_max = (float(s_raw.min()), float(s_raw.max())) if len(s_raw) >= 2 else (None, None)
r_min, r_max = (float(r_raw.min()), float(r_raw.max())) if len(r_raw) >= 2 else (None, None)

with st.sidebar:
    st.markdown("### 📊 분포 히스토그램 + 범위 필터")
    st.caption(
        "막대 위 주황 음영 영역이 현재 필터예요. Tab 3 분석이 이 범위의 개체만 대상으로 다시 계산됩니다."
    )

    # 슬라이더는 데이터가 있을 때만 생성, 없으면 None 유지
    f_w = st.slider(
        "생중량(g) 범위",
        min_value=0.0, max_value=30.0,
        value=(w_min, w_max) if w_min is not None else (0.0, 30.0),
        step=0.01,
        key="slider_w",
    ) if w_min is not None else None

    f_s = st.slider(
        "지상부 길이(mm) 범위",
        min_value=0.0, max_value=300.0,
        value=(s_min, s_max) if s_min is not None else (0.0, 300.0),
        step=0.1,
        key="slider_s",
    ) if s_min is not None else None

    f_r = st.slider(
        "지하부 길이(mm) 범위",
        min_value=0.0, max_value=400.0,
        value=(r_min, r_max) if r_min is not None else (0.0, 400.0),
        step=0.1,
        key="slider_r",
    ) if r_min is not None else None

    c1, c2, c3 = st.columns(3)
    with c1:
        vmin_w = f_w[0] if f_w is not None else w_min
        vmax_w = f_w[1] if f_w is not None else w_max
        st.plotly_chart(
            hist_fig(w_raw, "생중량(g)", "#2ecc71", vmin_w, vmax_w),
            use_container_width=True, key="hist_w",
        )
        if len(w_raw) >= 1:
            st.caption(
                f"n={len(w_raw)} · 평균 {w_raw.mean():.2f}g · "
                f"범위 {w_raw.min():.2f}~{w_raw.max():.2f}g"
            )
    with c2:
        vmin_s = f_s[0] if f_s is not None else s_min
        vmax_s = f_s[1] if f_s is not None else s_max
        st.plotly_chart(
            hist_fig(s_raw, "지상부 길이(mm)", "#3498db", vmin_s, vmax_s),
            use_container_width=True, key="hist_s",
        )
        if len(s_raw) >= 1:
            st.caption(
                f"n={len(s_raw)} · 평균 {s_raw.mean():.1f}mm · "
                f"범위 {s_raw.min():.1f}~{s_raw.max():.1f}mm"
            )
    with c3:
        vmin_r = f_r[0] if f_r is not None else r_min
        vmax_r = f_r[1] if f_r is not None else r_max
        st.plotly_chart(
            hist_fig(r_raw, "지하부 길이(mm)", "#9b59b6", vmin_r, vmax_r),
            use_container_width=True, key="hist_r",
        )
        if len(r_raw) >= 1:
            st.caption(
                f"n={len(r_raw)} · 평균 {r_raw.mean():.1f}mm · "
                f"범위 {r_raw.min():.1f}~{r_raw.max():.1f}mm"
            )

    filter_active = st.checkbox(
        "☑ 위 범위로 필터링 (Tab 3 전체 분석에 적용)",
        value=False,
    )

    growth_df_filtered = filter_growth(
        growth_df, filter_active,
        f_w[0] if f_w is not None else None,
        f_w[1] if f_w is not None else None,
        f_s[0] if f_s is not None else None,
        f_s[1] if f_s is not None else None,
        f_r[0] if f_r is not None else None,
        f_r[1] if f_r is not None else None,
    ) if not growth_df.empty else growth_df

    if not growth_df.empty:
        st.divider()
        st.caption(
            f"📌 현재 필터: {len(growth_df_filtered)}개체"
            + (" (전체 " + str(len(growth_df)) + "개체)" if filter_active else "")
        )

# ── 탭 ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    ["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — 실험 개요
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    try:
        if env_df.empty and growth_df.empty:
            st.info(
                "두 CSV를 모두 업로드하면 실험 개요, 환경 데이터, 생육 결과 분석을 볼 수 있습니다."
            )
        else:
            st.markdown("### 연구 배경 및 목적")
            st.markdown(
                """
- 극지식물은 북극·남극·고산 지대처럼 매우 춥고 열악한 환경에서도 살아남는 식물입니다.
- 이 대시보드에서는 우리 학교에 심은 극지식물 **개체마다** 생육(잎 수, 줄기 길이, 무게 등)에 어떤 차이가 있는지 살펴봅니다.
- 평균·표준편차·상관 같은 기본 통계로 개체 간 차이를 읽어내는 것이 목표입니다.
"""
            )

            st.markdown("#### 📌 통계 용어 짧은 설명")
            st.markdown(
                """
- **평균(mean)**: 여러 값을 더해 개수로 나눈 것 — 대표적인 '가운데 값'
- **표준편차(SD)**: 값들이 평균에서 얼마나 흩어져 있는지를 나타내는 숫자 — 클수록 개체 간 차이가 큼
- **변동계수(CV)**: 표준편차를 평균으로 나눈 값 — 단위가 다른 집단끼리 흩어짐 정도를 비교할 때 사용
- **상관계수(r)**: 두 값이 함께 움직이는 정도 — -1(반대로 움직임)에서 +1(함께 증가) 사이 값
"""
            )

            st.markdown("### 🏫 데이터 요약")
            col_n, col_period, col_items = st.columns(3)
            with col_n:
                if not growth_df.empty and "개체번호" in growth_df.columns:
                    st.metric("총 개체 수", f"{len(growth_df)}개체")
                else:
                    st.metric("총 개체 수", "—")
            with col_period:
                if not env_df.empty and "time" in env_df.columns:
                    times = env_df["time"].dropna()
                    if len(times) >= 1:
                        st.metric(
                            "측정 기간",
                            f"{times.min().date()} ~ {times.max().date()}",
                        )
                    else:
                        st.metric("측정 기간", "—")
                else:
                    st.metric("측정 기간", "—")
            with col_items:
                if not growth_df.empty:
                    num_cols = [
                        c for c in growth_df.columns if c not in ("개체번호", "time")
                    ]
                    st.metric(
                        "생육 측정 항목", ", ".join(num_cols) if num_cols else "—"
                    )
                else:
                    st.metric("생육 측정 항목", "—")

            st.divider()

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                if not growth_df.empty and "개체번호" in growth_df.columns:
                    st.metric("총 개체수", f"{len(growth_df)}개체")
                else:
                    st.metric("총 개체수", "CSV 필요")
            with m2:
                if not env_df.empty and "temperature" in env_df.columns:
                    t_clean = env_df["temperature"].dropna()
                    if len(t_clean) >= 1:
                        st.metric("평균 온도", f"{t_clean.mean():.1f}°C")
                    else:
                        st.metric("평균 온도", "—")
                else:
                    st.metric("평균 온도", "CSV 필요")
            with m3:
                if not env_df.empty and "humidity" in env_df.columns:
                    h_clean = env_df["humidity"].dropna()
                    if len(h_clean) >= 1:
                        st.metric("평균 습도", f"{h_clean.mean():.1f}%")
                    else:
                        st.metric("평균 습도", "—")
                else:
                    st.metric("평균 습도", "CSV 필요")
            with m4:
                if not growth_df.empty and "생중량" in growth_df.columns:
                    g_clean = growth_df["생중량"].dropna()
                    if len(g_clean) >= 1:
                        st.metric("평균 생중량", f"{g_clean.mean():.2f}g")
                    else:
                        st.metric("평균 생중량", "—")
                else:
                    st.metric("평균 생중량", "CSV 필요")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — 환경 데이터
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    try:
        if env_df.empty:
            st.info(
                "🌡️ 환경 CSV를 업로드하면 온도·습도·pH·EC 등 환경 데이터를 볼 수 있습니다."
            )
        else:
            env_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns
            ]

            st.markdown("### 환경 요약 (평균 · 최소 · 최대)")
            if env_cols:
                summary_rows = []
                unit_map = {
                    "temperature": "°C",
                    "humidity": "%",
                    "ph": "",
                    "ec": "mS/cm",
                }
                for c in env_cols:
                    s = env_df[c].dropna()
                    if len(s) >= 1:
                        unit = unit_map.get(c, "")
                        summary_rows.append(
                            {
                                "측정 항목": c,
                                "평균": f"{s.mean():.2f}{unit}",
                                "최소": f"{s.min():.2f}{unit}",
                                "최대": f"{s.max():.2f}{unit}",
                                "측정 횟수": f"{len(s)}회",
                            }
                        )
                st.dataframe(pd.DataFrame(summary_rows), width="stretch")
            else:
                st.info(
                    "환경 데이터에 온도·습도·pH·EC 열이 없어 요약표를 만들 수 없습니다."
                )

            st.markdown("### 날짜별 시계열 (일별 평균)")
            if "time" in env_df.columns and env_cols:
                daily = env_df.set_index("time")[env_cols].resample("D").mean()
                if len(daily) >= 2:
                    n_t = len(env_cols)
                    fig_ts = make_subplots(
                        rows=n_t, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    )
                    color_map = {
                        "temperature": "#e74c3c",
                        "humidity": "#3498db",
                        "ph": "#9b59b6",
                        "ec": "#f39c12",
                    }
                    for i, c in enumerate(env_cols, start=1):
                        fig_ts.add_trace(
                            go.Scatter(
                                x=daily.index,
                                y=daily[c],
                                mode="lines",
                                name=c,
                                line=dict(color=color_map.get(c, "#2c3e50"), width=2),
                                hovertemplate=f"{c}: %{{y:.2f}}<extra></extra>",
                            ),
                            row=i, col=1,
                        )
                        fig_ts.update_yaxes(
                            title_text=c, row=i, col=1,
                            title_font=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                        )
                    fig_ts.update_xaxes(
                        title_text="날짜", row=n_t, col=1,
                        title_font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                    )
                    fig_ts.update_layout(
                        height=180 * n_t,
                        margin=dict(l=40, r=20, t=30, b=40),
                        font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_ts, width="stretch")
                else:
                    st.info(
                        "날짜별 평균을 그리려면 최소 2일 이상의 데이터가 필요합니다."
                    )
            else:
                st.info("time 열이 없어 시계열을 그릴 수 없습니다.")

            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                display_cols = [
                    c for c in ["time"] + env_cols if c in env_df.columns
                ]
                if display_cols:
                    st.dataframe(env_df[display_cols], width="stretch")
                    st.download_button(
                        label="환경 데이터 CSV 다운로드",
                        data=csv_bytes(env_df[display_cols]),
                        file_name=f"{school_name or '환경'}_데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 열이 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — 생육 결과
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    try:
        if growth_df.empty:
            st.info(
                "📊 생육 결과 CSV를 업로드하면 개체별 생육 비교, 분포, 상관관계 분석을 볼 수 있습니다."
            )
        else:
            gcols = [
                c
                for c in [
                    "생중량",
                    "잎 수",
                    "지상부 길이",
                    "지하부 길이",
                    "지상부 질량",
                    "지하부 질량",
                ]
                if c in growth_df.columns
            ]

            st.markdown("### 🥇 핵심 결과")
            c1, c2, c3 = st.columns(3)
            with c1:
                if "생중량" in growth_df.columns:
                    g = growth_df["생중량"].dropna()
                    if len(g) >= 1:
                        st.metric("평균 생중량", f"{g.mean():.2f} g")
                    else:
                        st.metric("평균 생중량", "측정값 없음")
                else:
                    st.metric("평균 생중량", "열에 없음")
            with c2:
                if "생중량" in growth_df.columns:
                    g = growth_df["생중량"].dropna()
                    if len(g) >= 1:
                        idx_max = g.idxmax()
                        max_id = growth_df.loc[idx_max, "개체번호"]
                        st.metric(
                            "최대 생중량 개체",
                            f"{fmt_id(pd.Series([max_id])).iloc[0]}번 · {g.max():.2f}g",
                        )
                    else:
                        st.metric("최대 생중량 개체", "측정값 없음")
                else:
                    st.metric("최대 생중량 개체", "열에 없음")
            with c3:
                if "생중량" in growth_df.columns:
                    g = growth_df["생중량"].dropna()
                    if len(g) >= 1:
                        idx_min = g.idxmin()
                        min_id = growth_df.loc[idx_min, "개체번호"]
                        st.metric(
                            "최소 생중량 개체",
                            f"{fmt_id(pd.Series([min_id])).iloc[0]}번 · {g.min():.2f}g",
                        )
                    else:
                        st.metric("최소 생중량 개체", "측정값 없음")
                else:
                    st.metric("최소 생중량 개체", "열에 없음")

            color_map = {
                "생중량": "#2ecc71",
                "잎 수": "#e74c3c",
                "지상부 길이": "#3498db",
                "지하부 길이": "#9b59b6",
                "지상부 질량": "#f39c12",
                "지하부 질량": "#1abc9c",
            }

            def bar_chart(df, col, color):
                ids = fmt_id(df["개체번호"])
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                ids = ids.loc[vals.index]
                avg = vals.mean()
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=ids,
                        y=vals,
                        name=col,
                        marker_color=color,
                        text=[f"{v:.2f}" if pd.notna(v) else "" for v in vals],
                        textposition="outside",
                        hovertemplate=f"{col}: %{{y:.2f}}<br>개체: %{{x}}<extra></extra>",
                    )
                )
                if pd.notna(avg):
                    fig.add_hline(
                        y=avg,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text=f"평균 {avg:.2f}",
                        annotation_position="top right",
                    )
                fig.update_layout(
                    margin=dict(l=40, r=20, t=30, b=40),
                    xaxis_title="개체",
                    yaxis_title=col,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    showlegend=False,
                )
                return fig

            st.markdown("### 개체별 생육 비교 (막대그래프)")
            if gcols and "개체번호" in growth_df.columns:
                use_df = growth_df_filtered if filter_active else growth_df
                for col in gcols:
                    if col in use_df.columns and use_df[col].dropna().shape[0] >= 1:
                        fig = bar_chart(use_df, col, color_map.get(col, "#2c3e50"))
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(f"{col} 열에 표시할 값이 없습니다.")
            else:
                st.info(
                    "표시할 생육 측정 항목이 없거나 개체번호 열이 없습니다."
                )

            st.markdown("### 생중량 분포 (박스플롯)")
            if "생중량" in growth_df.columns:
                use_df = growth_df_filtered if filter_active else growth_df
                g = pd.to_numeric(use_df["생중량"], errors="coerce").dropna()
                if len(g) >= 1:
                    fig_box = go.Figure()
                    fig_box.add_trace(
                        go.Box(
                            y=g,
                            name="생중량",
                            fillcolor="rgba(69,130,201,0.55)",
                            boxpoints="outliers",
                            line_color="navy",
                            hovertemplate="생중량: %{y:.2f}<extra></extra>",
                        )
                    )
                    fig_box.update_layout(
                        margin=dict(l=40, r=20, t=30, b=40),
                        yaxis_title="생중량(g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_box, width="stretch")
                    if len(g) >= 2:
                        cv = g.std() / g.mean() if g.mean() != 0 else float("nan")
                        st.caption(
                            f"▶ 평균 {g.mean():.2f}g, 표준편차 {g.std():.2f}g, "
                            f"변동계수(CV) {cv:.2f}"
                            f"  —  CV는 표준편차를 평균으로 나눈 값으로, "
                            f"단위가 다른 집단끼리 흩어짐 정도를 비교할 때 씁니다."
                        )
                else:
                    st.info("필터된 범위에 생중량 측정값이 없습니다.")
            else:
                st.info("생중량 열이 없어 분포를 그릴 수 없습니다.")

            if "생중량" in growth_df.columns:
                use_df = growth_df_filtered if filter_active else growth_df
                g = pd.to_numeric(use_df["생중량"], errors="coerce").dropna()
                if len(g) >= 4:
                    q75 = g.quantile(0.75)
                    q25 = g.quantile(0.25)
                    g_valid = use_df["생중량"].dropna()
                    top_idx = g_valid.loc[g_valid >= q75].index
                    bot_idx = g_valid.loc[g_valid <= q25].index
                    top_ids = fmt_id(use_df.loc[top_idx, "개체번호"]) if len(top_idx) else pd.Series(dtype=object)
                    bot_ids = fmt_id(use_df.loc[bot_idx, "개체번호"]) if len(bot_idx) else pd.Series(dtype=object)
                    st.markdown("### 🏆 생중량 상위 25% vs 하위 25%")
                    t1, t2 = st.columns(2)
                    with t1:
                        st.markdown("**🥇 상위 25% 개체**")
                        st.write(", ".join(str(x) for x in top_ids))
                        st.caption(f"기준: 생중량 ≥ {q75:.2f}g")
                    with t2:
                        st.markdown("**😟 하위 25% 개체**")
                        st.write(", ".join(str(x) for x in bot_ids))
                        st.caption(f"기준: 생중량 ≤ {q25:.2f}g")

            st.markdown("### 상관관계 분석 (산점도 + 추세선)")
            pairs = [("잎 수", "생중량"), ("지상부 길이", "생중량")]
            for xcol, ycol in pairs:
                if xcol in growth_df.columns and ycol in growth_df.columns:
                    use_df = growth_df_filtered if filter_active else growth_df
                    tmp = use_df[[xcol, ycol]].dropna()
                    if len(tmp) >= 2:
                        fig_scatter = go.Figure()
                        xv = tmp[xcol].values
                        yv = tmp[ycol].values
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=xv,
                                y=yv,
                                mode="markers",
                                marker=dict(color="#2980b9", size=9, opacity=0.8),
                                name="개체",
                                hovertemplate=f"{xcol}: %{{x:.2f}}<br>{ycol}: %{{y:.2f}}<extra></extra>",
                            )
                        )
                        slope, intercept = np.polyfit(xv, yv, 1)
                        xs = np.linspace(xv.min(), xv.max(), 100)
                        r_val = float(np.corrcoef(xv, yv)[0, 1])
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=xs,
                                y=slope * xs + intercept,
                                mode="lines",
                                line=dict(color="crimson", width=2),
                                name=f"추세선 (r={r_val:.2f})",
                            )
                        )
                        fig_scatter.update_layout(
                            margin=dict(l=40, r=20, t=30, b=40),
                            xaxis_title=xcol,
                            yaxis_title=ycol,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        )
                        st.plotly_chart(fig_scatter, width="stretch")
                        direction = "함께 증가" if r_val > 0 else "반대 방향"
                        strength = (
                            "강한"
                            if abs(r_val) >= 0.7
                            else "중간"
                            if abs(r_val) >= 0.4
                            else "약한"
                        )
                        st.caption(
                            f"▶ 상관계수 r = {r_val:.2f}  —  {strength} {direction} 관계. "
                            f"상관계수(r)는 두 값이 함께 움직이는 정도를 −1에서 +1 사이로 나타낸 것입니다."
                        )
                    else:
                        st.info(
                            f"{xcol}과 {ycol}의 공통 측정값이 2개 미만이라 산점도를 그릴 수 없습니다."
                        )
                else:
                    missing = [c for c in [xcol, ycol] if c not in growth_df.columns]
                    st.info(
                        f"{', '.join(missing)} 열이 없어 상관관계 분석을 건너뜁니다."
                    )

            with st.expander("📄 생육 데이터 원본 + CSV 다운로드"):
                if gcols:
                    display_cols = [
                        c for c in ["개체번호"] + gcols if c in growth_df.columns
                    ]
                    st.dataframe(growth_df[display_cols], width="stretch")
                    st.download_button(
                        label="생육 데이터 CSV 다운로드",
                        data=csv_bytes(growth_df[display_cols]),
                        file_name=f"{school_name or '생육'}_데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 생육 열이 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
