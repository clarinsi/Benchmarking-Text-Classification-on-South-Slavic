import streamlit as st
import pandas as pd
import altair as alt


# ---------------------
# Data loading
# ---------------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")

def load_data(path: str = DEFAULT_RESULTS_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        fallback = os.path.join(BASE_DIR, "results.csv")
        if os.path.exists(fallback):
            path = fallback
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"task", "model", "language", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"results.csv is missing required columns: {', '.join(sorted(missing))}"
        )
    return df


# ---------------------
# Helper functions
# ---------------------
def format_metric_name(metric: str) -> str:
    if not isinstance(metric, str):
        return str(metric)
    metric = metric.replace("-", "_")
    parts = metric.split("_")
    return " ".join(p.capitalize() if p.lower() != "f1" else "F1" for p in parts)


def language_task_metric_summary(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []

    grouped = (
        df.groupby(["language", "task"])["metric"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )

    lang_map: dict[str, dict[str, list[str]]] = {}
    for _, row in grouped.iterrows():
        lang = row["language"]
        task = row["task"]
        metrics = [format_metric_name(m) for m in row["metric"]]
        lang_map.setdefault(lang, {})[task] = metrics

    lines: list[str] = []
    for lang in sorted(lang_map.keys()):
        task_parts = []
        for task in sorted(lang_map[lang].keys()):
            metrics_str = ", ".join(lang_map[lang][task])
            task_parts.append(f"{task} ({metrics_str})")
        if task_parts:
            lines.append(f"{lang}: " + "; ".join(task_parts))
    return lines


def compute_model_rankings(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    if df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "Model",
                    "Average ranking",
                    "Average ranking English",
                    "Average ranking South Slavic",
                ]
            ),
            {},
            [],
        )

    df_all = df.copy()
    tasks_all = sorted(df_all["task"].unique())
    n_tasks = len(tasks_all)

    tasks_per_model = df_all.groupby("model")["task"].nunique()
    full_models = tasks_per_model[tasks_per_model == n_tasks].index.tolist()

    if not full_models:
        return (
            pd.DataFrame(
                columns=[
                    "Model",
                    "Average ranking",
                    "Average ranking English",
                    "Average ranking South Slavic",
                ]
            ),
            {},
            [],
        )

    df_rank = df_all[df_all["model"].isin(full_models)].copy()

    agg = (
        df_rank.groupby(["task", "language", "model"])["value"]
        .mean()
        .reset_index(name="score")
    )

    if agg.empty:
        return (
            pd.DataFrame(
                columns=[
                    "Model",
                    "Average ranking",
                    "Average ranking English",
                    "Average ranking South Slavic",
                ]
            ),
            {},
            full_models,
        )

    agg["rank"] = agg.groupby(["task", "language"])["score"].rank(
        ascending=False, method="average"
    )

    overall_rank = agg.groupby("model")["rank"].mean().rename("Average ranking")

    per_lang_rank = agg.groupby(["model", "language"])["rank"].mean().reset_index()

    languages_all = sorted(per_lang_rank["language"].unique())
    english_langs = [l for l in languages_all if l.strip().lower() in {"english", "en"}]

    if english_langs:
        english_rank = (
            per_lang_rank[per_lang_rank["language"].isin(english_langs)]
            .groupby("model")["rank"]
            .mean()
            .rename("Average ranking English")
        )
    else:
        english_rank = pd.Series(
            index=overall_rank.index,
            data=float("nan"),
            name="Average ranking English",
        )

    if english_langs:
        south_slavic_rank = (
            per_lang_rank[~per_lang_rank["language"].isin(english_langs)]
            .groupby("model")["rank"]
            .mean()
            .rename("Average ranking South Slavic")
        )
    else:
        south_slavic_rank = overall_rank.rename("Average ranking South Slavic")

    lang_pivot = (
        per_lang_rank.pivot(index="model", columns="language", values="rank")
        if not per_lang_rank.empty
        else pd.DataFrame()
    )

    rank_df = overall_rank.to_frame().join(english_rank, how="left").join(
        south_slavic_rank, how="left"
    )

    if not lang_pivot.empty:
        rank_df = rank_df.join(lang_pivot, how="left")

    rank_df = rank_df.reset_index().rename(columns={"model": "Model"})

    base_cols = [
        "Model",
        "Average ranking",
        "Average ranking English",
        "Average ranking South Slavic",
    ]
    lang_cols = [c for c in rank_df.columns if c not in base_cols]
    rank_df = rank_df[base_cols + lang_cols]

    rank_df = rank_df.sort_values("Average ranking", ascending=True).reset_index(
        drop=True
    )

    tasks_per_lang_series = agg.groupby("language")["task"].apply(
        lambda s: sorted(s.unique())
    )
    tasks_per_lang = tasks_per_lang_series.to_dict()

    return rank_df, tasks_per_lang, full_models


def apply_filters(
    df: pd.DataFrame,
    tasks: list[str] | None,
    languages: list[str],
    models: list[str],
) -> pd.DataFrame:
    data = df.copy()

    if tasks:
        data = data[data["task"].isin(tasks)]

    if languages:
        data = data[data["language"].isin(languages)]

    if models:
        data = data[data["model"].isin(models)]

    return data


def order_metrics(metrics: list[str]) -> list[str]:
    def metric_key(m: str) -> tuple[int, str]:
        m_norm = m.lower().replace("-", "_")
        if "macro" in m_norm and "f1" in m_norm:
            return (0, m_norm)
        if "micro" in m_norm and "f1" in m_norm:
            return (1, m_norm)
        return (2, m_norm)

    return sorted(metrics, key=metric_key)


def sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


# ---------------------
# Visual style
# ---------------------
BRAND = "#0080aa"
BRAND_DARK = "#005f80"
HIGHLIGHT_PALETTE = [
    "#0080aa", "#e8590c", "#2f9e44", "#9c36b5", "#c92a2a",
    "#1971c2", "#f08c00", "#0c8599", "#5f3dc4", "#a61e4d",
]

CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* Text only: leave spans alone so Streamlit's Material icon font keeps working */
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp div[data-testid="stMetricValue"],
.stApp button p, .hero-eyebrow, .hero-title, .hero-lead {
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1400px; }
.hero-eyebrow {
    color: #0080aa; font-weight: 600; font-size: 0.85rem;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.hero-title {
    font-size: 2.4rem; font-weight: 700; line-height: 1.15;
    margin: 0 0 0.8rem 0; color: #10212b;
}
.hero-lead { font-size: 1.08rem; color: #3d4f58; max-width: 60rem; line-height: 1.6; }
.hero-lead a { color: #0080aa; }
.section-note { color: #5b6b73; font-size: 0.95rem; margin-top: -0.4rem; }
div[data-testid="stMetric"] {
    background: #f4f8fa; border: 1px solid #dde8ed; border-radius: 12px;
    padding: 0.9rem 1.1rem;
}
div[data-testid="stMetricValue"] { color: #005f80; font-weight: 700; font-size: 1.9rem; }
div[data-testid="stMetricValue"] > div { white-space: normal; overflow-wrap: anywhere; line-height: 1.2; }
div[data-testid="stMetricLabel"] p { color: #5b6b73; font-weight: 500; }
button[data-baseweb="tab"] p { font-size: 1.05rem; font-weight: 600; }
span[data-baseweb="tag"] { background-color: #0080aa !important; }
</style>
"""


def heatmap_domain(values: pd.Series) -> list[float]:
    lo = float(values.min()) if not values.empty else 0.0
    return [max(0.0, round(lo - 0.02, 2)), 1.0]


def task_score_matrix(df_tm: pd.DataFrame) -> pd.DataFrame:
    """Models x languages matrix of mean scores, sorted by average (best first)."""
    matrix = df_tm.pivot_table(index="model", columns="language", values="value", aggfunc="mean")
    if matrix.empty:
        return matrix
    matrix["Average"] = matrix.mean(axis=1)
    return matrix.sort_values("Average", ascending=False)


def plot_task_heatmap(matrix: pd.DataFrame, metric_label: str):
    long = (
        matrix.reset_index()
        .melt(id_vars="model", var_name="language", value_name="score")
        .dropna(subset=["score"])
    )
    model_order = list(matrix.index)
    lang_order = [c for c in matrix.columns if c != "Average"] + ["Average"]
    col_best = long.groupby("language")["score"].transform("max")
    long["is_best"] = long["score"] == col_best

    base = alt.Chart(long).encode(
        x=alt.X("language:N", sort=lang_order, title=None,
                axis=alt.Axis(orient="top", labelAngle=-35, labelLimit=180, labelOverlap=False)),
        y=alt.Y("model:N", sort=model_order, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
    )
    rect = base.mark_rect(stroke="white", strokeWidth=1.5, cornerRadius=2).encode(
        color=alt.Color(
            "score:Q",
            title=metric_label,
            scale=alt.Scale(scheme="redyellowgreen", domain=heatmap_domain(long["score"])),
            legend=alt.Legend(orient="bottom", gradientLength=260),
        ),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("language:N", title="Language"),
            alt.Tooltip("score:Q", title=metric_label, format=".3f"),
        ],
    )
    text_enc = dict(text=alt.Text("score:Q", format=".3f"), color=alt.value("#1b1b1b"))
    text = base.transform_filter("!datum.is_best").mark_text(fontSize=11).encode(**text_enc)
    text_best = base.transform_filter("datum.is_best").mark_text(fontSize=11, fontWeight="bold").encode(**text_enc)
    chart = (rect + text + text_best).properties(height=max(260, 26 * len(model_order)))
    st.altair_chart(chart, width="stretch")


def plot_language_profile(df_tm: pd.DataFrame, highlight: list[str], lang_order: list[str], metric_label: str):
    """All models as faint dots per language; highlighted models as coloured lines."""
    data = df_tm.groupby(["model", "language"])["value"].mean().reset_index()
    data["highlighted"] = data["model"].isin(highlight)

    x_enc = alt.X(
        "language:N", sort=lang_order, title=None,
        axis=alt.Axis(labelAngle=-35, labelLimit=180, labelOverlap=False),
    )
    y_enc = alt.Y("value:Q", title=metric_label, scale=alt.Scale(zero=False, nice=True))
    tooltip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("language:N", title="Language"),
        alt.Tooltip("value:Q", title=metric_label, format=".3f"),
    ]

    background = (
        alt.Chart(data[~data["highlighted"]])
        .mark_circle(size=45, color="#9aa9b1", opacity=0.45)
        .encode(x=x_enc, y=y_enc, tooltip=tooltip)
    )
    color = alt.Color(
        "model:N", title="Highlighted",
        scale=alt.Scale(domain=highlight, range=HIGHLIGHT_PALETTE[: len(highlight)]),
        legend=alt.Legend(orient="bottom", columns=4, labelLimit=220),
    )
    hl = alt.Chart(data[data["highlighted"]])
    lines = hl.mark_line(strokeWidth=2.5, opacity=0.85).encode(x=x_enc, y=y_enc, color=color)
    points = hl.mark_circle(size=90, opacity=1).encode(x=x_enc, y=y_enc, color=color, tooltip=tooltip)

    chart = (background + lines + points).properties(height=400)
    st.altair_chart(chart, width="stretch")


def styled_scores(table: pd.DataFrame, num_cols: list[str], cmap: str, vmin=None, vmax=None):
    styler = table.style.format("{:.3f}", subset=num_cols, na_rep="–")
    try:
        styler = styler.background_gradient(cmap=cmap, subset=num_cols, vmin=vmin, vmax=vmax, axis=None)
    except Exception:
        pass  # matplotlib missing: show plain numbers
    return styler


def show_metric_results_table(matrix: pd.DataFrame, task_name: str, metric_label: str, file_base: str):
    table = matrix.reset_index().rename(columns={"model": "Model"})
    table.columns.name = None
    num_cols = [c for c in table.columns if c != "Model"]
    lo = heatmap_domain(matrix.stack())[0]
    st.dataframe(
        styled_scores(table, num_cols, "RdYlGn", vmin=lo, vmax=1.0),
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(table), 760),
    )
    st.download_button(
        "Download table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{file_base}.csv",
        mime="text/csv",
        key=f"dl_{file_base}",
    )


def coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["language", "task"])["model"].nunique().unstack(fill_value=0)
    counts.columns.name = None
    return counts.reset_index().rename(columns={"language": "Language / dialect"})


# ---------------------
# Page sections
# ---------------------
def render_header():
    col_title, col_logos = st.columns([5, 1], vertical_alignment="center")
    with col_title:
        st.markdown(
            """
            <div class="hero-eyebrow">CLASSLA &middot; CLARIN.SI</div>
            <div class="hero-title">LLM Evaluation Dashboard<br>for South Slavic Languages</div>
            <div class="hero-lead">
            How well do large language models (LLMs) and fine-tuned classifiers handle text classification
            and commonsense reasoning in South Slavic languages and dialects? Details on the models and
            benchmarks are in the paper
            <a href="https://doi.org/10.48550/arXiv.2511.07989">"State of the Art in Text Classification for
            South Slavic Languages: Fine-Tuning or Prompting?"</a> (Kuzman Pungeršek et al., 2026). The code is on
            <a href="https://github.com/TajaKuzman/Benchmarking-Text-Classification-on-South-Slavic">GitHub</a>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_logos:
        st.image(os.path.join(BASE_DIR, "CLASSLA-k-centre-transparent-background.png"), width="stretch")
        st.image(os.path.join(BASE_DIR, "logo-clarin.si-en.png"), width="stretch")


def render_kpis(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", df["model"].nunique())
    c2.metric("Benchmarks", df["task"].nunique())
    c3.metric("Languages & dialects", df["language"].nunique())
    c4.metric("Reported scores", f"{len(df):,}")


def render_leaderboard(df: pd.DataFrame):
    st.markdown(
        '<p class="section-note">Models are ranked separately for each task–language pair by their average '
        "score (across all available metrics); ranks (1 = best) are then averaged. Only models evaluated on "
        "<b>every selected benchmark</b> are ranked, so selecting fewer benchmarks brings in more models.</p>",
        unsafe_allow_html=True,
    )

    tasks_all = sorted(df["task"].unique())
    rank_tasks = st.pills(
        "Benchmarks used for the ranking",
        options=tasks_all,
        selection_mode="multi",
        default=tasks_all,
        key="rank_tasks",
    )
    if not rank_tasks:
        st.info("Select at least one benchmark to compute a ranking.")
        return

    ranking_df, tasks_per_lang, full_models = compute_model_rankings(df[df["task"].isin(rank_tasks)])
    if ranking_df.empty:
        st.info("No model was evaluated on all selected benchmarks.")
        return

    st.caption(f"{len(full_models)} models were evaluated on all {len(rank_tasks)} selected benchmark(s).")

    base_cols = ["Model", "Average ranking", "Average ranking English", "Average ranking South Slavic"]
    lang_cols = [c for c in ranking_df.columns if c not in base_cols]

    col_chart, col_table = st.columns([2, 3], gap="large")
    with col_chart:
        chart_df = ranking_df[["Model", "Average ranking"]].copy()
        chart_df["Position"] = range(1, len(chart_df) + 1)
        bars = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusEnd=4, height={"band": 0.7})
            .encode(
                y=alt.Y("Model:N", sort=list(chart_df["Model"]), title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X(
                    "Average ranking:Q",
                    title="Average rank (lower is better)",
                    scale=alt.Scale(domain=[0, float(chart_df["Average ranking"].max()) * 1.15]),
                ),
                color=alt.Color(
                    "Average ranking:Q",
                    scale=alt.Scale(range=[BRAND_DARK, "#bfe3ef"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Position:Q"),
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Average ranking:Q", format=".2f"),
                ],
            )
        )
        labels = bars.mark_text(align="left", dx=4, fontSize=11, color="#33454e").encode(
            text=alt.Text("Average ranking:Q", format=".2f"), color=alt.value("#33454e")
        )
        st.altair_chart(
            (bars + labels).properties(height=max(220, 30 * len(chart_df))), width="stretch"
        )

    with col_table:
        show_langs = st.toggle("Show rank per language", value=False, key="rank_show_langs")
        cols = base_cols + (lang_cols if show_langs else [])
        table = ranking_df[cols].copy()
        table.insert(0, "#", range(1, len(table) + 1))
        num_cols = [c for c in cols if c != "Model"]
        st.dataframe(
            styled_scores(table, num_cols, "Blues_r", vmin=1 - 1.2 * len(table), vmax=1.4 * max(len(table), 2)).format(
                "{:.2f}", subset=num_cols, na_rep="–"
            ),
            hide_index=True,
            width="stretch",
            height=min(38 + 35 * len(table), 760),
        )

    if tasks_per_lang:
        with st.expander("Benchmarks included in the ranking per language"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Language / dialect": sorted(tasks_per_lang),
                        "Benchmarks": [", ".join(tasks_per_lang[l]) for l in sorted(tasks_per_lang)],
                    }
                ),
                hide_index=True,
                width="stretch",
            )


def render_task_explorer(df: pd.DataFrame):
    tasks = sorted(df["task"].unique())
    default_task = "COPA" if "COPA" in tasks else tasks[0]
    task = st.segmented_control("Benchmark", options=tasks, default=default_task, key="explore_task")
    if not task:
        task = default_task

    df_t = df[df["task"] == task]
    metrics = order_metrics(sorted(df_t["metric"].unique()))
    metric = metrics[0]
    if len(metrics) > 1:
        metric = st.segmented_control(
            "Metric", options=metrics, default=metrics[0], format_func=format_metric_name, key=f"metric_{task}"
        ) or metrics[0]
    metric_label = format_metric_name(metric)
    df_tm = df_t[df_t["metric"] == metric]

    languages = sorted(df_tm["language"].unique())
    models = sorted(df_tm["model"].unique())
    with st.expander("Filter languages and models"):
        sel_langs = st.multiselect(
            "Languages / dialects", languages, placeholder="All languages", key=f"langs_{task}"
        )
        sel_models = st.multiselect("Models", models, placeholder="All models", key=f"models_{task}")
    df_tm = apply_filters(df_tm, tasks=None, languages=sel_langs, models=sel_models)
    if df_tm.empty:
        st.warning("No results match the current filters.")
        return

    matrix = task_score_matrix(df_tm)
    lang_order = [c for c in matrix.columns if c != "Average"]

    best_model = matrix.index[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model (average)", best_model, f"{matrix.loc[best_model, 'Average']:.3f}", delta_color="off", delta_arrow="off")
    lang_means = matrix[lang_order].mean()
    if len(lang_order) > 1:
        c2.metric("Easiest language", lang_means.idxmax(), f"mean {lang_means.max():.3f}", delta_color="off", delta_arrow="off")
        c3.metric("Hardest language", lang_means.idxmin(), f"mean {lang_means.min():.3f}", delta_color="off", delta_arrow="off")

    st.markdown(f"#### Scores per model and language ({metric_label})")
    st.caption("Models are sorted by their average score. The best score in each column is in bold.")
    plot_task_heatmap(matrix, metric_label)

    st.markdown("#### How models compare across languages")
    highlight = st.multiselect(
        "Highlight models",
        options=list(matrix.index),
        default=list(dict.fromkeys([matrix.index[0], matrix.index[len(matrix) // 2], matrix.index[-1]])),
        max_selections=len(HIGHLIGHT_PALETTE),
        key=f"highlight_{task}_{metric}",
    )
    st.caption("By default the best, a middle and the weakest model are highlighted. Grey dots are the other models; hover over any point for details.")
    plot_language_profile(df_tm, highlight, lang_order, metric_label)

    st.markdown("#### Results table")
    show_metric_results_table(matrix, task, metric_label, sanitize_filename(f"{task}__{metric}"))

    if task in {"PIQA", "COPA"}:
        st.caption(
            "Serbian, Torlak and Šatrovački labels without a script suffix denote Latin; "
            "Cyrillic results are shown separately."
        )


def render_about(df: pd.DataFrame, model_summary_lines: list[str]):
    st.markdown("#### Coverage")
    st.caption("Number of models with results for each language and benchmark.")
    cov = coverage_matrix(df)
    task_cols = [c for c in cov.columns if c != "Language / dialect"]
    styler = cov.style
    try:
        styler = styler.background_gradient(
            cmap="Blues", subset=task_cols, vmin=0, vmax=2.2 * int(cov[task_cols].to_numpy().max()), axis=None
        )
    except Exception:
        pass
    st.dataframe(
        styler.format(lambda v: "" if v == 0 else v, subset=task_cols),
        hide_index=True,
        width="stretch",
        height=38 + 35 * len(cov),
    )

    st.markdown("#### Evaluated models")
    st.markdown("\n".join(f"- {line}" for line in model_summary_lines))


# ---------------------
# Streamlit layout
# ---------------------
def main():
    st.set_page_config(
        page_title="CLASSLA LLM Evaluation Dashboard for South Slavic Languages",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    render_header()

    try:
        df = load_data()
    except Exception as e:
        st.error(
            "Could not load results.csv. Make sure it exists next to app.py "
            "and has columns: task, model, language, metric, value."
        )
        st.exception(e)
        return

    st.write("")
    render_kpis(df)
    st.write("")

    model_summary_lines = [
        'Claude Haiku 4.5 - [`anthropic/claude-haiku-4.5`](https://openrouter.ai/anthropic/claude-haiku-4.5) ([Anthropic, 2025](https://www-cdn.anthropic.com/7aad69bf12627d42234e01ee7c36305dc2f6a970.pdf)): a closed-source hybrid reasoning large language model from Anthropic. Available details on the model architecture and language coverage are limited.',
        'DeepSeek-R1-Distill-Qwen-14B (`deepseek-r1:14b`) and DeepSeek-R1-0528 (`deepseek/deepseek-r1-0528`) are displayed separately. Each score retains its original model identifier in the results CSV.',
        '[GaMS-27B-Instruct](https://huggingface.co/cjvt/GaMS-27B-Instruct) - [`GaMS-27B-Instruct-i1-GGUF:latest`](https://huggingface.co/mradermacher/GaMS-27B-Instruct-i1-GGUF): an open-source 27B parameter model from the GaMS (Generative Model for Slovene) family ([Vreš et al., 2024](https://repozitorij.uni-lj.si/IzpisGradiva.php?id=164282)). The model is based on the Gemma 2 model and continually pretrained on Slovenian, English and some portion of Croatian, Serbian and Bosnian corpora.',
        'Gemini 2.5 Flash - [`google/gemini-2.5-flash`](https://openrouter.ai/google/gemini-2.5-flash), Gemini 2.5 Pro - [`google/gemini-2.5-pro`](https://openrouter.ai/google/gemini-2.5-pro) ([Comanici et al., 2025](https://arxiv.org/abs/2507.06261)) and Gemini 3 Flash Preview - [`google/gemini-3-flash-preview`](https://openrouter.ai/google/gemini-3-flash-preview) ([Google DeepMind, 2025](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf)): closed-source multilingual and multimodal instruction-tuned LLMs by Google DeepMind. The models are reported to be pretrained on over 400 languages, however, details on the language coverage are not available.',
        '[Gemma 3](https://ollama.com/library/gemma3) ([Gemma Team et al., 2025](https://arxiv.org/pdf/2503.19786)): an open-weight multilingual instruction-tuned LLM, developed by Google DeepMind. The model was pretrained on multimodal data with large quantities of multilingual texts and is reported to support over 140 languages. We use the model in 27 billion parameter size.',
        'GPT-3.5-Turbo - `gpt-3.5-turbo-0125` ([OpenAI, 2023](https://help.openai.com/en/articles/6783457-chatgpt-general-faq)); GPT-4o - `gpt-4o-2024-08-06` and GPT-4o-mini - `gpt-4o-mini-2024-07-18` ([OpenAI, 2024](https://openai.com/index/hello-gpt-4o/)); GPT-5 - `gpt-5-2025-0807`, GPT-5-Nano - `gpt-5-nano-2025-08-07`, and GPT-5-mini - `gpt-5-mini-2025-08-07` ([OpenAI, 2025](https://openai.com/index/introducing-gpt-5/)): closed-source instruction-tuned LLMs developed by OpenAI. OpenAI states that the models are trained on large multilingual web corpora, however, specific details about the training data, procedures, and architecture are not publicly known.',
        '[LLaMA 3.3](https://ollama.com/library/llama3.3) ([Meta, 2024](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md)): an open-weight instruction-tuned multilingual LLM, developed by Meta, with 70 billion parameters. The model was pretrained on a web text collection in various languages, however, it is reported to support only 8 languages, namely, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.',
        '[Llama 4 Scout](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) - [`llama4:scout`](https://ollama.com/library/llama4): an open-weight instruction-tuned multilingual and multimodal LLM, developed by Meta, with 17 billion active parameters and 109 billion total parameters. It was pretrained on 200 languages, but it is said to support only the following languages: Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese.',
        'Mistral Medium 3.1 - [`mistralai/mistral-medium-3.1`](https://openrouter.ai/mistralai/mistral-medium-3.1/api), and Mistral Small 3.2 - [`mistralai/mistral-small-3.2-24b-instruct`](https://openrouter.ai/mistralai/mistral-small-3.2-24b-instruct) ([Mistral AI, 2025](https://mistral.ai/news/mistral-medium-3)): closed-source multimodal instruction-tuned models by Mistral AI. Available details on the model architecture and language coverage are very limited.',
        'Qwen 3 - [`Qwen3-2504`](https://ollama.com/library/qwen3) ([Yang et al., 2025](https://arxiv.org/abs/2505.09388)): an open-weight LLM, developed by Alibaba Cloud. We use the model with the 32 billion parameter size, namely, the qwen3:32b model. The model is said to support over 100 languages and dialects.',
        'BERT-like models, fine-tuned to the evaluated task: [X-GENRE classifier](https://doi.org/10.57967/hf/0927) for automatic genre identification ([Kuzman et al., 2023](https://www.mdpi.com/2504-4990/5/3/59)), [IPTC XLM-R classifier](https://huggingface.co/classla/multilingual-IPTC-news-topic-classifier) for news topic classification ([Kuzman & Ljubešić, 2025](https://doi.org/10.1109/ACCESS.2025.3544814)), [ParlaCAP-classifier](https://huggingface.co/classla/ParlaCAP-Topic-Classifier) for policy topic classification ([Kuzman Pungeršek et al., 2026](http://arxiv.org/abs/2602.16516)), and [XLM-R-ParlaSent model](https://huggingface.co/classla/xlm-r-parlasent) for sentiment identification ([Mochtak et al., 2024](https://aclanthology.org/2024.lrec-main.1393/))'
    ]

    tab_rank, tab_explore, tab_about = st.tabs(["Leaderboard", "Explore by benchmark", "Models & coverage"])
    with tab_rank:
        render_leaderboard(df)
    with tab_explore:
        render_task_explorer(df)
    with tab_about:
        render_about(df, model_summary_lines)


if __name__ == "__main__":
    main()
