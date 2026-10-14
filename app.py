"""Restaurant Food Wastage Prediction & Management System (Streamlit)."""
from __future__ import annotations

import traceback

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import config as C
from utils.model_loader import (ModelLoadError, feature_importance, load_csv, load_dataset,
                                load_models, numeric_ranges)
from utils.predictor import band_for_amount, category_boundaries, predict, validate_inputs
from utils.recommendations import build_recommendations

st.set_page_config(page_title="Food Wastage Prediction", page_icon="🍽️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1180px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .hero-sub {font-size: 1.1rem; color: #3E5A52; margin-top: -0.6rem; margin-bottom: 1.2rem;}
    .flow {display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin: .4rem 0 1.4rem 0;}
    .flow span.step {background:#1F5A4B; color:#fff; padding:.55rem .9rem; border-radius:6px; font-weight:600;}
    .flow span.arrow {color:#1F5A4B; font-weight:700;}
    .badge {display:inline-block; padding:.35rem .9rem; border-radius:6px; color:#fff; font-weight:700; font-size:1.3rem;}
    .note {border-left:4px solid #E0A526; background:#FFF8E6; padding:.7rem 1rem; border-radius:4px; font-size:.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = ["Home / Dashboard", "Food Waste Prediction", "Waste Analysis",
         "Model Performance", "Recommendations", "About Project"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.caption("Predictions come from the saved Random Forest models in `models/`.")

dataset = load_dataset()


def get_models():
    """Load models, or show a precise error and stop the page."""
    try:
        return load_models()
    except ModelLoadError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error("Unexpected error while loading the models.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        st.stop()


def require_csv(path):
    df = load_csv(str(path))
    if df is None:
        st.warning(f"`{path.name}` was not found in `data/`, so this section is unavailable.")
    return df


def style_fig(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#17231F"))
    return fig


# ----------------------------------------------------------------------------- Home
def page_home():
    st.title(C.APP_TITLE)
    st.markdown(f"<div class='hero-sub'>{C.APP_SUBTITLE}</div>", unsafe_allow_html=True)
    st.write(
        "Restaurants and caterers often prepare more food than guests eat. Demand is uncertain, "
        "numbers of guests shift, and planning by habit rather than evidence leads to over-preparation. "
        "This app uses two trained models to estimate how much food is likely to be wasted for a planned "
        "service, and turns that estimate into practical guidance."
    )
    st.subheader("How it works")
    st.markdown(
        "<div class='flow'><span class='step'>Restaurant Inputs</span><span class='arrow'>&rarr;</span>"
        "<span class='step'>Regression Prediction</span><span class='arrow'>&rarr;</span>"
        "<span class='step'>Waste Category</span><span class='arrow'>&rarr;</span>"
        "<span class='step'>Recommendations</span></div>", unsafe_allow_html=True)

    cols = st.columns(4)
    cards = [
        ("Regression", "Random Forest regressor estimating the wastage amount from 10 event and food inputs."),
        ("Classification", "Random Forest classifier assigning Low, Medium or High wastage."),
        ("Clustering", "K-Means, hierarchical and DBSCAN results grouping similar records."),
        ("Recommendations", "Guidance based on the predicted category and the inputs you entered."),
    ]
    for col, (title, text) in zip(cols, cards):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)

    if dataset is not None:
        st.subheader("About the supplied dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(dataset):,}")
        c2.metric("Input columns", dataset.shape[1] - 1)
        c3.metric("Mean wastage amount", f"{dataset[C.TARGET_COL].mean():.1f}")
        st.caption("Figures are computed from `data/food_wastage_data.csv`.")


# ----------------------------------------------------------------------------- Prediction
def page_prediction():
    st.title("Food Waste Prediction")
    st.caption("Enter the planned service details. Both saved models receive exactly the same inputs.")
    models = get_models()
    schema = models.schema
    ranges = numeric_ranges(dataset, schema)
    bounds = category_boundaries(dataset)

    def default_for(col):
        return int(dataset[col].median()) if dataset is not None and col in dataset else 1

    values = {}
    with st.form("predict_form"):
        st.markdown("##### Event details")
        a, b, c = st.columns(3)
        values["Event Type"] = a.selectbox("Event Type", schema.categorical["Event Type"])
        values["Number of Guests"] = b.number_input("Number of Guests", min_value=1, step=1,
                                                    value=default_for("Number of Guests"))
        values["Geographical Location"] = c.selectbox("Geographical Location",
                                                      schema.categorical["Geographical Location"])
        st.markdown("##### Food and preparation")
        a, b, c = st.columns(3)
        values["Type of Food"] = a.selectbox("Type of Food", schema.categorical["Type of Food"])
        values["Quantity of Food"] = b.number_input("Quantity of Food", min_value=1, step=1,
                                                    value=default_for("Quantity of Food"))
        values["Preparation Method"] = c.selectbox("Preparation Method",
                                                   schema.categorical["Preparation Method"])
        st.markdown("##### Storage, purchasing and context")
        a, b, c, d = st.columns(4)
        values["Storage Conditions"] = a.selectbox("Storage Conditions", schema.categorical["Storage Conditions"])
        values["Purchase History"] = b.selectbox("Purchase History", schema.categorical["Purchase History"])
        values["Seasonality"] = c.selectbox("Seasonality", schema.categorical["Seasonality"])
        values["Pricing"] = d.selectbox("Pricing", schema.categorical["Pricing"])
        submitted = st.form_submit_button("Predict Food Waste", type="primary")

    if not submitted:
        return

    errors, warns = validate_inputs(values, models, ranges)
    for e in errors:
        st.error(e)
    if errors:
        return
    for w in warns:
        st.warning(w)

    try:
        res = predict(values, models)
    except Exception:  # noqa: BLE001
        st.error("The prediction could not be completed. Check the inputs and try again.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return

    st.session_state["last_prediction"] = {"values": values, "result": res}

    st.subheader("Result")
    left, right = st.columns([1, 1])
    with left.container(border=True):
        st.caption("Predicted wastage amount (regression model)")
        st.metric("Predicted wastage", f"{res['predicted_amount']:.1f}",
                  help="The dataset does not state a unit for 'Wastage Food Amount', so none is shown.")
        st.caption("Same scale as the 'Wastage Food Amount' column in the dataset (unit not specified).")
    with right.container(border=True):
        st.caption("Predicted waste category (classification model)")
        color = C.CATEGORY_COLORS.get(res["category"], "#555")
        st.markdown(f"<span class='badge' style='background:{color}'>{res['category']}</span>",
                    unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Category": list(res["probabilities"]),
                                "Probability": list(res["probabilities"].values())})
        fig = px.bar(prob_df, x="Probability", y="Category", orientation="h", color="Category",
                     color_discrete_map=C.CATEGORY_COLORS, range_x=[0, 1])
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig, 170), width="stretch")

    band = band_for_amount(res["predicted_amount"], bounds)
    if bounds:
        st.caption(
            f"Category bands used when the labels were created (tertiles of the dataset): "
            f"Low up to {bounds['low_max']:.1f}, Medium up to {bounds['medium_max']:.1f}, High above that."
        )
        if band and band != res["category"]:
            st.info(f"The two models are trained separately. The regression value falls in the "
                    f"**{band}** band while the classifier predicts **{res['category']}**. "
                    "Treat this as a borderline case.")

    st.subheader("Your inputs")
    summary = pd.DataFrame({"Input": list(values), "Value": [str(v) for v in values.values()]})
    st.dataframe(summary, hide_index=True, width="stretch")

    st.subheader("Recommendations")
    recs = build_recommendations(res["category"], values, dataset)
    for r in recs["core"]:
        st.markdown(f"- {r}")
    if recs["context"]:
        st.markdown("**Based on your inputs**")
        for r in recs["context"]:
            st.markdown(f"- {r}")
    st.markdown("<div class='note'>Recommendations are guidance, not guaranteed outcomes.</div>",
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- Analysis
def cluster_tab(path, col, title, noise_label=False):
    df = require_csv(path)
    if df is None:
        return
    df = df.copy()
    df["Cluster"] = df[col].map(lambda x: "Noise (-1)" if x == -1 else f"Cluster {x}")
    st.caption(f"Clustering result from `{path.name}` ({len(df):,} records). "
               "Clusters group similar records; they are not predictions.")
    sizes = df["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Records")
    c1, c2 = st.columns(2)
    c1.plotly_chart(style_fig(px.bar(sizes, x="Cluster", y="Records", title=f"{title}: cluster sizes",
                                     color_discrete_sequence=["#1F5A4B"])), width="stretch")
    c2.plotly_chart(style_fig(px.box(df.sort_values("Cluster"), x="Cluster", y=C.TARGET_COL,
                                     title="Wastage amount by cluster",
                                     color_discrete_sequence=["#1F5A4B"])), width="stretch")
    prof = df.groupby("Cluster").agg(Records=(C.TARGET_COL, "size"),
                                     Avg_Guests=("Number of Guests", "mean"),
                                     Avg_Quantity=("Quantity of Food", "mean"),
                                     Avg_Wastage=(C.TARGET_COL, "mean")).round(1)
    st.markdown("**Cluster profile (averages)**")
    st.dataframe(prof, width="stretch")
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("str")]
    cat_cols = [c for c in cat_cols if c != "Cluster"]
    pick = st.selectbox("Compare a categorical column across clusters", cat_cols, key=f"pick_{col}")
    mix = df.groupby(["Cluster", pick]).size().reset_index(name="Records")
    st.plotly_chart(style_fig(px.bar(mix, x="Cluster", y="Records", color=pick, barmode="stack")),
                    width="stretch")


def page_analysis():
    st.title("Waste Analysis")
    st.caption("Everything here is drawn from the supplied output files. Each tab says what kind of output it is.")
    tabs = st.tabs(["K-Means", "Hierarchical", "DBSCAN", "DBSCAN noise",
                    "Regression: actual vs predicted", "Classification: actual vs predicted"])
    with tabs[0]:
        cluster_tab(C.KMEANS_PATH, "KMeans_Cluster", "K-Means")
    with tabs[1]:
        cluster_tab(C.HIER_PATH, "Hierarchical_Cluster", "Hierarchical")
    with tabs[2]:
        cluster_tab(C.DBSCAN_PATH, "DBSCAN_Cluster", "DBSCAN")
    with tabs[3]:
        noise = require_csv(C.DBSCAN_NOISE_PATH)
        allr = load_csv(str(C.DBSCAN_PATH))
        if noise is not None:
            st.caption("Records DBSCAN labelled as noise (cluster -1), from `dbscan_noise_records.csv`. "
                       "Noise means they did not fit a dense cluster, not that they are errors.")
            m1, m2 = st.columns(2)
            m1.metric("Noise records", f"{len(noise):,}")
            if allr is not None:
                m2.metric("Share of all records", f"{len(noise) / len(allr):.1%}")
                cmp = pd.concat([noise.assign(Group="Noise"),
                                 allr[allr["DBSCAN_Cluster"] != -1].assign(Group="In a cluster")])
                st.plotly_chart(style_fig(px.histogram(cmp, x=C.TARGET_COL, color="Group", barmode="overlay",
                                                       opacity=0.65, nbins=30,
                                                       color_discrete_sequence=["#C8553D", "#1F5A4B"],
                                                       title="Wastage amount: noise vs clustered records")),
                                width="stretch")
            st.dataframe(noise, width="stretch", hide_index=True)
    with tabs[4]:
        rp = require_csv(C.REG_PRED_PATH)
        if rp is not None:
            st.caption(f"Model evaluation output on held-out test rows ({len(rp):,} rows). "
                       "These are predictions compared with known values, not historical observations.")
            lo, hi = rp.min().min(), rp.max().max()
            fig = px.scatter(rp, x="Actual Wastage", y="Predicted Wastage", opacity=0.6,
                             color_discrete_sequence=["#1F5A4B"], title="Actual vs predicted wastage")
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                     line=dict(dash="dash", color="#C8553D")))
            c1, c2 = st.columns(2)
            c1.plotly_chart(style_fig(fig), width="stretch")
            err = (rp["Predicted Wastage"] - rp["Actual Wastage"]).rename("Prediction error")
            c2.plotly_chart(style_fig(px.histogram(err, nbins=30, title="Prediction error (predicted minus actual)",
                                                   color_discrete_sequence=["#E0A526"])), width="stretch")
    with tabs[5]:
        cp = require_csv(C.CLS_PRED_PATH)
        if cp is not None:
            st.caption(f"Model evaluation output on held-out test rows ({len(cp):,} rows).")
            order = [c for c in C.CATEGORY_ORDER]
            cm = pd.crosstab(cp["Actual Category"], cp["Predicted Category"]).reindex(
                index=order, columns=order, fill_value=0)
            c1, c2 = st.columns(2)
            c1.plotly_chart(style_fig(px.imshow(cm, text_auto=True, color_continuous_scale="Greens",
                                                labels=dict(x="Predicted", y="Actual", color="Rows"),
                                                title="Confusion matrix")), width="stretch")
            counts = pd.concat([cp["Actual Category"].value_counts().rename("Actual"),
                                cp["Predicted Category"].value_counts().rename("Predicted")], axis=1
                               ).reindex(order).fillna(0).reset_index(names="Category").melt(
                id_vars="Category", var_name="Source", value_name="Rows")
            c2.plotly_chart(style_fig(px.bar(counts, x="Category", y="Rows", color="Source", barmode="group",
                                             color_discrete_sequence=["#1F5A4B", "#E0A526"],
                                             title="Category counts")), width="stretch")


# ----------------------------------------------------------------------------- Performance
METRIC_INFO = {
    "Random Forest Regressor": {
        "MAE": "Average size of the prediction error, in the same scale as wastage. Lower is better.",
        "MSE": "Average of squared errors. Punishes large mistakes more. Lower is better.",
        "RMSE": "Square root of MSE, back in wastage scale. Lower is better.",
        "R2": "Share of the variation in wastage the model explains. 1.0 is perfect.",
    },
    "Random Forest Classifier": {
        "Accuracy": "Share of test rows given the correct category.",
        "Precision": "When the model predicts a category, how often it is right (weighted average over categories).",
        "Recall": "Of the rows truly in a category, how many the model finds (weighted average).",
        "F1 Score": "Balance of precision and recall (weighted average).",
    },
}


def page_performance():
    st.title("Model Performance")
    metrics = require_csv(C.METRICS_PATH)
    if metrics is None:
        return
    st.markdown(
        "<div class='note'>These are <b>test-set</b> results from <code>model_metrics.csv</code>: "
        "a 20% hold-out split (random_state=42) of the dataset, as coded in the notebook. "
        "They are not training scores.</div>", unsafe_allow_html=True)
    for model, infos in METRIC_INFO.items():
        st.subheader(model)
        cols = st.columns(len(infos))
        for col, (metric, desc) in zip(cols, infos.items()):
            row = metrics[(metrics["Model"] == model) & (metrics["Metric"] == metric)]
            with col.container(border=True):
                st.metric(metric, f"{row['Value'].iloc[0]:.4f}" if len(row) else "Not available")
                st.caption(desc)
    with st.expander("Raw metrics table"):
        st.dataframe(metrics, hide_index=True, width="stretch")
    st.caption("`model_metrics.csv` holds only the final Random Forest models, so no model comparison table is shown.")

    st.subheader("Model insights: feature importance")
    models = get_models()
    imp = feature_importance(models)
    c1, c2 = st.columns(2)
    for col, name in zip((c1, c2), ("Regression", "Classification")):
        d = imp[imp["Model"] == name].sort_values("Importance")
        col.plotly_chart(style_fig(px.bar(d, x="Importance", y="Feature", orientation="h",
                                          title=f"{name} model", color_discrete_sequence=["#1F5A4B"]), 420),
                         width="stretch")
    st.caption("Importances are read from the saved Random Forests (impurity-based) and summed across "
               "one-hot columns for each original input. They show which inputs the model relied on, "
               "not what causes waste.")


# ----------------------------------------------------------------------------- Recommendations
def page_recommendations():
    st.title("Recommendations")
    st.write("Guidance by predicted waste category. Run a prediction to get advice tailored to your inputs.")
    from utils.recommendations import BASE
    cols = st.columns(3)
    for col, cat in zip(cols, C.CATEGORY_ORDER):
        with col.container(border=True):
            st.markdown(f"<span class='badge' style='background:{C.CATEGORY_COLORS[cat]}'>{cat}</span>",
                        unsafe_allow_html=True)
            for r in BASE[cat]:
                st.markdown(f"- {r}")
    last = st.session_state.get("last_prediction")
    st.subheader("Your latest prediction")
    if not last:
        st.info("No prediction yet. Use **Food Waste Prediction** first.")
    else:
        res, vals = last["result"], last["values"]
        st.markdown(f"Predicted category: **{res['category']}** (predicted amount {res['predicted_amount']:.1f})")
        recs = build_recommendations(res["category"], vals, dataset)
        for r in recs["core"] + recs["context"]:
            st.markdown(f"- {r}")
    st.markdown("<div class='note'>Recommendations are guidance, not guaranteed outcomes.</div>",
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- About
def page_about():
    st.title("About Project")
    st.write("A capstone data-mining project on restaurant and event food wastage. "
             "This app wraps the trained models and output files from the original notebook.")
    st.subheader("Models")
    st.markdown(
        "- **Regression:** Random Forest Regressor (200 trees) predicting `Wastage Food Amount`.\n"
        "- **Classification:** Random Forest Classifier (200 trees, balanced class weights) predicting "
        "Low, Medium or High. Categories were created by splitting wastage into three equal-sized groups "
        "(`pd.qcut`, q=3), so there is no separate probability threshold.\n"
        "- **Preprocessing** (median/most-frequent imputation, scaling, one-hot encoding) is inside each saved pipeline.\n"
        "- **Clustering:** K-Means, hierarchical and DBSCAN, supplied as result files.")
    st.subheader("Inputs the models use")
    st.write(", ".join(f"`{c}`" for c in get_models().schema.features))
    st.subheader("Known limitations")
    st.markdown(
        "- The unit of `Wastage Food Amount` is not stated in the notebook or dataset.\n"
        "- Metrics come from a single train/test split, with no cross-validation.\n"
        "- The models only know the categories and ranges in the supplied dataset.\n"
        "- Models require scikit-learn 1.6.1, the version used to save them.")


{"Home / Dashboard": page_home, "Food Waste Prediction": page_prediction,
 "Waste Analysis": page_analysis, "Model Performance": page_performance,
 "Recommendations": page_recommendations, "About Project": page_about}[page]()
