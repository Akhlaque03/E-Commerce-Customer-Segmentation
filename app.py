import pandas as pd
import streamlit as st
import joblib


# Load Saved Model and Files
gmm_model = joblib.load("gmm_tuned_model.pkl")
scaler = joblib.load("scaler.pkl")
rfm_features = joblib.load("rfm_features.pkl")
cluster_segment_mapping = joblib.load("cluster_segment_mapping.pkl")
gmm_tuned_final_profile = joblib.load("gmm_tuned_final_profile.pkl")


# Page Configuration
st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# Sidebar Header
st.sidebar.header("E-Commerce Customer Segmentation")


# Recency Input
recency = st.sidebar.number_input(
    "Recency",
    min_value=0,
    value=30,
    step=1
)


# Frequency Input
frequency = st.sidebar.number_input(
    "Frequency",
    min_value=1,
    value=5,
    step=1
)


# Monetary Input
monetary = st.sidebar.number_input(
    "Monetary",
    min_value=0.0,
    value=1000.0,
    step=10.0
)


# Prediction Button
predict_button = st.sidebar.button("Predict Customer Segment")


# Prediction Setup
prediction = None
prediction_label = None


# Create RFM Input Data
if predict_button:

    input_data = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }

    input_df = pd.DataFrame([input_data])


    # Scale Input Data
    input_scaled = scaler.transform(
        input_df[rfm_features]
    )


    # Predict Customer Cluster
    prediction = gmm_model.predict(input_scaled)


    # Convert Cluster to Customer Segment
    prediction_label = cluster_segment_mapping[prediction[0]]


# Header
st.title("👥 E-Commerce Customer Segmentation")

st.caption(
    "An end-to-end machine learning application for segmenting "
    "customers using RFM analysis and a tuned Gaussian Mixture Model."
)


# Prediction Result
left, right = st.columns([1.3, 1])

with left:

    st.subheader("Customer Segment Prediction")

    if prediction is not None:

        if prediction_label == "High-Value / Loyal Customers":
            st.success("🟢 High-Value / Loyal Customer")
        else:
            st.error("🔴 Low-Value / At-Risk Customer")

        st.warning("Model Used: GMM Tuned")

    else:

        st.info(
            "Enter RFM values from the sidebar and click "
            "Predict Customer Segment."
        )


# Deployed Model Information
with right:

    st.subheader("Deployed Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "GMM Tuned"
        )

    with col2:
        st.metric(
            "Clusters",
            "2"
        )

    with col3:
        st.metric(
            "Features",
            "3"
        )


# Segment Explanation and Business Recommendation
if prediction is not None:

    st.divider()

    explanation_col, recommendation_col = st.columns(2)

    with explanation_col:

        st.subheader("Segment Explanation")

        if prediction_label == "High-Value / Loyal Customers":

            st.success(
                "Strong purchasing value and loyalty based on RFM characteristics."
            )

        else:

            st.warning(
                "Lower engagement and customer value, indicating a potential risk of churn."
            )

    with recommendation_col:

        st.subheader("Business Recommendation")

        if prediction_label == "High-Value / Loyal Customers":

            st.info(
                "Use personalized discounts, win-back campaigns, product, "
                "recommendations, and repeat-purchase reminders."
            )

        else:

            st.info(
                "Use personalized discounts, win-back campaigns, "
                "product recommendations, and repeat-purchase reminders."
            )


# Selected Customer RFM Scenario
st.divider()

st.subheader("Selected Customer RFM Scenario")

scenario_df = pd.DataFrame({
    "Features": [
        "Recency",
        "Frequency",
        "Monetary"
    ],
    "Value": [
        recency,
        frequency,
        monetary
    ]
})

st.dataframe(
    scenario_df,
    use_container_width=True,
    hide_index=True
)




# Baseline Clustering Model Comparison
baseline_df = pd.DataFrame({
    "Model": [
        "K-Means",
        "Hierarchical Clustering",
        "DBSCAN",
        "GMM"
    ],
    "Silhouette Score": [
        0.509691,
        0.465100,
        0.016987,
        0.372065
    ],
    "Calinski-Harabasz Index": [
        6414.374820,
        5712.805553,
        689.726561,
        3312.856627
    ],
    "Davies-Bouldin Index": [
        0.668082,
        0.739640,
        2.411384,
        1.050568
    ]
})

baseline_df = baseline_df.sort_values(
    by="Silhouette Score",
    ascending=False
).reset_index(drop=True)

st.subheader("Baseline Clustering Model Performance")

st.dataframe(
    baseline_df,
    use_container_width=True,
    hide_index=True
)


# Baseline Model Performance Visualization
import matplotlib.pyplot as plt
# Sort by Silhouette Score — highest to lowest
plot_df = baseline_df.sort_values(
    by="Silhouette Score",
    ascending=False
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 7))

colors = ["#4F46E5"] * len(plot_df)

# Highest best model
best_index = plot_df["Silhouette Score"].idxmax()

colors[best_index] = "#059669"

bars = ax.bar(
    plot_df["Model"],
    plot_df["Silhouette Score"],
    color=colors,
    edgecolor="#111827",
    linewidth=1.2,
    width=0.72
)

ax.set_title(
    "Baseline Clustering Model Performance Comparison (Silhouette Score)",
    fontsize=18,
    fontweight="bold",
    color="#111827",
    pad=22
)

ax.text(
    0.5,
    1.02,
    "Higher Silhouette Score indicates better-defined and well-separated clusters",
    transform=ax.transAxes,
    ha="center",
    fontsize=12,
    color="#64748B",
    style="italic"
)

ax.set_xlabel(
    "Clustering Model",
    fontsize=12,
    fontweight="bold",
    color="#111827"
)

ax.set_ylabel(
    "Silhouette Score",
    fontsize=12,
    fontweight="bold",
    color="#111827"
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.25
)

ax.set_axisbelow(True)

ax.tick_params(
    axis="x",
    rotation=0,
    labelsize=10
)

ax.tick_params(
    axis="y",
    labelsize=10
)

# Remove top/right border
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

# Top values
for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.006,
        f"{height:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#475569"
    )

ax.set_ylim(0, 0.75)

fig.tight_layout()

st.pyplot(fig)

plt.close(fig)



# Tuned Clustering Model Comparison
tuned_df = pd.DataFrame({
    "Model": [
        "GMM Tuned",
        "Hierarchical Tuned",
        "K-Means Tuned",
        "DBSCAN Tuned"
    ],
    "Silhouette Score": [
        0.524530,
        0.520767,
        0.509691,
        0.303232
    ],
    "Calinski-Harabasz Index": [
        4362.054862,
        4140.285687,
        6414.374820,
        15.281160
    ],
    "Davies-Bouldin Index": [
        0.691640,
        0.702080,
        0.668082,
        0.752984
    ]
})

tuned_df = tuned_df.sort_values(
    by="Silhouette Score",
    ascending=False
).reset_index(drop=True)

st.subheader("Tuned Clustering Model Performance")

st.dataframe(
    tuned_df,
    use_container_width=True,
    hide_index=True
)



# Tuned Model Performance Visualization
import matplotlib.pyplot as plt

plot_df = tuned_df.copy()

plt.figure(figsize=(10, 6))

colors = ["#F59E0B"] + ["#7C3AED"] * (len(plot_df) - 1)

bars = plt.bar(
    plot_df["Model"],
    plot_df["Silhouette Score"],
    color=colors,
    edgecolor="#111827",
    linewidth=1.2,
    width=0.62
)

plt.title(
    "Final Tuned Model Performance Comparison",
    fontsize=18,
    fontweight="bold",
    color="#111827",
    pad=20
)

plt.text(
    0.5,
    1.02,
    "Higher Silhouette Score indicates better cluster separation and cohesion",
    transform=plt.gca().transAxes,
    ha="center",
    fontsize=11,
    color="#64748B",
    style="italic"
)

plt.xlabel(
    "Clustering Model",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Silhouette Score",
    fontsize=12,
    fontweight="bold"
)

plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.25
)

plt.xticks(
    fontsize=10,
    fontweight="bold"
)

plt.yticks(
    fontsize=10
)

ax = plt.gca()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.006,
        f"{height:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#475569"
    )

plt.tight_layout()

st.pyplot(plt.gcf())

plt.close()



# Cluster / Segment Profile
st.subheader("Customer Segment Profile")

st.dataframe(
    gmm_tuned_final_profile[
        [
            "Segment",
            "Customer_Count",
            "Avg_Recency",
            "Avg_Frequency",
            "Avg_Monetary"
        ]
    ],
    use_container_width=True,
    hide_index=True
)



# Customer Segment Distribution Visualization
plot_df = gmm_tuned_final_profile.copy()

plt.figure(figsize=(10, 6))

colors = ["#059669", "#F59E0B"]

bars = plt.bar(
    plot_df["Segment"],
    plot_df["Customer_Count"],
    color=colors,
    edgecolor="#111827",
    linewidth=1.2,
    width=0.62
)

plt.title(
    "Customer Segment Distribution",
    fontsize=18,
    fontweight="bold",
    color="#111827",
    pad=20
)

plt.text(
    0.5,
    1.02,
    "Customer distribution across the final GMM segments",
    transform=plt.gca().transAxes,
    ha="center",
    fontsize=11,
    color="#64748B",
    style="italic"
)

plt.xlabel(
    "Customer Segment",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Customer Count",
    fontsize=12,
    fontweight="bold"
)

plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.25
)

plt.xticks(
    fontsize=10,
    fontweight="bold"
)

plt.yticks(
    fontsize=10
)

ax = plt.gca()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + (height * 0.01),
        f"{int(height):,}",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#475569"
    )

plt.tight_layout()

st.pyplot(plt.gcf())

plt.close()