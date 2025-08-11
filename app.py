import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
from Class import RFMLBuilder, ClusterModel, RFMLPredictor  # Import your custom classes
import joblib

# === Load Pipelines ===
with open("Safe Cleaning Pipe.joblib", "rb") as f:
    safe_clean_pipe = joblib.load(f)

with open("Cluster Pipe.joblib", "rb") as f:
    cluster_pipe = joblib.load(f)

with open("Best Predictor Pipe.joblib", "rb") as f:
    predictor_pipe = joblib.load(f)

# === Cluster Descriptions & Images ===
cluster_info = {
    0: {
        "name": "Low-Value Inactive",
        "desc": "_'It’s been ages since I last bought from Olist — maybe just once or twice ever. I might come back for the right deal.'_",
        "business": "High Recency, Low Frequency, Low Monetary, Zero Loyalty. At-risk / churned customers — re-engage with urgency-driven win-back campaigns and low-barrier incentives.",
        "what_you_should_get": [
            "Win-back campaigns with urgent, time-limited deals",
            "Suggestions tailored to what you once bought",
            "Fun incentives like double points if you return soon",
            "‘Welcome back’ offers that make trying again easy"
        ],
        "image": "images\LV-removebg-preview.png"
    },
    1: {
        "name": "One-Time High Spenders",
        "desc": "_'I came for one big purchase and spent a lot, but I haven’t been back. I liked the experience but need a reason to return.'_",
        "business": "High Recency, Low Frequency, High Monetary, Zero Loyalty. Conversion potential — win-back with targeted bundles, reminders, and premium post-purchase offers.",
        "what_you_should_get": [
            "Personalized bundles or add-on suggestions based on your big purchase",
            "Exclusive post-purchase deals to keep you engaged",
            "Friendly reminders for product upgrades or refills",
            "VIP-style communication and offers for premium buyers"
        ],
        "image": "images\OHS-removebg-preview.png"
    },
    2: {
        "name": "Engaged Regulars",
        "desc": "_'I’m a steady Olist customer. I don’t spend the most, but I’ve made several purchases and I like coming back.'_",
        "business": "Lower Recency, Moderate Frequency, Moderate Monetary, High Loyalty. Growth opportunity — nurture with loyalty programs, upselling, and community engagement.",
        "what_you_should_get": [
            "Tiered loyalty rewards that increase with your purchases",
            "‘Spend more, save more’ promotions to encourage upselling",
            "Invitations to product previews or community events",
            "Special referral bonuses for bringing in friends"
        ],
        "image": "images\ER-removebg-preview.png"
    },
    3: {
        "name": "VIP Customers",
        "desc": "_'I’m one of Olist’s most loyal shoppers. I buy often, I’ve made many purchases, and I’m happy to spend more here than anywhere else.'_",
        "business": "Moderate Recency, Highest Frequency, Very High Monetary, Moderate Loyalty. Core revenue drivers — retain through exclusives, premium service, and upselling.",
        "what_you_should_get": [
            "Priority customer service with fast response times",
            "Exclusive early access to new collections",
            "Special VIP-only promotions and experiences",
            "Personalized suggestions to enhance your shopping"
        ],
        "image": "images\VIP-removebg-preview.png"
    },
    4: {
        "name": "Moderate Spenders",
        "desc": "_'I shop here occasionally. I like Olist but usually only buy when I see something I need or a promotion that interests me.'_",
        "business": "High Recency, Moderate Frequency, Moderate Monetary, Zero Loyalty. Reactivation target — use personalized offers, seasonal promotions, and event-based triggers.",
        "what_you_should_get": [
            "Seasonal sale alerts tailored to your taste",
            "Personalized offers on categories you’ve browsed",
            "Gamified shopping perks to make buying more fun",
            "Bonus points or discounts to encourage more purchases"
        ],
        "image": "images\MS-removebg-preview.png"
    }
}

# === Streamlit UI ===

st.set_page_config(page_title="Customer Segmentation & Contribution Predictor", layout="centered")

st.title("🛒 Olist Customer Behaviour Segmentation & Revenue Prediction")

st.markdown("""
This interactive tool lets you:
1. **Identify your customer segment** using Recency, Frequency, Monetary, and Loyalty (RFML) metrics.
2. **Predict your estimated lifetime contribution** (total sales) to the platform.

---

### 📌 Business Context
Customer retention is more cost-effective than acquisition.  
By segmenting customers based on RFML metrics, businesses can **target the right group with the right strategy** —  
increasing loyalty, maximizing lifetime value, and reducing churn.

---

### 📊 Dataset
- Based on the **Brazilian E-Commerce Public Dataset by Olist**.
- Covers **100k+ orders** from multiple marketplaces (2016–2018).
- Includes order timestamps, payment data, and delivery performance.

---

### 🔍 Methodology
- **Segmentation:** Gaussian Mixture clustering for 5 distinct customer profiles.
- **Prediction:** Regression model to estimate lifetime contribution.
- **Tools:** Python (Pandas, Scikit-Learn), Streamlit for deployment.

---

💡 *Tip:* The cluster description and contribution are shown after you run the prediction.
""")

# ---- USER INPUT FORM ----
with st.form("user_input_form"):
    st.markdown("### 📋 Your Purchase Information")

    col1, col2 = st.columns(2)

    with col1:
        first_purchase = st.date_input(
            "📅 First Purchase Date",
            help="The date when you first bought from the platform."
        )
        last_order = st.date_input(
            "📅 Last Order Date",
            help="The date of your most recent order."
        )
        last_delivery = st.date_input(
            "📅 Last Delivery Date",
            help="The date your last order was delivered."
        )

    with col2:
        frequency = st.number_input(
            "🛍 Number of Purchases",
            min_value=1, step=1,
            help="Total number of orders you have made."
        )
        payment_value = st.number_input(
            "💰 Total Spend (USD)",
            min_value=0.0, step=0.01,
            help="Total amount of money you have spent."
        )

    st.markdown("### 🎯 Choose Your Objective")
    st.write(
        "- **See My Cluster** → Find out your customer segment and its key traits.\n"
        "- **See My Contribution** → Predict your lifetime sales contribution."
    )

    col_btn1, col_btn2 = st.columns([1,3])

    with col_btn1:
        cluster_selected = st.form_submit_button(
            "🔍 See My Cluster",
            help="Identify your RFML-based customer segment."
        )

    with col_btn2:
        contribution_selected = st.form_submit_button(
            "📈 See My Contribution",
            help="Estimate your lifetime value for the company."
        )

# === Prepare DataFrame for Prediction ===
def prepare_input():
    data = pd.DataFrame({
        "first-purchase": [first_purchase],
        "order_purchase_timestamp": [last_order],
        "order_delivered_customer_date": [last_delivery],
        "frequency": [frequency],
        "payment_value": [payment_value]
    })
    return data

# === Objective 1: Predict Cluster ===
# ---- ACTIONS ----
if cluster_selected:
    user_data = prepare_input()

    # Safe cleaning & feature engineering
    processed_data = safe_clean_pipe.transform(user_data)

    # Predict cluster
    cluster = cluster_pipe.predict(processed_data)[0]
    info = cluster_info[cluster]

    st.success(f"Cluster {cluster} — {info['name']}")
    col1,col2 = st.columns([1,3])
    with col1:
        st.image(info["image"], caption=f"Cluster {cluster} Visualization", use_container_width=True)
    with col2:
        st.write(info["desc"])

        st.markdown("**Business Interpretation**")
        st.write(info["business"])

        st.markdown("**What You Should Get:**")
        for item in info["what_you_should_get"]:
            st.markdown(f"- {item}")

# === Objective 2: Predict Lifetime Contribution ===
if contribution_selected:
    user_data = prepare_input()

    # Safe cleaning & feature engineering
    processed_data = safe_clean_pipe.transform(user_data)

    # Predict contribution
    predicted_value = predictor_pipe.predict(processed_data)[0]

    st.success(f"Estimated Lifetime Contribution: **${predicted_value:,.2f}**")
    st.write("💡 This estimation is based on your purchase history and behaviour patterns.")