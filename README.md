# 🛒 Olist Behaviour Segmentation & Revenue Prediction Analysis

This project analyzes **customer behaviour** from the Olist e-commerce dataset, segments customers using **RFML (Recency, Frequency, Monetary, Loyalty)** metrics, and predicts their **lifetime contribution** to the platform.

The project is divided into **two main objectives**:
1. **Customer Segmentation** – Determine which cluster a customer belongs to, along with cluster insights and visualization.
2. **Revenue Prediction** – Predict a customer’s lifetime contribution (total sales) based on their profile and activity.

---

## 📊 Project Links

- **🔗 Streamlit App (Model Deployment)**: [Click to Open](https://olist-behaviour-segmentation-and-revenue-prediction-analysis.streamlit.app/)
- **📈 Tableau Dashboard**: [Click to Open](https://public.tableau.com/views/OlistCustomerSegmentationDashboard/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- **💻 Google Colab (Model Training)**: [Click to Open](https://colab.research.google.com/drive/1-BfOm2Yw1gpZ1Txl2iHPUdp7dZweSvWv?usp=sharing)
- **📑 Report Deck**: [Click to Open](https://drive.google.com/file/d/1-wcBygXqqV9TBxMwN9F6EDFEvn0_nvCs/view?usp=drive_link)

---

## 🛠 Features in Streamlit App

- **Choose Your Objective**:
  - **Find Your Segment**: See which customer cluster you belong to.
  - **Predict Contribution**: Estimate your lifetime contribution in total sales.
- **Cluster Visualizations**: Shows a visual representation of your assigned cluster.
- **User-Friendly Inputs**: Simply enter your details (purchase dates, frequency, total spend, etc.).

---

## 🧠 Methods & Models

- **Data Preprocessing**: Custom transformer (`RFMLBuilder`) to calculate RFML metrics.
- **Segmentation Model**: Gaussian Mixture Model (GMM) with scaling.
- **Prediction Model**: Regression model predicting total sales.
- **Feature Engineering**: Recency, Frequency, Monetary, Loyalty metrics.

---

## 📦 Installation & Usage

**1️⃣ Clone the repository**
```bash
git clone https://github.com/rizkyprtr2209/olist-behaviour-segmentation-and-revenue-prediction-analysis.git
cd olist-behaviour-segmentation-and-revenue-prediction-analysis
```

**2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

**3️⃣ Run locally**
```bash
streamlit run app.py
```

## 📌 Author

**Rizky Putra Laksmana**
- 📧 Email: [rizkypl2209@gmail.com](mailto:rizkypl2209@gmail.com)  
- 🔗 LinkedIn: [linkedin.com/in/rizkyputral22](https://www.linkedin.com/in/rizkyputral22)
