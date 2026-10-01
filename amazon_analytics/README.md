# 🛒 Amazon Sales & Customer Purchase Prediction

An end-to-end data analytics and machine learning project built with **Streamlit**.

## 📁 Project Structure

```
amazon_analytics/
├── app.py                                      # Main Streamlit application
├── requirements.txt                            # Python dependencies
├── Amazon_Sales_Customer_Purchase_1000.csv     # Dataset
└── README.md                                   # This file
```

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the app**
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`

---

## 📋 Analysis Steps Covered

| Step | Description |
|------|-------------|
| 1 | Collect and load the dataset |
| 2 | Check for missing and incorrect values |
| 3 | Data cleaning and preprocessing |
| 4 | Calculate total sales and revenue |
| 5 | Analyze sales by product and category |
| 6 | Analyze customer purchasing behavior |
| 7 | Charts and visualizations |
| 8 | Identify important sales trends |
| 9 | Prepare data for machine learning |
| 10 | Train classification models (Random Forest + Logistic Regression) |
| 11 | Predict whether a customer will purchase (live form) |
| 12 | Evaluate with accuracy, precision, recall, F1, confusion matrix, ROC |
| 13 | Business insights and recommendations |

---

## 🤖 ML Models Used

- **Random Forest Classifier** — Primary model
- **Logistic Regression** — Baseline comparison

## 📊 Dataset Columns

`Customer_ID`, `Age`, `Gender`, `City`, `State`, `Product_Category`, `Product_Name`,
`Price`, `Quantity`, `Discount_Percent`, `Payment_Method`, `Device`, `Customer_Rating`,
`Previous_Purchases`, `Website_Visits`, `Cart_Additions`, `Time_on_Site_Min`,
`Delivery_Days`, `Return_Status`, `Purchase_Status`, `Purchase_Probability`, `Sales_Amount`
