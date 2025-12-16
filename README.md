# Retail Analysis Dashboard

## Project Overview
The **Retail Analysis Dashboard** is a comprehensive solution for **retail store performance forecasting and visualization**. It leverages **Python-based forecasting models** and a **Power BI executive dashboard** to provide actionable insights for management, enabling **data-driven decisions** regarding store performance, growth opportunities, and risk management.

---

## Project Objectives
- Predict store performance for January using historical data from August to December.
- Identify **Star Performers**, **High Growth**, **Consistent Laggards**, and other store categories.
- Visualize **risk vs reward** and performance metrics for executive decision-making.
- Create a professional dashboard suitable for **corporate presentations**.


---

## Data

**Input File:** `Excel File`  
Columns used for forecasting:

- **Branch**: Store branch location
- **Store_Name**: Store identifier
- **Monthly Performance**: Aug, Sep, Oct, Nov, Dec (percent values)

**Output File:** `store_forecast_output_FINAL.csv`  
Includes:

- Jan_Prediction: Forecasted performance for January
- Volatility: Standard deviation of past 5 months
- Momentum: Difference between last and first month
- Confidence: High / Medium / Low based on volatility
- Store_Category: Classification based on forecast and performance

---

## Forecasting Methodology

The forecasting pipeline uses a **weighted ensemble approach** combining three models:

1. **Linear Trend Prediction**
   - Fits a linear regression on historical monthly data.
   - Captures consistent trends.

2. **Rolling Mean Prediction**
   - Computes mean of the last 3 months.
   - Captures short-term recent performance.

3. **Holt’s Exponential Smoothing**
   - Accounts for trends and seasonality.
   - Falls back to simple exponential smoothing if model fails.

**Weighting Strategy:**
- Each model is weighted inversely by its **cross-validated RMSE**.
- Weights are adjusted using **volatility shrinkage** to avoid overreaction to unstable stores.
- **Predictions are capped** with a maximum jump (30%) to prevent unrealistic forecasts.

**Additional Metrics:**
- **Volatility:** `std(aug-dec)`  
- **Momentum:** `dec - aug`  
- **Confidence:** Low / Medium / High based on volatility  
- **Store Category:** Derived from Jan_Prediction, volatility, and momentum

---

## Power BI Dashboard

### PAGE 1 — Executive Overview 
Designed for **senior-level decision making**.

**A. KPI Cards**
- Avg Dec %
- Avg Jan %
- Forecast Delta %
- Total Stores
- Star Stores

**B. Risk vs Reward Scatter (Core Visual)**
- X-Axis: Volatility  
- Y-Axis: Jan_Prediction  
- Size: Momentum  
- Legend: Store_Category  
- Details: Store_Name  


**C. Store Category Distribution**
- Horizontal bar chart  
- Axis: Store_Category  
- Values: Count of Store_Name

**D. Branch-wise Performance**
- Clustered column chart  
- Axis: Branch  
- Values: Avg Jan %  

---

### PAGE 2 — Performance Table 
**Columns in exact order:**
1. Store_Name  
2. Branch  
3. Dec  
4. Jan_Prediction  
5. Forecast Delta %  
6. Volatility  
7. Momentum  
8. Confidence  
9. Store_Category  

**Purpose:** Full accountability and scoring of each store.

---

## How to Run the Project

### 1. Run the Forecasting Script
- Script file: `analysis.py`  

Output: outputs/store_forecast_output_FINAL.csv will be generated with the following columns:

- Jan_Prediction
- Volatility
- Momentum
- Confidence
- Store_Category
- Historical months (Aug–Dec) for reference

### 2. Open the PBIX file in Power BI Desktop to view dashboards.
- File : Assignment.pbix
  
Key Features:
- Top KPI Cards: Avg Dec %, Avg Jan %, Forecast Delta %, Total Stores, Star Stores
- Risk vs Reward Scatter
- Store Category Distribution
- Branch Performance
- Performance Table: Sorted view of all stores with forecast metrics

