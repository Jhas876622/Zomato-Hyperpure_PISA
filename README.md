# 🍅 PISA — Predictive Inventory & Spoilage Alert System
### Zomato Hyperpure B2B — Perishable Wastage Reduction
**B.Tech IT | Product Analyst Portfolio Project**

---

## 📌 Problem Statement
Zomato Hyperpure operates a 1P (inventory-led) B2B supply chain serving 1 lakh+
restaurant partners. As the inventory owner, every kg of spoiled perishable produce
is a **100% direct P&L loss** — making wastage reduction the highest-ROI product
problem in a low-margin B2B business.

## 🎯 Solution: PISA
A 3-engine ML decision system that:
1. **Forecasts demand** (SKU-level, 14-day horizon) → prevents over-ordering
2. **Predicts spoilage risk** (lot-level, real-time) → enables early action
3. **Recommends optimal orders** (Newsvendor model) → minimises total cost

**Projected Impact:** ~35% relative reduction in perishable wastage rate

---

## 🗂️ Project Structure
```
hyperpure_pisa/
├── config.py           # SKU master data, festival calendar, constants
├── data_generator.py   # Generates synthetic but realistic supply chain data
├── ml_models.py        # ML model training (Random Forest + Gradient Boosting)
├── app.py              # Streamlit dashboard (5 tabs)
├── requirements.txt    # Python dependencies
├── data/               # Generated CSV files (auto-created on first run)
└── models/             # Saved ML model files (auto-created on first run)
```

---

## 🚀 Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Generate data + train models manually
python data_generator.py
python ml_models.py

# 3. Launch the dashboard
streamlit run app.py
```
> **Note:** The dashboard auto-generates data and trains models on first launch.
> You don't need to run steps 2 separately.

---

## 📊 Dashboard Tabs

| Tab | What it shows | Interview talking point |
|-----|---------------|------------------------|
| 🏠 Live Overview | KPI cards, wastage by category, monthly trends | North Star metric + Guardrail |
| 📈 Demand Forecast | 14-day SKU forecast with confidence bands | GBM + lag features, MAPE |
| 🚨 Spoilage Alerts | Traffic-light lot risk table, immediate actions | Random Forest, risk score |
| 📦 Inventory Optimizer | Newsvendor order recommendations | Critical ratio, cost trade-off |
| 🤖 Model Health | MAPE, F1, confusion matrix, feature importance | Model evaluation, bias monitoring |

---

## 🤖 ML Models Used

### 1. Spoilage Risk Classifier (Random Forest)
- **Target:** Will this lot spoil? (binary)
- **Key features:** Age % of shelf life, temperature deviation, category, over-order factor
- **Why RF?** Handles mixed features, robust to outliers, interpretable feature importance

### 2. Demand Forecasting (Gradient Boosting)
- **Target:** Daily demand per SKU per warehouse (regression)
- **Key features:** Lag (1,2,3,7,14 days), rolling mean/std, day-of-week, month, weather, festival multiplier
- **Why GBM?** Captures non-linear effects (festivals, weather spikes) better than ARIMA

### 3. Order Quantity Optimizer (Newsvendor Model)
- **Critical ratio** = Cu / (Cu + Co) = 0.25 / 1.25 = **0.20**
- Orders at the 20th percentile of the demand distribution
- Explicitly balances spoilage cost (100%) vs stockout cost (25%)

---

## 📈 Key Metrics (KPIs)

| Tier | Metric | Definition |
|------|--------|------------|
| ⭐ North Star | Perishable Wastage % | Spoiled value ÷ handled value |
| 🛡️ Guardrail | Fill Rate / Stockout Rate | Never optimise wastage by starving stock |
| ⚙️ Efficiency | Inventory Turnover Ratio | COGS ÷ avg inventory |
| 🤖 Model Health | MAPE / F1 / Precision-Recall | Forecast accuracy + alert quality |
| 💰 Business | ₹ Wastage Recovered | Direct P&L impact |

---

## 🔮 Future Enhancements 
1. **LightGBM / DeepAR** for multi-SKU global forecasting models
2. **Cox Survival Analysis** for more precise shelf-life prediction
3. **IoT Integration** — real-time cold-chain sensor data
4. **Dynamic Discounting** — auto-trigger discounts on near-expiry lots
5. **Supplier Scorecards** — quality/reliability tracking fed back to procurement

---
