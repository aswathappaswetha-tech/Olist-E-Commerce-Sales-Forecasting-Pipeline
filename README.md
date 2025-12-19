## Olist E‑Commerce Sales Forecasting Pipeline
End‑to‑end time series forecasting pipeline built using Python, Prophet, and Tableau, designed to predict daily e‑commerce sales for the Olist marketplace.
This project demonstrates a production‑style workflow with modular code, clean ETL, automated forecasting, and interactive BI dashboards.

##  Executive Summary
E‑commerce businesses rely heavily on accurate demand forecasting to optimize inventory, logistics, staffing, and marketing.
This project builds a fully modular forecasting pipeline that:
- Ingests and cleans raw Olist marketplace data
- Engineers time‑series features
- Trains a Prophet forecasting model
- Evaluates performance using industry‑standard metrics
- Exports forecast outputs for visualization
- Powers an interactive Tableau dashboard for business insights
The result is a scalable, reproducible forecasting system suitable for real‑world analytics workflows

## Architecture Overview

flowchart LR
    A[Raw Olist Data] --> B[Ingestion]
    B --> C[Cleaning & Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Prophet Modeling]
    E --> F[Evaluation]
    F --> G[Export Forecasts
    G --> H[Tableau Dashboard]


##  Dataset: Olist Brazilian E‑Commerce Public Dataset
This project uses the well‑known Olist dataset, containing multiple relational tables:
orders : Order -level details with timestamps 
order_items : product-level details per order
products : Prodeuct category metadata
customers : Customer demographies
sellers : Seller information
payments : Payment methods and values
reviews : Customer review scores
The pipeline aggregates these into a daily sales time series.

##  Pipeline Components
1. Ingestion
- Load raw CSVs
- Merge relational tables
- Create unified order-level dataset
2. Cleaning & Preprocessing
- Handle missing values
- Remove duplicates
- Convert timestamps
- Standardize column formats
3. Feature Engineering
- Aggregate daily sales
- Create lag features
- Rolling averages
- Optional: holiday effects
4. Modeling (Prophet)
- Train/test split
- Prophet model with:
- Trend
- Weekly seasonality
- Yearly seasonality
- Hyperparameter tuning
- Forecast generation
5. Evaluation
- Metrics:
- MAE
- RMSE
- MAPE
- Baseline vs Prophet comparison
6. Export
- Save forecast results to /export/
- Output CSV used in Tableau dashboard

📊 Tableau Dashboard
The dashboard visualizes:
- Historical sales trends
- Forecasted sales
- Seasonality patterns
- Category-level insights
- Interactive filters
👉 Dashboard Link: (Tableau Public link here)
👉 Screenshots: <img width="1487" height="822" alt="image" src="https://github.com/user-attachments/assets/ad351136-c418-457f-9746-67d01012a51a" />



🛠 Tech Stack
- Python
- pandas, NumPy
- Prophet
- Matplotlib / Seaborn
- Tableau
- Git & Terminal workflow

▶️ How to Run the Pipeline
1. Clone the repository
git clone https://github.com/aswathappaswetha-tech/Olist-E-Commerce-Sales-Forecasting-Pipeline
cd Olist-E-Commerce-Sales-Forecasting-Pipeline


2. Install dependencies
pip install -r requirements.txt


3. Run the pipeline
python src/main.py


4. View outputs
- Cleaned data → /data/processed/
- Forecast results → /export/forecasts.csv
- Tableau-ready dataset → /export/tableau/

📁 Folder Structure
Olist-E-Commerce-Sales-Forecasting-Pipeline/
│
├── src/
│   ├── ingestion/
│   ├── cleaning/
│   ├── feature_engineering/
│   ├── modeling/
│   ├── evaluation/
│   ├── export/
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── dashboard/
│   └── tableau_files/
│
├── README.md
├── requirements.txt
└── .gitignore



📈 Business Insights
Key insights derived from the Olist dataset:
- Strong weekly and yearly seasonality
- Sales spikes around holidays
- Long-tail distribution of product categories
- Forecasts help optimize:
- Inventory planning
- Delivery logistics
- Marketing campaigns
- Seller performance management

🔮 Future Improvements
- Add ARIMA / XGBoost forecasting models
- Build automated retraining pipeline
- Add Docker containerization
- Deploy API endpoint for real-time forecasts
- CI/CD with GitHub Actions











    
