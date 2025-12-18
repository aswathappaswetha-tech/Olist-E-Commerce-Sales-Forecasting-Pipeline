from src.ingestion.load_data import load_olist_data
from src.cleaning.clean_data import clean_olist_data
from src.features.feature_engineering import build_features
from src.modeling.prophet_model import run_prophet_pipeline
from src.evaluation.evaluate import run_evaluation_plots

print("Loading raw data...")
df = load_olist_data()

print("Cleaning data...")
df_clean = clean_olist_data(df)

print("Building features...")
df_features = build_features(df_clean)

print("Running forecasting model...")
forecast, metrics = run_prophet_pipeline(df_features)

print("Generating evaluation plots...")
test_df = df_features.tail(60)
run_evaluation_plots(test_df, forecast)

print("\nPipeline executed successfully.")
print("Evaluation metrics:", metrics)
