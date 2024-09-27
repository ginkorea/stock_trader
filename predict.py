from transformer.pipeline import Pipeline

pipeline = Pipeline(start_date="2022-10-01", end_date="2022-12-30", window_size=30, pred_days=1)

try:
    pipeline.predict()
except ValueError as e:
    print(f"Error during prediction: {e}")