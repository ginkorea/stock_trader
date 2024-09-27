from transformer.pipeline import Pipeline

pipeline = Pipeline(start_date="2022-01-01", end_date="2022-10-01", window_size=30, pred_days=1)
pipeline.train_model(learning_rate=0.001, batch_size=32, epochs=50)
