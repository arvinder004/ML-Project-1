# Student Exam Performance Predictor

Predict a student's mathematics score from demographic information and reading/writing scores. The project bundles the entire workflow—EDA notebooks, modular training code, experiment artifacts, and a lightweight Flask UI for inference—so you can both reproduce the model and deploy it quickly.

## Tech Snapshot
- Core: Python 3.10+, scikit-learn, CatBoost, XGBoost
- Serving: Flask + HTML templates
- Experiment tracking: persisted artifacts in `artifacts/`, training traces in `catboost_info/`, structured logs in `logs/`
- Packaging: `setup.py`, reusable `src/` modules

## Repository Structure
```
ML-Project-1/
├── app.py                     # Flask entrypoint exposing UI + inference route
├── artifacts/                 # Persisted datasets, trained model, preprocessor
├── catboost_info/             # Auto-generated CatBoost training diagnostics
├── logs/                      # Time-stamped log folders per training run
├── notebook/                  # EDA + model training notebooks (experimentation)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Loads raw CSV, creates train/test splits
│   │   ├── data_transformation.py   # Builds preprocessing pipelines
│   │   └── model_trainer.py         # Trains + tunes regressors, saves best model
│   ├── pipeline/
│   │   ├── predict_pipeline.py      # Runtime inference helpers (CustomData, PredictPipeline)
│   │   └── train_pipeline.py        # Placeholder for orchestration (extend as needed)
│   ├── utils.py               # Shared helpers (persist objects, model eval)
│   ├── exception.py           # Custom exception wrapper with context enrichment
│   └── logger.py              # Centralized logging configuration
├── templates/                 # HTML for landing & prediction forms
├── requirements.txt
├── setup.py
└── README.md
```

## Data & Feature Schema
- **Source**: `notebook/data/stud.csv` (copied to `artifacts/data.csv`)
- **Target**: `math_score`
- **Predictors**: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, `reading_score`, `writing_score`

Numerical fields (`reading_score`, `writing_score`) are median-imputed and standardized. Categorical fields are most-frequent imputed, one-hot encoded, and scaled without centering to keep sparse matrices efficient.

## ML Pipeline (code-first view)
1. **Data Ingestion – `DataIngestion.initiate_data_ingestion()`**  
   - Reads `notebook/data/stud.csv` into a DataFrame.  
   - Persists a raw snapshot in `artifacts/data.csv`.  
   - Splits 80/20 using `train_test_split` and writes `artifacts/train.csv` and `artifacts/test.csv`.  
   - Logging is centralized via `src/logger.py`; all errors are wrapped by `CustomeException` for better tracebacks.

2. **Data Transformation – `DataTransformation.get_data_transformer_object()` & `.initiate_data_transformation()`**  
   - Builds two `Pipeline`s (numeric + categorical) joined with a `ColumnTransformer`.  
   - Fits the transformer on training predictors, applies it to train/test, and concatenates the target column to create NumPy arrays ready for modeling.  
   - Serializes the preprocessor to `artifacts/preprocessor.pkl` using `save_object()` for reuse during inference.

3. **Model Training & Evaluation – `ModelTrainer.initiate_model_trainer()`**  
   - Splits transformed arrays into `x_train, y_train, x_test, y_test`.  
   - Defines a model zoo (RandomForest, GradientBoosting, CatBoost, XGBoost, etc.) plus search grids.  
   - `evaluate_models()` (from `src/utils.py`) performs `GridSearchCV` per model, updates parameters, and records R² on the holdout set.  
   - Chooses the highest-scoring model, enforces a minimum 0.60 R², saves it to `artifacts/model.pkl`, and returns the final score.

4. **Predict Pipeline – `PredictPipeline.predict()`**  
   - Loads `model.pkl` and `preprocessor.pkl`, transforms incoming features, and outputs a predicted math score.  
   - `CustomData` wraps form inputs into a pandas DataFrame so the same preprocessing path can be reused.

## Web Application
- **`/`** (`index.html`): Landing page describing the tool.
- **`/predictdata`** (`home.html`): Form that collects categorical selections and reading/writing scores; POST submits to Flask, which builds a `CustomData` object and renders the prediction inline.
- The Flask server in `app.py` binds everything together. Run it locally for interactive scoring or deploy behind any WSGI server.

## Notebooks & Artifacts
- `notebook/EDA_STUDENT_PERFORMANCE.ipynb`: Exploratory analysis and visualization of the dataset.
- `notebook/MODEL_TRAINING.ipynb`: Prototype training runs before codifying the flow under `src/components/`.
- `catboost_info/` + `logs/`: Automatically created during training to keep error curves, ETA estimates, and structured logs for post-mortems.

## Getting Started
```bash
# 1. Create & activate a virtual environment (example)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. (Optional) Regenerate train/test splits + model
python -m src.components.data_ingestion   # runs ingestion, transformation, trainer via __main__

# 4. Launch the web app
python app.py
# -> visit http://localhost:8000/
```

## Retraining Workflow
1. Update or replace `notebook/data/stud.csv` with new labeled data.
2. Run the ingestion script (step 3 above) to refresh `train.csv`, `test.csv`, the preprocessor, and the trained model.
3. Inspect `logs/<timestamp>.log` for run metadata and `catboost_info/` for CatBoost-specific diagnostics.
4. Restart the Flask server so inference picks up the new artifacts.

## Extending the Project
- **Automate orchestration**: implement `src/pipeline/train_pipeline.py` to chain ingestion → transformation → training with CLI arguments or a scheduler.
- **Model registry**: push `artifacts/model.pkl` to S3, MLflow, or DVC for versioning.
- **Monitoring**: log inference requests/responses and compare with ground truth when available to detect drift.

---