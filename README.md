<div align="center">
	<img src="dashboards/dashboard.png" alt="Hospital Resource Analytics Dashboard" width="100%" />
	<h1>Hospital Resource Analytics</h1>
	<p>Forecast hospital workloads and surface operational insights with an end-to-end machine learning and business intelligence pipeline.</p>
</div>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Major Workflow](#major-workflow)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
	- [Run the ETL Pipeline](#run-the-etl-pipeline)
	- [Train the Machine Learning Models](#train-the-machine-learning-models)
	- [Launch the Streamlit App](#launch-the-streamlit-app)
- [Dashboard Preview](#dashboard-preview)
- [Data and Features](#data-and-features)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Future Enhancements](#future-enhancements)

## Project Overview
Hospital systems generate diverse operational signals every month: attendances, admissions, wait times, and follow-up activity. This project consolidates those records, cleans the data, and produces predictive analytics that support strategic planning for staffing and bed management. The repository combines a production-ready ETL process, forecasting models, and an interactive visualization surface so analysts can move from raw spreadsheets to decision-ready insights in a single workflow.

## Major Workflow
1. **Ingest** historical hospital activity files through the configurable extract layer in [etl_pipeline.py](etl_pipeline.py).
2. **Transform** and harmonize complex Excel sources, then load them into a normalized SQLite warehouse and Tableau-friendly CSV outputs.
3. **Model** emergency wait times and consultation volume trends with the experiments orchestrated in [ml_model.py](ml_model.py), covering Random Forest, Prophet, SARIMA, and LSTM techniques.
4. **Serve** cleaned data and serialized models to the Streamlit interface in [app.py](app.py), enabling automated data cleaning, model evaluation, and forward forecasting with minimal user input.
5. **Visualize** organizational trends in Tableau or other BI tools by consuming the curated outputs and the assets in [dashboards](dashboards).

## Architecture
- **ETL layer:** [etl_pipeline.py](etl_pipeline.py) ingests raw Excel and CSV files, reshapes them into long-form tables, and publishes consolidated data to [output/nhs_data.db](output/nhs_data.db) and [output/tableau_data.csv](output/tableau_data.csv).
- **Model training layer:** [ml_model.py](ml_model.py) handles cross-sectional feature engineering, resamples monthly time series, and produces persisted artifacts in [models](models) (Random Forest, Prophet, SARIMA, LSTM).
- **Interactive app:** [app.py](app.py) auto-detects date and value columns from uploaded CSVs, applies the saved models, and surfaces diagnostics such as RMSE and R² to users in a Streamlit dashboard.
- **Documentation:** [reports/system_architecture.md](reports/system_architecture.md) includes the Mermaid system diagram summarizing dependencies and data movement.

## Installation
1. Install Python 3.9 or later.
2. Clone the repository and create an isolated environment:
	 ```bash
	 git clone https://github.com/NahumDm/hospital-resource-analytics.git
	 cd hospital-resource-analytics
	 python -m venv .venv
	 source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
	 ```
3. Install the project dependencies:
	 ```bash
	 pip install --upgrade pip
	 pip install -r requirements.txt
	 ```
4. (Optional) Install Prophet and TensorFlow if they are not included in your base environment:
	 ```bash
	 pip install prophet tensorflow
	 ```

## Usage

### Run the ETL Pipeline
Generate the SQLite database and Tableau-friendly CSV from the raw Excel and CSV assets:
```bash
python etl_pipeline.py
```
Outputs are written to [output](output) and logged to the console for traceability.

### Train the Machine Learning Models
Experiment with forecasting algorithms and export trained models:
```bash
python ml_model.py
```
This script trains a Random Forest regression baseline for cross-sectional features, evaluates Prophet and SARIMA on monthly aggregates, and trains an LSTM when TensorFlow is available. Artifacts are persisted to [models](models) and can be consumed by the Streamlit application.

### Launch the Streamlit App
Run the self-service forecasting interface, upload new CSVs, and compare model performance:
```bash
streamlit run app.py
```
The app automatically infers relevant columns, triggers the selected model, visualizes the forecast, and displays error metrics.

## Dashboard Preview
The Tableau workbook built from [data/processed/ER_wait_time_dataset.csv](data/processed/ER_wait_time_dataset.csv) highlights seasonal behavior, regional comparisons, and hospital-level wait time trends.

<div align="center">
	<img src="dashboards/dashboard.png" alt="Dashboard Preview" width="85%" />
</div>

## Data and Features
- **Core inputs:** AE activity/quality Excel workbooks, consultation-level CSVs, and ER wait-time logs stored in the project root.
- **Key engineered fields:** aggregated attendances per season, quality indices by measure, and median wait times resampled to calendar months.
- **Target variables:** Total wait time (minutes) for emergency departments and APC finished consultant counts for monthly forecasting exercises.
- **Model outputs:** Feature importance rankings from Random Forest, serialized time-series models, and optional persistence baseline comparisons.

### Feature Reference
| Column | Description |
| --- | --- |
| CALENDAR_MONTH_END_DATE | Month stamp used for consultation aggregation in YYYY-MM or MMM-YY format |
| APC_Finished_Consultant | Completed consultant-led episodes (primary target) |
| APC_FCEs_with_a_procedure | Finished consultant episodes involving a procedure |
| APC_Day_Case_Episodes | Volume of day case episodes |
| APC_Emergency | Emergency admissions count |
| Outpatient_Total_Appointments | Total outpatient appointments |
| Outpatient_Percent_DNA | Did-not-attend rate, expressed as a percent |
| Total Wait Time (min) | Emergency room wait time in minutes, modelled in [ml_model.py](ml_model.py) |

## Project Structure
```
hospital-resource-analytics/
├── app.py                  # Streamlit interface for automated cleaning and forecasting
├── etl_pipeline.py         # End-to-end extract, transform, load process
├── ml_model.py             # Model training, evaluation, and artifact export
├── dashboards/             # Tableau dashboard assets and banner imagery
├── data/                   # Staging area for raw and processed datasets
├── models/                 # Serialized models consumed by the app
├── notebooks/              # Exploratory and development notebooks
├── output/                 # SQLite database and Tableau-ready CSV
├── reports/                # Architecture documentation and supporting materials
├── requirements.txt        # Python dependency pinning
├── test_app.py             # Streamlit interface tests
└── test.py                 # Additional data quality checks
```

## Testing
Unit tests cover core data validation and application logic. Execute the suite with:
```bash
pytest
```
Add new tests alongside changes to maintain reproducibility across the ETL, modeling, and application layers.

## Future Enhancements
- Automate hyperparameter search for Prophet and SARIMA to improve forecast accuracy across sites.
- Integrate model monitoring to track error drift when new hospital data is ingested.
- Package the ETL and modeling workflow as scheduled jobs (Airflow or Azure Data Factory) for production deployment.
- Extend the Streamlit UI to enable side-by-side comparisons of alternative datasets or custom model uploads.
