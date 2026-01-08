# Hospital Resource Analytics

A comprehensive **machine learning pipeline** for hospital resource analytics.  
This project focuses on analyzing hospital consultation data, understanding key metrics, and predicting future values of critical hospital KPIs such as `APC_Finished_Consultant` using LSTM, PROPHET, SARIMA algorthm

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Installation](#installation)  
5. [Usage](#usage)  
    - [Train the Model](#train-the-model)  
    - [Load and Predict](#load-and-predict)  
6. [Machine Learning Details](#machine-learning-details)  
7. [File Structure](#file-structure)  
8. [Future Improvements](#future-improvements)  
9. [License](#license)  

---

## Project Overview

Hospitals generate a lot of operational data every month, including:

- Finished consultations  
- Procedures performed  
- Outpatient appointments and attendance  
- Emergency and ordinary admissions  

The goal of this project is to **analyze this data**, understand trends, and **predict future workloads** to help hospital management with resource planning.

The pipeline includes:

1. Data loading and cleaning  
2. Feature engineering  
3. Random Forest model training  
4. Model evaluation (RMSE and R²)  
5. Feature importance analysis  
6. Saving the trained model for future predictions  

---

## Features

- Predict **APC_Finished_Consultant** (number of consultations completed by consultants)  
- Handle **multiple hospital KPIs** as input features  
- Evaluate model using **Root Mean Squared Error (RMSE)** and **R² score**  
- Identify **most important features** influencing predictions  
- Save trained model as `rf_model.pkl` for easy reuse  
- Easy-to-extend for **other targets** or additional features  

---

## Dataset

The main dataset is a **Consultation dataset** with monthly hospital data.  

**Columns used:**

| Column Name | Description |
|-------------|-------------|
| `CALENDAR_MONTH_END_DATE` | Month in format `MMM-YY` or `YYYY-MM` |
| `APC_Finished_Consultant` | Target variable: finished consultations |
| `APC_FCEs_with_a_procedure` | Number of Finished Consultant Episodes with a procedure |
| `APC_Percent_FCEs_with_procedure` | % of FCEs with procedures |
| `APC_Ordinary_Episodes` | Ordinary inpatient episodes |
| `APC_Day_Case_Episodes` | Day case episodes |
| `APC_Day_Case_Episodes_with_proc` | Day cases with procedures |
| `APC_Percent_Day_Cases_with_proc` | % of day cases with procedures |
| `APC_Finished_Admission_Episodes` | Finished admissions |
| `APC_Emergency` | Emergency episodes |
| `Outpatient_Total_Appointments` | Total outpatient appointments |
| `Outpatient_Attended_Appointments` | Attended appointments |
| `Outpatient_Percent_Attended` | % of attended appointments |
| `Outpatient_DNA_Appointment` | Did not attend appointments |
| `Outpatient_Percent_DNA` | % of DNA |
| `Outpatient_Follow_Up_Attendance` | Follow-up attendance |
| `Outpatient_Attendance_Type_1` | Attendance type 1 |
| `Outpatient_Attendance_Type_2` | Attendance type 2 |

> Note: The dataset can be replaced with updated hospital data as long as the column structure is consistent.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/hospital-resource-analytics.git
cd hospital-resource-analytics
