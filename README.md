# Predicting the Unpredictable  
### ACSE / EDSML Mini-Project (January 2026) -- Lee

This repository contains our group solution to the **“Predicting the Unpredictable”**
mini-project, which focuses on **machine learning for lightning storm prediction**
using multi-modal satellite imagery and lightning observations.

The project is structured around **four independent tasks**, each addressing
a different prediction problem in storm nowcasting and analysis.

## Project Overview

Lightning storms pose significant risks to infrastructure, aviation, and public safety.
Accurate short-term forecasting requires extracting meaningful spatio-temporal patterns
from heterogeneous data sources, including satellite imagery and lightning observations.

In this project, we develop and evaluate deep learning–based solutions for four tasks,
ranging from radar image prediction to storm type classification and lightning flash
forecasting. Our emphasis is on **clear workflows**, **justified design choices**, and
**reproducible experimentation**, in line with the project assessment criteria.

## Dataset

Each storm event contains:

- **Satellite imagery (36 frames, 5-minute intervals)**:
  - `vis`   : visible imagery
  - `ir069` : water vapour (infrared)
  - `ir107` : cloud / surface temperature (infrared)
  - `vil`   : vertically integrated liquid (radar)
- **Lightning data (`lght`)**:
  - Irregular time series of lightning flashes
- **Metadata**:
  - Event time and geographical location

The training dataset consists of **800 storm events** across the United States,
labelled into eight event types:
Flash Flood, Flood, Funnel Cloud, Hail, Heavy Rain, Lightning,
Thunderstorm Wind, and Tornado.

## Task Breakdown
### Task 1: Future Radar Image Prediction

**Objective:**  
Predict the next 12 `vil` radar images given 12 past `vil` frames from a storm.

**Task Type:**  
Spatio-temporal image forecasting.

**Location:**  
- Code: `Task1/`  
- Final report & results: `notebooks-final/Task_1_report.ipynb`


### Task 2: Cross-Modal Radar Reconstruction

**Objective:**  
Reconstruct missing `vil` images using `vis`, `ir069`, and `ir107` satellite channels.

**Task Type:**  
Multi-modal image-to-image prediction.

**Location:**  
- Code: `Task2/`  
- Final report & results: `notebooks-final/Task-2-report.ipynb`

### Task 3: Storm Event Classification

**Objective:**  
Classify each storm into one of eight event types using all available image channels.

**Task Type:**  
Multi-class classification over spatio-temporal data.

**Location:**  
- Code: `Task3/`  
- Final report & results: `notebooks-final/Task-3-report.ipynb`

### Task 4: Lightning Flash Prediction

**Objective:**  
Predict the number, timing, and spatial location of lightning flashes during a storm.

**Task Type:**  
Spatio-temporal regression / point prediction.

**Location:**  
- Code: `Task4/`  
- Final report & results: `notebooks-final/Task-4-report.ipynb`


## Repository Structure

```text
.
├── Task1/                # Task 1 implementation code
├── Task2/                # Task 2 implementation code
├── Task3/                # Task 3 implementation code
├── Task4/                # Task 4 implementation code
├── notebooks-final/      # Final notebooks (Tasks 1–4 reports & results)
├── README.md             # Project-level documentation
├── requirements.txt      # Python dependencies
├── reference/            # Shared reference implementations and utilities
├── LICENSE
└── .gitignore
```

## How to Use This Repository

- To **understand the methodology, experiments, and results**, refer to the notebooks
  in `notebooks-final/`, which contain the final reports for Tasks 1–4.
- To **inspect or reuse implementation details**, see the corresponding `TaskX/`
  directories, where each task is implemented independently.
- A shared `reference/` module is provided for common utilities and baseline
  components used across tasks.

For reproducibility and consistent imports, the repository can be installed in
editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

#### Useful resources

[Project introduction slides](https://imperiallondon-my.sharepoint.com/:b:/g/personal/bm1417_ic_ac_uk/IQDh5erhwXmATaT5ANnvqAFEAaUeRS_WN1DHDecpt3FPDrc)

[Colab notebook: Example data downloading and exploration](https://colab.research.google.com/drive/15Rw4zi3V8S9g4h89KJ28LPzZakDJAT0r?usp=sharing)

[Colab notebook: Suprise storms - description and submission instructions](https://colab.research.google.com/drive/19OuZsdBfTslpJdY60T8BOhbRKOtSP-XX?usp=sharing)

#### Submitting your final Jupyter notebooks

- Please place your final Jupyter notebooks in a top-level folder called `notebooks-final`. 
- Please use the [template](notebooks-final/Task-%5Bnumber%5D-report.ipynb) provided.
## How to Use This Repository

- To **understand the methodology, experiments, and results**, refer to the notebooks
  in `notebooks-final/`, which contain the final reports for Tasks 1–4.
- To **inspect or reuse implementation details**, see the corresponding `TaskX/`
  directories, where each task is implemented independently.
- A shared `reference/` module is provided for common utilities and baseline
  components used across tasks.

For reproducibility and consistent imports, the repository can be installed in
editable mode:

```bash
pip install -r requirements.txt
pip install -e .
