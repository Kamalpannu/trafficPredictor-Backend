# Traffic Prediction Backend

This is the backend service for the **Traffic Prediction App**.  
It uses **FastAPI**, **SQLAlchemy**, and a **RandomForest-based traffic predictor** to simulate and predict traffic in different zones.

---

## Features

- Simulate current traffic with random spikes during peak hours.
- Predict traffic for the next 30 minutes using a trained RandomForest model.
- Evaluate prediction accuracy per zone (R², MAE, MSE, RMSE).
- Asynchronous database operations with SQLAlchemy.
- CORS enabled for frontend integration.
- Docker support for containerized deployment.

---

## Tech Stack

- **Python 3.11+**
- **FastAPI**
- **SQLAlchemy (async)**
- **SQLite** (default, can be changed)
- **scikit-learn** (RandomForest)
- **pandas & numpy**
- **Docker**

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Kamalpannu/trafficPredictor-Backend.git

