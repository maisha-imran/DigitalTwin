# 🛣️ Road Digital Twin — Physics-Informed GNN

> **Graph Neural Network-based Predictive Digital Twin for Road Deterioration Analysis**

A full-stack AI research platform that creates a **digital twin of a city road network**, predicts road deterioration using a **Physics-Informed Graph Attention Network (PI-GNN)**, and visualises results on an interactive smart-city dashboard.

---

## 🌟 Features

| Component | Description |
|-----------|-------------|
| **PI-GNN** | 3-layer Graph Attention Network with physics constraints |
| **Physics Loss** | Monotonicity · Traffic sensitivity · Rainfall influence · IRI bounds |
| **Digital Twin** | Real-time IRI prediction for every road segment |
| **Interactive Map** | Leaflet map with condition-coloured segments |
| **Maintenance Planner** | Priority ranking + repair cost estimates + CSV export |
| **Colab Notebook** | End-to-end GPU-accelerated training notebook |

---

## 🏗️ Architecture

```
Inputs (9 features)
  lat, lon, road_type, lanes, traffic_volume,
  rainfall_mm, length_m, speed_limit, iri_current
        │
        ▼
  Linear Projection (→ 64d)
        │
    GAT Layer 1 (4 heads) + Residual + BN
        │
    GAT Layer 2 (4 heads) + Residual + BN
        │
    GAT Layer 3 (1 head) + BN
        │
  MLP Head (64 → 32 → 1)
        │
        ▼
  Predicted IRI (m/km)

Total Loss = MSE(ŷ, y) + λ × Physics_Loss
Physics_Loss = Monotonicity + Traffic + Rainfall + Boundary
```

---

## 📁 Project Structure

```
road-digital-twin/
├── backend/
│   ├── main.py                # FastAPI application
│   ├── requirements.txt
│   ├── model/
│   │   ├── gnn_model.py       # PI-GNN architecture
│   │   ├── physics_loss.py    # Physics-informed constraints
│   │   ├── trainer.py         # Training loop
│   │   └── inference.py       # Inference utilities
│   ├── data/
│   │   ├── synthetic_data.py  # Synthetic road network generator
│   │   ├── graph_builder.py   # PyG graph construction
│   │   └── preprocess.py      # Data loading and splitting
│   └── utils/
│       ├── metrics.py
│       └── helpers.py
├── notebooks/
│   └── Road_Digital_Twin_PI_GNN.ipynb
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/maisha-imran/DigitalTwin
cd DigitalTwin
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note**: `torch-geometric` requires matching PyTorch version. If auto-install fails:
> ```bash
> pip install torch==2.2.1
> pip install torch-geometric --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html
> ```

Start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## 🤖 Training the Model

### Google Colab (GPU training)

1. Open `notebooks/Road_Digital_Twin_PI_GNN.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run All Cells
4. Download `saved_models/pignn_model.pt` and `saved_models/scaler.pkl`
5. Place them in `backend/saved_models/`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/train` | Train the PI-GNN model |
| `POST` | `/predict` | Run inference with env. parameters |
| `GET`  | `/metrics` | Evaluation metrics + training history |
| `GET`  | `/roads` | All road segments with predictions |
| `GET`  | `/maintenance` | Priority maintenance ranking |

---


## 🧪 IRI Condition Classes

| IRI (m/km) | Condition | Urgency |
|------------|-----------|---------|
| < 2.0 | ✅ Good | None |
| 2.0 – 3.5 | 🟡 Fair | Low |
| 3.5 – 5.0 | 🟠 Moderate | Medium |
| 5.0 – 7.0 | 🔴 Poor | High |
| ≥ 7.0 | 🚨 Critical | Immediate |

---

## ⚙️ Physics Constraints

```python
# 1. Monotonicity — roads don't self-heal
loss_mono = relu(0.9 × IRI_current - IRI_predicted).mean()

# 2. Traffic sensitivity — high traffic → higher deterioration
loss_traffic = relu(-delta × high_traffic_mask).mean()

# 3. Rainfall influence — high rain → no improvement
loss_rain = relu(-delta × high_rain_mask × 0.5).mean()

# 4. Boundary enforcement
loss_bounds = (relu(0.5 - pred) + relu(pred - 12.0)).mean()

Total Physics Loss = λ × (mono + traffic + rain + bounds)
```

---

## 🔮 Future Improvements

- [ ] Real-time IoT sensor data ingestion
- [ ] Temporal GNN for multi-step IRI forecasting
- [ ] OBIA (Object-Based Image Analysis) from satellite imagery
- [ ] Budget optimisation with ILP solver
- [ ] Federated learning across municipalities
- [ ] Mobile app for field inspection with AR overlay

---

## Frontend Code At

 https://github.com/maisha-imran/DigitalTwin-Frontend

---

## 📄 License

MIT License · Built for research demos and hackathons.
