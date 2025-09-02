# main.py (Final Version with BOTH Live and Demo Modes)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pickle
import numpy as np
import uvicorn

# --- 1. SETUP & MODEL DEFINITION ---
app = FastAPI(title="GNN Fraud Detection API")
origins = ["http://127.0.0.1:5500", "http://localhost:5500"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GraphSAGE_Fraud_Detector(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

# Pydantic model for the "Live Mode"
class SimpleTransaction(BaseModel):
    Amount: float
    Time: float

# Pydantic model for the "Demo Mode"
class FullTransaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    Time: float
    
# --- 2. LAZY LOADING SETUP ---
# We use lazy loading to prevent your computer from hanging on startup
models = {}

def load_all_models():
    """Loads all models into the global 'models' dictionary if they haven't been loaded."""
    if "gnn_model" in models:
        return # Models are already loaded

    print("Lazy loading all models for the first time...")
    # GNN Model and its specific scalers
    with open('amount_scaler.pkl', 'rb') as f: models['amount_scaler'] = pickle.load(f)
    with open('time_scaler.pkl', 'rb') as f: models['time_scaler'] = pickle.load(f)
    
    gnn_model = GraphSAGE_Fraud_Detector(in_channels=30, hidden_channels=64, out_channels=2)
    gnn_model.load_state_dict(torch.load('fraud_gnn_model.pth'))
    gnn_model.eval()
    models['gnn_model'] = gnn_model

    # Feature Predictor Model and its scaler
    with open('feature_scaler.pkl', 'rb') as f: models['feature_scaler'] = pickle.load(f)
    with open('feature_predictor_model.pkl', 'rb') as f: models['feature_predictor_model'] = pickle.load(f)
        
    print("All models loaded successfully into memory.")


# --- 3. DEFINE API ENDPOINTS ---

@app.post("/predict_live")
def predict_live_fraud(transaction: SimpleTransaction):
    load_all_models() # Ensure all models are loaded
    
    input_data = np.array([[transaction.Time, transaction.Amount]])
    scaled_input = models['feature_scaler'].transform(input_data)
    predicted_v_features = models['feature_predictor_model'].predict(scaled_input)
    
    scaled_amount = models['amount_scaler'].transform([[transaction.Amount]])[0, 0]
    scaled_time = models['time_scaler'].transform([[transaction.Time]])[0, 0]
    full_feature_vector = np.concatenate([predicted_v_features[0], [scaled_amount, scaled_time]])
    input_tensor = torch.tensor([full_feature_vector], dtype=torch.float)

    dummy_edge_index = torch.tensor([[], []], dtype=torch.long)
    with torch.no_grad():
        logits = models['gnn_model'](input_tensor, dummy_edge_index)
        probabilities = F.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()

    is_fraud = bool(prediction == 1)
    return {"is_fraud": is_fraud, "confidence": f"{confidence:.2%}"}


@app.post("/predict_demo")
def predict_demo_fraud(transaction: FullTransaction):
    load_all_models() # Ensure all models are loaded

    scaled_amount = models['amount_scaler'].transform([[transaction.Amount]])[0, 0]
    scaled_time = models['time_scaler'].transform([[transaction.Time]])[0, 0]
    
    feature_vector = [
        transaction.V1, transaction.V2, transaction.V3, transaction.V4,
        transaction.V5, transaction.V6, transaction.V7, transaction.V8,
        transaction.V9, transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28,
        scaled_amount, scaled_time
    ]
    input_tensor = torch.tensor([feature_vector], dtype=torch.float)

    dummy_edge_index = torch.tensor([[], []], dtype=torch.long)
    with torch.no_grad():
        logits = models['gnn_model'](input_tensor, dummy_edge_index)
        probabilities = F.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()

    is_fraud = bool(prediction == 1)
    return {"is_fraud": is_fraud, "confidence": f"{confidence:.2%}"}