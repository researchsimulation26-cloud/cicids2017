# CyberSAGE — GNN-based Network Intrusion Detection

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B)](https://your-app-url.streamlit.app)

**CyberSAGE** is a Graph Neural Network (GraphSAGE-based) for multi-class intrusion detection on the CIC-IDS2017 dataset. It uses a custom per-class k-NN graph structure to model network flow relationships.

## Architecture

- **Graph Construction**: FAISS-based intra-class k-NN edges + cross-class edges to BENIGN
- **Model**: 2-layer SAGEConv with skip connections, batch norm, and dropout
- **Class Imbalance**: Dampened class weights (sqrt of balanced weights)
- **Explainability**: SHAP, GNNExplainer, Gradient Saliency

## Project Structure

```
csir/
├── app/                  # Streamlit application (6 pages)
├── data/                 # Data loading & graph construction
├── model/                # CyberSAGE architecture & training
├── explain/              # SHAP, GNNExplainer, saliency
├── visualize/            # Plotly interactive figure generators
├── config.py             # Configuration & HF repo IDs
├── requirements.txt      # Python dependencies
└── upload_to_hf.py       # Script to upload artifacts to HF
```

## Deployment

### 1. Upload Artifacts to Hugging Face

Create two repos on Hugging Face:

**Dataset Repo** (`research-simulation26/cicids2017-gnn`):
- `cicids2017_final.parquet` — Preprocessed dataset
- `scaler.pkl` — StandardScaler fitted on training data
- `label_encoder_grouped.pkl` — LabelEncoder with grouped classes
- `cicids_graph.pt` — PyTorch Geometric Data object with pre-built graph
- `X_explain.pkl` — 500 test samples for SHAP explanation
- `y_explain.pkl` — Corresponding labels
- `shap_values.pkl` — Precomputed SHAP values

**Model Repo** (`research-simulation26/cybersage-cicids2017`):
- `best_model_v2.pt` — Trained CyberSAGE checkpoint

Run the upload script:
```bash
export HF_TOKEN=hf_YOUR_TOKEN
python upload_to_hf.py
```

### 2. Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Set the main file path to `app/streamlit_app.py`
5. Add a secret `HF_TOKEN` with your Hugging Face token
6. Deploy

## Local Development

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Results

The model achieves strong performance on majority classes (BENIGN, DDoS, DoS_Hulk, PortScan) with F1 > 0.95. Minority classes (Bot, Web attacks) show lower F1 due to extreme class imbalance.
