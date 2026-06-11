HF_USERNAME = "research-simulation26"
HF_DATASET_REPO = f"{HF_USERNAME}/cicids2017-gnn"
HF_MODEL_REPO = f"{HF_USERNAME}/cybersage-cicids2017"

FEATURE_COLS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Min', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Bwd Packets/s', 'Min Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'FIN Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'Down/Up Ratio',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean',
    'Active Std', 'Idle Std'
]

CLASS_NAMES = [
    'BENIGN', 'Bot', 'DDoS', 'DoS_GoldenEye', 'DoS_Hulk',
    'DoS_Slowhttptest', 'DoS_slowloris', 'FTPPatator',
    'PortScan', 'SSHPatator', 'Web_Attack_Brute_Force',
    'Web_Attack_XSS'
]

MODEL_CONFIG = {
    "in_channels": 40,
    "hidden_channels": 128,
    "out_channels": 12,
    "dropout": 0.4
}

TRAINING_CONFIG = {
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 400,
    "early_stop_patience": 50,
    "k_same": 5,
    "k_cross": 2
}
