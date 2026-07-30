import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ONNX_DIR = os.path.join(DATA_DIR, "onnx")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ONNX_DIR, exist_ok=True)

# Model Settings
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FP32_MODEL_PATH = os.path.join(DATA_DIR, "pytorch_model")
ONNX_MODEL_PATH = os.path.join(ONNX_DIR, "model.onnx")
QUANTIZED_ONNX_MODEL_PATH = os.path.join(ONNX_DIR, "model_int8.onnx")

# FAISS Index Path
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_hnsw.index")

# Confidence Escalation Thresholds
# If confidence >= THRESHOLD_HIGH -> Return Vetted FAQ Answer (95% target, no hallucination)
# If THRESHOLD_LOW <= confidence < THRESHOLD_HIGH -> Run RAG generative response
# If confidence < THRESHOLD_LOW -> Escalate to human agent (Containment rate tuning)
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", "0.80"))
THRESHOLD_LOW = float(os.getenv("THRESHOLD_LOW", "0.50"))

# Celery & Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# API Configuration
API_TITLE = "Bilingual RAG Support Chatbot"
API_VERSION = "1.0.0"
