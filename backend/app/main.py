import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from .core import config, database, models, model, retriever
from .tasks import trigger_log_relevance, build_faiss_index_task

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# State management for Thread-safe Model Switching
class ModelManager:
    """
    Manages active model instances for hot-swapping between FP32 and INT8.
    Ensures zero downtime when switching precisions or index types in real-time.
    """
    def __init__(self):
        self.embedders = {}
        self.retrievers = {}

    def get_resources(self, precision: str = "int8") -> tuple[model.BilingualEmbedder, retriever.FAQRetriever]:
        precision = precision.lower()
        if precision not in self.embedders:
            logger.info(f"[ModelManager] Initializing new resources for precision: {precision}")
            self.embedders[precision] = model.BilingualEmbedder(precision=precision)
            self.retrievers[precision] = retriever.FAQRetriever(self.embedders[precision])
        return self.embedders[precision], self.retrievers[precision]

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler.
    Creates tables on startup, builds/loads ONNX models, and prepares FAISS index.
    """
    logger.info("Starting up Bilingual RAG Support Chatbot API...")
    
    # 1. Create SQL Database Tables
    logger.info("Initializing SQLite database tables...")
    models.Base.metadata.create_all(bind=database.engine)
    
    # 2. Pre-load default INT8 Embedder and Retriever
    logger.info("Warm-starting embedding models...")
    try:
        model_manager.get_resources(precision="int8")
        logger.info("Default INT8 resources warm-started successfully.")
    except Exception as e:
        logger.error(f"Warmstart failed: {e}. PyTorch FP32 fallback will resolve on first query.")
        
    yield
    logger.info("Shutting down Bilingual RAG Support Chatbot API...")

# Initialize FastAPI App
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    lifespan=lifespan
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")
async def chat_endpoint(
    payload: dict = Body(...),
    db: Session = Depends(database.get_db)
):
    """
    Core Chat Endpoint.
    Accepts bilingual message queries, executes semantic HNSW search,
    applies confidence-based routing, and logs analytics asynchronously.
    """
    query = payload.get("message", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    # Load configuration overrides with defaults
    precision_mode = payload.get("precision_mode", "int8").lower()
    search_method = payload.get("search_method", "hnsw").lower()
    threshold_high = payload.get("threshold_high")
    threshold_low = payload.get("threshold_low")
    
    if precision_mode not in ["fp32", "int8"]:
        precision_mode = "int8"
    if search_method not in ["hnsw", "brute_force"]:
        search_method = "hnsw"

    # Fetch corresponding model & retriever instances (hot-swappable)
    _, active_retriever = model_manager.get_resources(precision=precision_mode)
    
    # Convert thresholds if provided
    th_high = float(threshold_high) if threshold_high is not None else None
    th_low = float(threshold_low) if threshold_low is not None else None

    # Run the query pipeline
    response = active_retriever.process_query(
        query=query,
        search_method=search_method,
        threshold_high=th_high,
        threshold_low=th_low
    )

    # Dispatch relevance log asynchronously (non-blocking)
    # Includes all transaction metadata for dashboard tracking
    log_payload = {
        "query": response["query"],
        "detected_language": response["detected_language"],
        "matched_faq_id": response["matched_faq_id"],
        "similarity_score": response["similarity_score"],
        "latency_ms": response["latency_ms"],
        "search_method": search_method,
        "precision_mode": precision_mode,
        "status": response["status"],
        "response_text": response["response_text"],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Decouple DB write using Celery (with sync fallback automatically handled)
    trigger_log_relevance(log_payload, use_celery=True)

    return response


@app.get("/api/logs")
async def get_logs_endpoint(limit: int = 50, db: Session = Depends(database.get_db)):
    """Retrieves historical automated relevance logs for real-time dashboard display."""
    logs = db.query(models.RelevanceLog).order_by(models.RelevanceLog.timestamp.desc()).limit(limit).all()
    return logs


@app.post("/api/logs/{log_id}/feedback")
async def log_feedback_endpoint(
    log_id: int,
    feedback: str = Body(embed=True),
    db: Session = Depends(database.get_db)
):
    """Records thumbs up / thumbs down feedback on chatbot responses."""
    feedback = feedback.lower()
    if feedback not in ["like", "dislike", "none"]:
        raise HTTPException(status_code=400, detail="Feedback must be 'like', 'dislike', or 'none'.")
        
    log_entry = db.query(models.RelevanceLog).filter(models.RelevanceLog.id == log_id).first()
    if not log_entry:
        raise HTTPException(status_code=404, detail="Log entry not found.")
        
    log_entry.feedback = feedback if feedback != "none" else None
    db.commit()
    return {"status": "success", "message": "Feedback recorded successfully."}


@app.get("/api/benchmarks")
async def get_benchmarks_endpoint():
    """Fetches static embedding model benchmarks (INT8 vs FP32 sizes & ratios)."""
    # Force initialize to calculate sizes
    embedder_int8, _ = model_manager.get_resources(precision="int8")
    benchmarks = embedder_int8.get_benchmarks()
    return benchmarks


@app.get("/api/stats")
async def get_stats_endpoint(db: Session = Depends(database.get_db)):
    """
    Computes real-time analytical statistics.
    Calculates overall containment rate, latency averages, and throughput metrics
    comparing FP32 vs. INT8 and HNSW vs. Cosine.
    """
    total_queries = db.query(models.RelevanceLog).count()
    
    if total_queries == 0:
        return {
            "total_queries": 0,
            "containment_rate": 100.0,
            "avg_latency_ms": 0.0,
            "urdu_count": 0,
            "english_count": 0,
            "int8_avg_latency_ms": 0.0,
            "fp32_avg_latency_ms": 0.0,
            "hnsw_avg_latency_ms": 0.0,
            "brute_avg_latency_ms": 0.0,
            "retention_vetted_count": 0,
            "retention_rag_count": 0,
            "escalated_count": 0,
            "feedback_likes": 0,
            "feedback_dislikes": 0
        }

    # Containment calculations (vetted + RAG are contained. Escalated are handoffs)
    escalated_count = db.query(models.RelevanceLog).filter(models.RelevanceLog.status == "escalated").count()
    vetted_count = db.query(models.RelevanceLog).filter(models.RelevanceLog.status == "contained_vetted").count()
    rag_count = db.query(models.RelevanceLog).filter(models.RelevanceLog.status == "contained_rag").count()
    
    containment_rate = round(((total_queries - escalated_count) / total_queries) * 100, 1)

    # Average general latency
    avg_latency = db.query(models.RelevanceLog.latency_ms).all()
    avg_latency_ms = round(sum(l[0] for l in avg_latency) / total_queries, 2)

    # Language breakdown
    urdu_count = db.query(models.RelevanceLog).filter(models.RelevanceLog.detected_language == "ur").count()
    english_count = total_queries - urdu_count

    # Performance breakdowns
    int8_logs = db.query(models.RelevanceLog.latency_ms).filter(models.RelevanceLog.precision_mode == "int8").all()
    int8_avg = round(sum(l[0] for l in int8_logs) / len(int8_logs), 2) if int8_logs else 0.0

    fp32_logs = db.query(models.RelevanceLog.latency_ms).filter(models.RelevanceLog.precision_mode == "fp32").all()
    fp32_avg = round(sum(l[0] for l in fp32_logs) / len(fp32_logs), 2) if fp32_logs else 0.0

    hnsw_logs = db.query(models.RelevanceLog.latency_ms).filter(models.RelevanceLog.search_method == "hnsw").all()
    hnsw_avg = round(sum(l[0] for l in hnsw_logs) / len(hnsw_logs), 2) if hnsw_logs else 0.0

    brute_logs = db.query(models.RelevanceLog.latency_ms).filter(models.RelevanceLog.search_method == "brute_force").all()
    brute_avg = round(sum(l[0] for l in brute_logs) / len(brute_logs), 2) if brute_logs else 0.0

    # User satisfaction
    likes = db.query(models.RelevanceLog).filter(models.RelevanceLog.feedback == "like").count()
    dislikes = db.query(models.RelevanceLog).filter(models.RelevanceLog.feedback == "dislike").count()

    return {
        "total_queries": total_queries,
        "containment_rate": containment_rate,
        "avg_latency_ms": avg_latency_ms,
        "urdu_count": urdu_count,
        "english_count": english_count,
        "int8_avg_latency_ms": int8_avg,
        "fp32_avg_latency_ms": fp32_avg,
        "hnsw_avg_latency_ms": hnsw_avg,
        "brute_avg_latency_ms": brute_avg,
        "retention_vetted_count": vetted_count,
        "retention_rag_count": rag_count,
        "escalated_count": escalated_count,
        "feedback_likes": likes,
        "feedback_dislikes": dislikes
    }


@app.post("/api/settings/rebuild-index")
async def rebuild_index_endpoint(precision_mode: str = Body("int8", embed=True)):
    """Triggers background HNSW vector index rebuild."""
    try:
        # Trigger background task
        build_faiss_index_task.delay(precision_mode)
        return {"status": "success", "message": "Index rebuild triggered in background."}
    except Exception as e:
        logger.warning(f"Celery failed to run task, running inline: {e}")
        # Run inline
        embedder, active_retriever = model_manager.get_resources(precision=precision_mode)
        active_retriever.rebuild_index()
        return {"status": "success", "message": "Index rebuilt synchronously (fallback)."}


@app.post("/api/evaluation/run")
async def run_evaluation_endpoint(
    precision_mode: str = Body("int8", embed=True),
    search_method: str = Body("hnsw", embed=True),
    db: Session = Depends(database.get_db)
):
    """
    Evaluation Harness.
    Executes a simulated batch of 25 benchmark queries (English and Urdu) with varying intents.
    Measures accuracy, latency parameters, and containment, logging a standardized run result.
    Allows instant demonstration of your release-validation engineering.
    """
    _, active_retriever = model_manager.get_resources(precision=precision_mode)
    
    # 25 test queries representing various logistics intents (some match, some are ambiguous/out-of-domain)
    eval_queries = [
        # Explicit FAQ Match (Tracking)
        ("Where is my parcel?", "faq_1", "en"),
        ("میں اپنی چیز کو کیسے ٹریک کروں؟", "faq_1", "ur"),
        ("Track parcel", "faq_1", "en"),
        
        # Explicit FAQ Match (Rates)
        ("What are delivery charges?", "faq_2", "en"),
        ("ڈیلیوری فیس کتنی ہے؟", "faq_2", "ur"),
        
        # Explicit FAQ Match (Customs)
        ("Do I need to pay customs?", "faq_3", "en"),
        ("ٹیکس کون دے گا؟", "faq_3", "ur"),
        
        # Explicit FAQ Match (Address Redirect)
        ("Can I change my address after shipping?", "faq_4", "en"),
        ("پتہ تبدیل کر سکتے ہیں؟", "faq_4", "ur"),
        
        # Explicit FAQ Match (Missed Delivery)
        ("What if I am not at home?", "faq_5", "en"),
        ("اگر میں گھر نہ ہوں تو؟", "faq_5", "ur"),
        
        # RAG / Medium Confidence (Near Matches)
        ("How to change delivery tracking code?", "faq_4", "en"),
        ("پیکج کینسل کروانے کا طریقہ", "faq_6", "ur"),
        ("I received a damaged phone cover", "faq_7", "en"),
        ("کیا میں بیٹری جہاز میں بھیج سکتا ہوں؟", "faq_8", "ur"),
        
        # Out-of-domain / Low Confidence (Should Escalate)
        ("What is your CEO name?", None, "en"),
        ("کیا مجھے نوکری مل سکتی ہے؟", None, "ur"),
        ("Can I order a pizza from your app?", None, "en"),
        ("موسم کیسا ہے؟", None, "ur"),
        ("Let me talk to a human right now", None, "en")
    ]
    
    start_run_time = time.time()
    latencies = []
    correct_retrievals = 0
    contained_count = 0
    
    # Process queries sequentially and record metrics
    for query, expected_faq_id, lang in eval_queries:
        res = active_retriever.process_query(query, search_method=search_method)
        latencies.append(res["latency_ms"])
        
        # Retrieval Accuracy check (only check if it mapped to the expected FAQ for in-domain queries)
        if expected_faq_id:
            if res["matched_faq_id"] == expected_faq_id:
                correct_retrievals += 1
        else:
            # Out-of-domain should escalate
            if res["status"] == "escalated":
                correct_retrievals += 1
                
        if res["status"] in ["contained_vetted", "contained_rag"]:
            contained_count += 1
            
        # Write to general relevance logs synchronously for evaluation metrics history
        log_payload = {
            "query": query,
            "detected_language": res["detected_language"],
            "matched_faq_id": res["matched_faq_id"],
            "similarity_score": res["similarity_score"],
            "latency_ms": res["latency_ms"],
            "search_method": search_method,
            "precision_mode": precision_mode,
            "status": res["status"],
            "response_text": res["response_text"],
            "timestamp": datetime.utcnow().isoformat()
        }
        trigger_log_relevance(log_payload, use_celery=False)
        time.sleep(0.01) # Small sleep to avoid duplicate timestamps

    total_cases = len(eval_queries)
    accuracy = round((correct_retrievals / total_cases) * 100, 1)
    containment_rate = round((contained_count / total_cases) * 100, 1)
    avg_latency = round(sum(latencies) / total_cases, 2)
    p95_latency = round(np.percentile(latencies, 95), 2)
    
    # Record Evaluation Run
    eval_run = models.EvaluationRun(
        total_cases=total_cases,
        accuracy=accuracy,
        containment_rate=containment_rate,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        precision_mode=precision_mode,
        search_method=search_method
    )
    db.add(eval_run)
    db.commit()
    
    return {
        "timestamp": eval_run.timestamp.isoformat(),
        "total_cases": total_cases,
        "accuracy": accuracy,
        "containment_rate": containment_rate,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "precision_mode": precision_mode,
        "search_method": search_method
    }


@app.get("/api/evaluation/runs")
async def get_eval_runs_endpoint(db: Session = Depends(database.get_db)):
    """Retrieves all recorded evaluation runs to plot progress or history on the dashboard."""
    runs = db.query(models.EvaluationRun).order_by(models.EvaluationRun.timestamp.desc()).limit(20).all()
    return runs


# Serve Frontend Static Files
# Set up paths
frontend_dir = os.path.join(os.path.dirname(config.BASE_DIR), "frontend")

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
        
    @app.get("/css/style.css")
    async def serve_css():
        return FileResponse(os.path.join(frontend_dir, "css", "style.css"))
        
    @app.get("/js/app.js")
    async def serve_js():
        return FileResponse(os.path.join(frontend_dir, "js", "app.js"))
else:
    logger.warning(f"Frontend static directory not found at {frontend_dir}. API runs alone.")
    @app.get("/")
    async def root_message():
        return {"message": "Bilingual RAG Chatbot API is online. Frontend static files are not yet deployed."}
