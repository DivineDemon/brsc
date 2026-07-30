import os
import logging
from datetime import datetime
from celery import Celery
from .core import config, database, models, model, retriever

logger = logging.getLogger(__name__)

# Initialize Celery App
# Uses Redis as the broker and backend for task state persistence
celery_app = Celery(
    "brsc_tasks", 
    broker=config.REDIS_URL, 
    backend=config.REDIS_URL
)

# Optional configuration overrides for Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


@celery_app.task(name="tasks.build_faiss_index_task")
def build_faiss_index_task(precision_mode: str = "int8"):
    """
    Background Task: Rebuilds the FAISS HNSW retrieval index.
    Triggered during startup, or during bulk FAQ ingestion/updates.
    Enables low-latency vector search under heavy write loads.
    """
    logger.info(f"[Celery] Rebuilding FAISS Index in background using precision: {precision_mode}")
    try:
        embedder = model.BilingualEmbedder(precision=precision_mode)
        search_engine = retriever.FAQRetriever(embedder)
        search_engine.rebuild_index()
        logger.info("[Celery] FAISS Index rebuilt and saved successfully.")
        return {"status": "success", "message": "Index rebuilt successfully."}
    except Exception as e:
        logger.error(f"[Celery] Failed to rebuild FAISS Index: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(name="tasks.log_relevance_async_task")
def log_relevance_async_task(log_data: dict):
    """
    Background Task: Writes transaction telemetry to SQLite/Postgres.
    Decoupling logging writes from the main API thread ensures 
    sub-400ms p95 latencies and high throughput (95+ req/s).
    """
    logger.debug(f"[Celery] Logging transaction async for query: '{log_data.get('query')}'")
    db = database.SessionLocal()
    try:
        log_entry = models.RelevanceLog(
            query=log_data["query"],
            detected_language=log_data["detected_language"],
            matched_faq_id=log_data.get("matched_faq_id"),
            similarity_score=log_data.get("similarity_score"),
            latency_ms=log_data["latency_ms"],
            search_method=log_data["search_method"],
            precision_mode=log_data["precision_mode"],
            status=log_data["status"],
            response_text=log_data["response_text"],
            feedback=log_data.get("feedback")
        )
        if "timestamp" in log_data and log_data["timestamp"]:
            try:
                log_entry.timestamp = datetime.fromisoformat(log_data["timestamp"])
            except ValueError:
                pass
                
        db.add(log_entry)
        db.commit()
        return {"status": "success", "log_id": log_entry.id}
    except Exception as e:
        logger.error(f"[Celery] Failed to write relevance log async: {e}")
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


def trigger_log_relevance(log_data: dict, use_celery: bool = True):
    """
    Helper function to dispatch logging.
    If Celery/Redis is available and configured, sends to Celery.
    Otherwise, executes synchronously (fallback for local development).
    """
    if use_celery:
        try:
            # Send to Celery worker asynchronously
            log_relevance_async_task.delay(log_data)
            return
        except Exception as e:
            logger.warning(f"Celery dispatch failed: {e}. Executing logging inline as fallback.")
            
    # Inline Fallback execution
    db = database.SessionLocal()
    try:
        log_entry = models.RelevanceLog(
            query=log_data["query"],
            detected_language=log_data["detected_language"],
            matched_faq_id=log_data.get("matched_faq_id"),
            similarity_score=log_data.get("similarity_score"),
            latency_ms=log_data["latency_ms"],
            search_method=log_data["search_method"],
            precision_mode=log_data["precision_mode"],
            status=log_data["status"],
            response_text=log_data["response_text"]
        )
        db.add(log_entry)
        db.commit()
    except Exception as db_err:
        logger.error(f"Inline database logging failed: {db_err}")
    finally:
        db.close()
