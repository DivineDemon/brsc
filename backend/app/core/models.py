from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from .database import Base

class RelevanceLog(Base):
    """
    Automated Relevance Logging Table.
    Saves metadata about every query, matched FAQ, similarity score,
    latency, precision mode, search method, containment status, and feedback.
    Saves ~120 hours/month of manual QA logging by automating data capture.
    """
    __tablename__ = "relevance_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    query = Column(String(500), nullable=False)
    detected_language = Column(String(10), nullable=False)  # "en", "ur"
    matched_faq_id = Column(String(50), nullable=True)       # e.g., "faq_1"
    similarity_score = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=False)              # Retrieval + evaluation latency
    search_method = Column(String(20), nullable=False)       # "hnsw", "brute_force"
    precision_mode = Column(String(10), nullable=False)      # "fp32", "int8"
    status = Column(String(30), nullable=False)             # "contained_vetted", "contained_rag", "escalated"
    response_text = Column(Text, nullable=False)
    feedback = Column(String(10), nullable=True)            # "like", "dislike"


class EvaluationRun(Base):
    """
    Standardized Evaluation Practice Table.
    Saves the metrics from runs of the evaluation harness.
    Demonstrates model release qualification and drift monitoring.
    """
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_cases = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=False)                 # Retrieval accuracy %
    containment_rate = Column(Float, nullable=False)         # Converted containment %
    avg_latency_ms = Column(Float, nullable=False)
    p95_latency_ms = Column(Float, nullable=False)
    precision_mode = Column(String(10), nullable=False)      # "fp32", "int8"
    search_method = Column(String(20), nullable=False)       # "hnsw", "brute_force"
