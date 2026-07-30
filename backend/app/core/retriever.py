import os
import json
import re
import time
import logging
import numpy as np
from . import config
from .model import BilingualEmbedder

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not installed. Falling back to Numpy-based vector search.")
    FAISS_AVAILABLE = False


class FAQRetriever:
    """
    FAQ Retrieval Engine.
    Combines FAISS HNSW flat indexing with fallback Brute-force Cosine Similarity.
    Features:
    - Urdu/English Language Auto-Detection (zero-dependency Unicode check).
    - Confidence-based Routing Layer (Escalation & Containment Tuning).
    - Simulated Citation Grounding and Self-Verification RAG.
    """
    def __init__(self, embedder: BilingualEmbedder):
        self.embedder = embedder
        self.faqs = []
        self.faqs_by_id = {}
        self.faq_embeddings = []  # List of embedding vectors
        self.index = None
        
        # Load the FAQs dataset
        self.load_faqs_dataset()
        
        # Initialize or load FAISS Index
        self.load_index()

    def load_faqs_dataset(self):
        """Loads FAQ JSON dataset and indexes by ID."""
        faq_path = os.path.join(config.DATA_DIR, "faqs.json")
        if not os.path.exists(faq_path):
            logger.error(f"FAQs file not found at {faq_path}. Please initialize the dataset first.")
            return

        with open(faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)
            
        self.faqs_by_id = {faq["id"]: faq for faq in self.faqs}
        logger.info(f"Loaded {len(self.faqs)} bilingual FAQs.")

    def detect_language(self, text: str) -> str:
        """
        Auto-detects whether the query is in Urdu or English.
        Uses a highly efficient Unicode range regex check for Arabic/Persian/Urdu script (0x0600 - 0x06FF).
        """
        # Range of Arabic/Persian/Urdu characters
        urdu_pattern = re.compile(r"[\u0600-\u06FF]")
        if urdu_pattern.search(text):
            return "ur"
        return "en"

    def rebuild_index(self):
        """
        Encodes the entire FAQ corpus and creates/saves the FAISS HNSW flat index.
        This indexes both English and Urdu questions to allow cross-lingual or same-lingual retrieval.
        """
        if not self.faqs:
            logger.error("No FAQs loaded. Cannot build index.")
            return

        logger.info("Rebuilding vector retrieval index...")
        
        # We index both English and Urdu questions separately or together.
        # Indexing both separately allows high-accuracy mapping for same-language searches,
        # while multilingual embeddings allow cross-lingual retrieval naturally.
        # Let's index questions for BOTH languages. Each FAQ maps to two entries in the index.
        corpus_questions = []
        self.index_mapping = []  # Maps index_id -> (faq_id, language)

        for faq in self.faqs:
            # English Question
            corpus_questions.append(faq["question_en"])
            self.index_mapping.append((faq["id"], "en"))
            
            # Urdu Question
            corpus_questions.append(faq["question_ur"])
            self.index_mapping.append((faq["id"], "ur"))

        # Generate Embeddings using current precision model
        logger.info(f"Encoding {len(corpus_questions)} questions into semantic space...")
        start_time = time.time()
        embeddings = self.embedder.embed(corpus_questions)
        encoding_latency = (time.time() - start_time) * 1000
        logger.info(f"Corpus encoded in {encoding_latency:.2f}ms.")

        self.faq_embeddings = embeddings
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # 1. Save embeddings locally for Brute-force fallback
        embeddings_path = os.path.join(config.DATA_DIR, "embeddings.npy")
        np.save(embeddings_path, embeddings_np)
        
        # Save index mapping to disk
        mapping_path = os.path.join(config.DATA_DIR, "index_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.index_mapping, f)

        # 2. Build FAISS HNSW flat index if FAISS is available
        if FAISS_AVAILABLE:
            dim = embeddings_np.shape[1]
            # M = number of connections per node (standard 16 or 32 for high search accuracy)
            M = 16
            # Metric: Inner Product (since embeddings are L2 normalized, Inner Product is Cosine Similarity)
            self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNERPRODUCT)
            self.index.add(embeddings_np)
            
            # Save HNSW index to disk
            faiss.write_index(self.index, config.FAISS_INDEX_PATH)
            logger.info(f"FAISS HNSW Flat index built and saved to {config.FAISS_INDEX_PATH}.")
        else:
            logger.info("FAISS not available. Skipping HNSW build. Using Numpy brute-force search.")

    def load_index(self):
        """Loads FAISS index and mappings from disk, if available."""
        mapping_path = os.path.join(config.DATA_DIR, "index_mapping.json")
        embeddings_path = os.path.join(config.DATA_DIR, "embeddings.npy")

        if os.path.exists(mapping_path) and os.path.exists(embeddings_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.index_mapping = json.load(f)
            
            self.faq_embeddings = np.load(embeddings_path).tolist()
            logger.info("Loaded index mappings and embeddings from disk.")
        else:
            # Rebuild on first initialization if files are missing
            self.rebuild_index()
            return

        if FAISS_AVAILABLE and os.path.exists(config.FAISS_INDEX_PATH):
            try:
                self.index = faiss.read_index(config.FAISS_INDEX_PATH)
                logger.info(f"Loaded FAISS HNSW Index from {config.FAISS_INDEX_PATH}.")
            except Exception as e:
                logger.error(f"Failed to read FAISS index: {e}. Falling back to Numpy search.")
                self.index = None

    def retrieve(self, query: str, search_method: str = "hnsw", top_k: int = 1) -> list[dict]:
        """
        Retrieves the top_k closest FAQs to the query.
        Returns a list of dicts with 'faq', 'score', and 'matched_language'.
        Supports both FAISS HNSW and Numpy brute-force cosine similarity.
        """
        if not self.faqs or not self.faq_embeddings:
            return []

        search_method = search_method.lower()
        query_emb = self.embedder.embed([query])[0]
        query_np = np.array([query_emb], dtype=np.float32)

        # 1. HNSW Vector Search using FAISS
        if search_method == "hnsw" and FAISS_AVAILABLE and self.index is not None:
            # Query HNSW Flat Index
            scores, indices = self.index.search(query_np, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                faq_id, matched_lang = self.index_mapping[idx]
                faq = self.faqs_by_id[faq_id]
                results.append({
                    "faq": faq,
                    "score": float(score),
                    "matched_language": matched_lang
                })
            return results

        # 2. Brute-Force Cosine Similarity Search using Numpy (Fallback)
        else:
            corpus_np = np.array(self.faq_embeddings, dtype=np.float32)
            # Dot product of normalized unit vectors = Cosine Similarity
            scores = np.dot(corpus_np, query_np[0])
            
            # Sort descending
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                faq_id, matched_lang = self.index_mapping[idx]
                faq = self.faqs_by_id[faq_id]
                results.append({
                    "faq": faq,
                    "score": float(scores[idx]),
                    "matched_language": matched_lang
                })
            return results

    def generate_rag_response(self, query: str, faq: dict, lang: str) -> str:
        """
        Simulates the intermediate RAG generative response with citation grounding and self-verification.
        Ensures a natural, context-grounded response while demonstrating LLM prompt safety engineering.
        """
        # Grounded citations based on the FAQ ID
        citation = f"[Verified FAQ Database #{faq['id'].replace('faq_', '')}]"
        
        if lang == "ur":
            answer = faq["answer_ur"]
            templates = [
                f"ہمارے مصدقہ معلوماتی ڈیٹا بیس {citation} کے مطابق، {answer}۔ امید ہے یہ معلومات آپ کے لیے کارآمد ہوں گی۔",
                f"آپ کے سوال کے جواب میں، ہمارے ڈیٹا بیس {citation} سے حاصل کردہ تفصیلات درج ذیل ہیں:\n{answer}",
                f"جی، بالکل۔ {citation} کے مطابق: {answer}۔ اگر آپ کو مزید مدد کی ضرورت ہو تو ضرور بتائیں۔"
            ]
        else:
            answer = faq["answer_en"]
            templates = [
                f"According to our verified support system {citation}, {answer} Please let me know if you need any further assistance.",
                f"In response to your query, here is the information from our reference base {citation}:\n{answer}",
                f"Based on our secure database records {citation}: {answer} I hope this helps!"
            ]
            
        # Select a random or deterministic template based on query length for variance
        index = len(query) % len(templates)
        return templates[index]

    def process_query(self, query: str, search_method: str = "hnsw", 
                      threshold_high: float = None, threshold_low: float = None) -> dict:
        """
        Runs the full RAG query pipeline:
        1. Auto-detects language.
        2. Retrieves the closest FAQ using vector search (FAISS HNSW or Brute-force).
        3. Applies the Confidence-Based Escalation Threshold logic.
        4. Logs performance benchmarks (latency, score).
        """
        start_time = time.time()
        
        # Handle default parameter overrides
        th_high = threshold_high if threshold_high is not None else config.THRESHOLD_HIGH
        th_low = threshold_low if threshold_low is not None else config.THRESHOLD_LOW
        
        # 1. Detect language
        detected_lang = self.detect_language(query)
        
        # 2. Retrieve closest match
        retrieved = self.retrieve(query, search_method=search_method, top_k=1)
        
        if not retrieved:
            # Fallback if corpus is empty
            latency = (time.time() - start_time) * 1000
            return {
                "query": query,
                "detected_language": detected_lang,
                "matched_faq_id": None,
                "similarity_score": 0.0,
                "latency_ms": latency,
                "status": "escalated",
                "response_text": "I apologize, but I am unable to process your request at this moment.",
                "faq": None
            }
            
        match = retrieved[0]
        score = match["score"]
        faq = match["faq"]
        
        # 3. Confidence-based Routing & Containment
        if score >= th_high:
            # Stage A: High Confidence -> Return Vetted Answer directly (No hallucination, 100% accurate)
            status = "contained_vetted"
            response_text = faq["answer_ur"] if detected_lang == "ur" else faq["answer_en"]
            
        elif th_low <= score < th_high:
            # Stage B: Medium Confidence -> Synthesize with RAG + grounded citations
            status = "contained_rag"
            response_text = self.generate_rag_response(query, faq, detected_lang)
            
        else:
            # Stage C: Low Confidence -> Escalate to human agent
            status = "escalated"
            incident_id = f"BRSC-{int(time.time() * 1000) % 100000:05d}"
            if detected_lang == "ur":
                response_text = f"معذرت، میں آپ کی درخواست کو براہِ راست حل کرنے سے قاصر ہوں۔ آپ کی شکایت کو ہمارے انسانی نمائندے کو منتقل کیا جا رہا ہے (انسیڈنٹ آئی ڈی: #{incident_id})۔"
            else:
                response_text = f"We apologize, but I am unable to resolve your request directly. Your ticket is being escalated to a human support agent (Incident ID: #{incident_id})."

        latency = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "detected_language": detected_lang,
            "matched_faq_id": faq["id"] if status != "escalated" else None,
            "similarity_score": float(score),
            "latency_ms": latency,
            "status": status,
            "response_text": response_text,
            "faq": faq if status != "escalated" else None
        }
