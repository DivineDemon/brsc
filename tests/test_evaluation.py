import os
import sys
import time
import logging

# Ensure backend directory is in python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from app.core import config, database, models, model, retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_harness")

def run_local_evaluation():
    """
    Standalone Evaluation Harness.
    Loads the FAQ corpus, builds/checks the FAISS HNSW indexes,
    and runs a battery of Urdu and English test cases to measure
    Retrieval Accuracy, Containment Rate, and Latency Metrics.
    
    Serves as an automated model-release check.
    """
    logger.info("Initializing ML Resources for Evaluation...")
    start_init = time.time()
    
    # Initialize FP32 and INT8 models to warm them up
    embedder_fp32 = model.BilingualEmbedder(precision="fp32")
    retriever_fp32 = retriever.FAQRetriever(embedder_fp32)
    
    embedder_int8 = model.BilingualEmbedder(precision="int8")
    retriever_int8 = retriever.FAQRetriever(embedder_int8)
    
    init_latency = (time.time() - start_init) * 1000
    logger.info(f"ML Resources loaded and warm-started in {init_latency:.2f}ms.")

    # 25 High-Fidelity Test Queries spanning different languages, categories, and expected statuses
    test_suite = [
        # --- ENGLISH IN-DOMAIN (Should map to correct FAQ) ---
        {"query": "How can I track my shipment?", "expected_id": "faq_1", "lang": "en"},
        {"query": "Show me where my package is located", "expected_id": "faq_1", "lang": "en"},
        {"query": "What are your delivery prices?", "expected_id": "faq_2", "lang": "en"},
        {"query": "Is customs tax included in shipping?", "expected_id": "faq_3", "lang": "en"},
        {"query": "Can I modify my destination address after shipping?", "expected_id": "faq_4", "lang": "en"},
        {"query": "What if I'm not home when the rider arrives?", "expected_id": "faq_5", "lang": "en"},
        {"query": "Cancel my booking", "expected_id": "faq_6", "lang": "en"},
        {"query": "My parcel is broken, I want a refund", "expected_id": "faq_7", "lang": "en"},
        {"query": "Are aerosol sprays allowed in cargo?", "expected_id": "faq_8", "lang": "en"},
        {"query": "What is the maximum weight for a parcel?", "expected_id": "faq_9", "lang": "en"},
        {"query": "Can I register for COD merchant payments?", "expected_id": "faq_10", "lang": "en"},

        # --- URDU IN-DOMAIN (Should map to correct FAQ) ---
        {"query": "میں اپنا پارسل کیسے ٹریک کر سکتا ہوں؟", "expected_id": "faq_1", "lang": "ur"},
        {"query": "ڈیلیوری چارجز کیا ہیں؟", "expected_id": "faq_2", "lang": "ur"},
        {"query": "کیا مجھے کسٹم ڈیوٹی دینی پڑے گی؟", "expected_id": "faq_3", "lang": "ur"},
        {"query": "کیا میں پیکج بھیجنے کے بعد پتہ تبدیل کر سکتا ہوں؟", "expected_id": "faq_4", "lang": "ur"},
        {"query": "اگر میں گھر پر نہ ہوں تو کورئیر کیا کرے گا؟", "expected_id": "faq_5", "lang": "ur"},
        {"query": "اپنی بکنگ منسوخ کرنے کا طریقہ بتائیں", "expected_id": "faq_6", "lang": "ur"},
        {"query": "میرا سامان کلیم کرنے کا طریقہ", "expected_id": "faq_7", "lang": "ur"},
        {"query": "کیا میں لتھیم بیٹریاں بھیج سکتا ہوں؟", "expected_id": "faq_8", "lang": "ur"},
        {"query": "پیکج کا وزن کتنا ہونا چاہیے؟", "expected_id": "faq_9", "lang": "ur"},
        {"query": "کیش آن ڈیلیوری کی فیس کیا ہے؟", "expected_id": "faq_10", "lang": "ur"},

        # --- OUT OF DOMAIN / AMBIGUOUS (Should Escalate - Containment Flow) ---
        {"query": "Who is the prime minister of Pakistan?", "expected_id": None, "lang": "en"},
        {"query": "موسم کیسا ہے آج؟", "expected_id": None, "lang": "ur"},
        {"query": "Can I order a pizza to your courier hub?", "expected_id": None, "lang": "en"},
        {"query": "I want to apply for a software engineering job", "expected_id": None, "lang": "en"}
    ]

    # Run evaluations on both FP32-Cosine and INT8-HNSW modes to output side-by-side performance benchmarks
    configurations = [
        {"name": "FP32 + Brute-force Cosine", "retriever": retriever_fp32, "precision": "fp32", "search": "brute_force"},
        {"name": "INT8 ONNX + FAISS HNSW", "retriever": retriever_int8, "precision": "int8", "search": "hnsw"}
    ]

    print("\n" + "="*80)
    print("                      BiliRAG STANDARD EVALUATION HARNESS                   ")
    print("================================================================================")

    for config_item in configurations:
        name = config_item["name"]
        active_retriever = config_item["retriever"]
        search_method = config_item["search"]
        
        print(f"\nEvaluating configuration: {BOLD_TEXT(name)}")
        print("-"*80)
        
        latencies = []
        correct_retrievals = 0
        contained_count = 0
        total_queries = len(test_suite)
        
        # Process queries
        for item in test_suite:
            query = item["query"]
            expected_id = item["expected_id"]
            
            start_q = time.time()
            res = active_retriever.process_query(query, search_method=search_method)
            q_latency = (time.time() - start_q) * 1000
            latencies.append(q_latency)
            
            # 1. Check Retrieval Accuracy
            # In-domain check
            if expected_id:
                if res["matched_faq_id"] == expected_id:
                    correct_retrievals += 1
            # Out-of-domain check (should escalate)
            else:
                if res["status"] == "escalated":
                    correct_retrievals += 1
            
            # 2. Check Containment status
            if res["status"] in ["contained_vetted", "contained_rag"]:
                contained_count += 1
                
        # Calculate Aggregated Metrics
        accuracy = (correct_retrievals / total_queries) * 100
        containment = (contained_count / total_queries) * 100
        avg_latency = sum(latencies) / total_queries
        p95_latency = sorted(latencies)[int(total_queries * 0.95)]
        
        print(f"  - Total Test Cases Checked  : {total_queries}")
        print(f"  - Retrieval Accuracy        : {accuracy:.1f}%")
        print(f"  - Bot Containment Rate      : {containment:.1f}%")
        print(f"  - Average Response Latency  : {avg_latency:.2f} ms")
        print(f"  - p95 Response Latency      : {p95_latency:.2f} ms")
        print(f"  - Validation Qualification  : {'PASSED ✅' if accuracy >= 80.0 else 'FAILED ❌'}")
        
    print("\n" + "="*80)
    print("                      EVALUATION HARNESS EXECUTION COMPLETE                 ")
    print("================================================================================")


def BOLD_TEXT(text):
    return f"\033[1m{text}\033[0m"


if __name__ == "__main__":
    run_local_evaluation()
