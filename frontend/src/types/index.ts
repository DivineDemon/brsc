export interface FAQItem {
  id: string;
  category: string;
  question_en: string;
  answer_en: string;
  question_ur: string;
  answer_ur: string;
  tags: string[];
}

export interface RelevanceLog {
  id: number;
  timestamp: string;
  query: string;
  detected_language: 'en' | 'ur';
  matched_faq_id: string | null;
  similarity_score: number | null;
  latency_ms: number;
  search_method: 'hnsw' | 'brute_force';
  precision_mode: 'int8' | 'fp32';
  status: 'contained_vetted' | 'contained_rag' | 'escalated';
  feedback: 'like' | 'dislike' | null;
}

export interface SystemStats {
  total_queries: number;
  containment_rate: number;
  avg_latency_ms: number;
  urdu_count: number;
  english_count: number;
  int8_avg_latency_ms: number;
  fp32_avg_latency_ms: number;
  hnsw_avg_latency_ms: number;
  brute_avg_latency_ms: number;
}

export interface EvalRun {
  timestamp: string;
  total_cases: number;
  accuracy: number;
  containment_rate: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  precision_mode: string;
  search_method: string;
}

export interface ConfigState {
  precision: 'int8' | 'fp32';
  engine: 'hnsw' | 'brute_force';
  thresholdHigh: number;
  thresholdLow: number;
  isStaticSpace: boolean;
}

export interface ChatMessage {
  id: string | number;
  sender: 'user' | 'bot';
  text: string;
  isError?: boolean;
  meta?: {
    status: 'contained_vetted' | 'contained_rag' | 'escalated';
    language: 'en' | 'ur';
    latencyMs: number;
    score: number | null;
    matchedFaqId: string | null;
    faq?: FAQItem | null;
  };
}
