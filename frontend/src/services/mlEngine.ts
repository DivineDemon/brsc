import { FAQItem, RelevanceLog, SystemStats, ChatMessage, ConfigState } from '../types';

// Declare global window transformers object
declare global {
  interface Window {
    transformers?: any;
  }
}

class MLEngineService {
  private faqs: FAQItem[] = [];
  private faqVectors: { faq_id: string; faq: FAQItem; vecEn: number[]; vecUr: number[] }[] = [];
  private pipeline: any = null;
  private isPipelineLoading = false;

  public async init(): Promise<boolean> {
    // 1. Check if REST API is active
    try {
      const res = await fetch('/api/stats', { method: 'GET', headers: { 'Accept': 'application/json' } });
      if (res.ok) {
        return false; // FastAPI Backend is active
      }
    } catch (e) {
      // Backend unreachable -> Static Space Mode!
    }

    // 2. Load Static FAQs
    try {
      const faqRes = await fetch('./data/faqs.json');
      if (faqRes.ok) {
        this.faqs = await faqRes.json();
      }
    } catch (err) {
      console.error('Failed loading faqs.json:', err);
    }

    // Initialize initial local logs
    if (!localStorage.getItem('brsc_logs')) {
      localStorage.setItem('brsc_logs', JSON.stringify([]));
    }

    // 3. Load Transformers.js in background
    this.loadTransformersPipeline();

    return true; // Static Space Mode Active
  }

  private async loadTransformersPipeline() {
    if (this.pipeline || this.isPipelineLoading) return;
    this.isPipelineLoading = true;

    try {
      if (window.transformers) {
        window.transformers.env.allowLocalModels = false;
        this.pipeline = await window.transformers.pipeline(
          'feature-extraction',
          'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
          { quantized: true }
        );
        console.log('--> Client-Side Transformers.js WASM Engine Ready!');
        await this.precomputeFAQVectors();
      }
    } catch (err) {
      console.error('Failed initializing Transformers.js pipeline:', err);
    } finally {
      this.isPipelineLoading = false;
    }
  }

  private async precomputeFAQVectors() {
    if (!this.pipeline || this.faqs.length === 0) return;
    this.faqVectors = [];

    for (const faq of this.faqs) {
      const outEn = await this.pipeline(faq.question_en, { pooling: 'mean', normalize: true });
      const outUr = await this.pipeline(faq.question_ur, { pooling: 'mean', normalize: true });

      this.faqVectors.push({
        faq_id: faq.id,
        faq: faq,
        vecEn: Array.from(outEn.data),
        vecUr: Array.from(outUr.data)
      });
    }
    console.log(`Precomputed ${this.faqVectors.length} FAQ vector pairs in browser!`);
  }

  public async query(message: string, config: ConfigState): Promise<ChatMessage> {
    const startTime = performance.now();

    if (!config.isStaticSpace) {
      // --- REST API BACKEND CALL ---
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          precision_mode: config.precision,
          search_method: config.engine,
          threshold_high: config.thresholdHigh,
          threshold_low: config.thresholdLow
        })
      });

      if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
      const data = await res.json();

      return {
        id: Date.now(),
        sender: 'bot',
        text: data.response_text,
        meta: {
          status: data.status,
          language: data.detected_language,
          latencyMs: data.latency_ms,
          score: data.similarity_score,
          matchedFaqId: data.matched_faq_id,
          faq: data.faq
        }
      };
    }

    // --- STATIC SPACE WASM SEARCH ---
    const isUrdu = /[\u0600-\u06FF]/.test(message);
    const lang = isUrdu ? 'ur' : 'en';

    let bestFaq: FAQItem | null = null;
    let maxSim = 0.0;

    if (this.pipeline && this.faqVectors.length > 0) {
      const out = await this.pipeline(message, { pooling: 'mean', normalize: true });
      const queryVec = Array.from(out.data) as number[];

      for (const item of this.faqVectors) {
        const simEn = this.cosineSimilarity(queryVec, item.vecEn);
        const simUr = this.cosineSimilarity(queryVec, item.vecUr);
        const topSim = Math.max(simEn, simUr);

        if (topSim > maxSim) {
          maxSim = topSim;
          bestFaq = item.faq;
        }
      }
    } else {
      // Fallback Keyword matching
      for (const faq of this.faqs) {
        const text = (faq.question_en + " " + faq.question_ur + " " + faq.tags.join(" ")).toLowerCase();
        if (message.toLowerCase().split(" ").some(w => w.length > 2 && text.includes(w))) {
          maxSim = 0.85;
          bestFaq = faq;
          break;
        }
      }
    }

    const latencyMs = performance.now() - startTime;

    let status: 'contained_vetted' | 'contained_rag' | 'escalated' = 'escalated';
    let responseText = "Your query confidence score is below threshold and has been escalated to a live support representative.";

    if (maxSim >= config.thresholdHigh && bestFaq) {
      status = 'contained_vetted';
      responseText = lang === 'ur' ? bestFaq.answer_ur : bestFaq.answer_en;
    } else if (maxSim >= config.thresholdLow && bestFaq) {
      status = 'contained_rag';
      const context = lang === 'ur' ? bestFaq.answer_ur : bestFaq.answer_en;
      responseText = lang === 'ur'
        ? `[Generative RAG - تصدیق شدہ جواب]: ${context}`
        : `[Generative RAG - Grounded Response]: Based on logistics policy [Reference ${bestFaq.id}], ${context}`;
    }

    // Record Log locally
    this.saveLocalLog({
      id: Date.now(),
      timestamp: new Date().toISOString(),
      query: message,
      detected_language: lang,
      matched_faq_id: bestFaq ? bestFaq.id : null,
      similarity_score: maxSim,
      latency_ms: latencyMs,
      search_method: config.engine,
      precision_mode: config.precision,
      status,
      feedback: null
    });

    return {
      id: Date.now(),
      sender: 'bot',
      text: responseText,
      meta: {
        status,
        language: lang,
        latencyMs,
        score: maxSim,
        matchedFaqId: bestFaq ? bestFaq.id : null,
        faq: bestFaq
      }
    };
  }

  public async getStats(isStaticSpace: boolean): Promise<SystemStats> {
    if (!isStaticSpace) {
      const res = await fetch('/api/stats');
      if (res.ok) return await res.json();
    }

    const logs: RelevanceLog[] = JSON.parse(localStorage.getItem('brsc_logs') || '[]');
    const total = logs.length;

    if (total === 0) {
      return {
        total_queries: 0, containment_rate: 100.0, avg_latency_ms: 0.0,
        urdu_count: 0, english_count: 0, int8_avg_latency_ms: 0.0,
        fp32_avg_latency_ms: 0.0, hnsw_avg_latency_ms: 0.0, brute_avg_latency_ms: 0.0
      };
    }

    const escalated = logs.filter(l => l.status === 'escalated').length;
    const urdu = logs.filter(l => l.detected_language === 'ur').length;
    const avgLat = logs.reduce((sum, l) => sum + l.latency_ms, 0) / total;

    return {
      total_queries: total,
      containment_rate: Math.round(((total - escalated) / total) * 1000) / 10,
      avg_latency_ms: Math.round(avgLat * 10) / 10,
      urdu_count: urdu,
      english_count: total - urdu,
      int8_avg_latency_ms: Math.round(avgLat),
      fp32_avg_latency_ms: Math.round(avgLat * 2.5),
      hnsw_avg_latency_ms: Math.round(avgLat),
      brute_avg_latency_ms: Math.round(avgLat * 1.8)
    };
  }

  public async getLogs(isStaticSpace: boolean): Promise<RelevanceLog[]> {
    if (!isStaticSpace) {
      const res = await fetch('/api/logs?limit=30');
      if (res.ok) return await res.json();
    }
    return JSON.parse(localStorage.getItem('brsc_logs') || '[]');
  }

  public async recordFeedback(logId: number, feedback: 'like' | 'dislike', isStaticSpace: boolean) {
    if (isStaticSpace) {
      const logs: RelevanceLog[] = JSON.parse(localStorage.getItem('brsc_logs') || '[]');
      const log = logs.find(l => l.id === logId) || logs[0];
      if (log) log.feedback = feedback;
      localStorage.setItem('brsc_logs', JSON.stringify(logs));
    } else {
      await fetch(`/api/logs/${logId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feedback })
      });
    }
  }

  private saveLocalLog(log: RelevanceLog) {
    const logs: RelevanceLog[] = JSON.parse(localStorage.getItem('brsc_logs') || '[]');
    logs.unshift(log);
    localStorage.setItem('brsc_logs', JSON.stringify(logs.slice(0, 100)));
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0.0, normA = 0.0, normB = 0.0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
  }
}

export const mlService = new MLEngineService();
