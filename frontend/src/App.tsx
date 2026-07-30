import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { ControlPanel } from './components/ControlPanel';
import { ChatWidget } from './components/ChatWidget';
import { Telemetry } from './components/Telemetry';
import { LogsTable } from './components/LogsTable';
import { mlService } from './services/mlEngine';
import { ConfigState, ChatMessage, SystemStats, RelevanceLog } from './types';

export const App: React.FC = () => {
  const [config, setConfig] = useState<ConfigState>({
    precision: 'int8',
    engine: 'hnsw',
    thresholdHigh: 0.80,
    thresholdLow: 0.50,
    isStaticSpace: false
  });

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init_1',
      sender: 'bot',
      text: 'Hello! I am your bilingual logistics and support assistant. How can I help you track packages, calculate shipping rates, or answer delivery queries today?'
    }
  ]);

  const [stats, setStats] = useState<SystemStats>({
    total_queries: 0,
    containment_rate: 100.0,
    avg_latency_ms: 0.0,
    urdu_count: 0,
    english_count: 0,
    int8_avg_latency_ms: 0.0,
    fp32_avg_latency_ms: 0.0,
    hnsw_avg_latency_ms: 0.0,
    brute_avg_latency_ms: 0.0
  });

  const [logs, setLogs] = useState<RelevanceLog[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);

  // Initialize Environment Detection & Services
  useEffect(() => {
    const initApp = async () => {
      const isStatic = await mlService.init(config);
      setConfig(prev => ({ ...prev, isStaticSpace: isStatic }));
      refreshData(isStatic);
    };

    initApp();

    const interval = setInterval(() => {
      refreshData(config.isStaticSpace);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const refreshData = async (isStatic: boolean) => {
    const currentStats = await mlService.getStats(isStatic);
    const currentLogs = await mlService.getLogs(isStatic);
    setStats(currentStats);
    setLogs(currentLogs);
  };

  const handleSendMessage = async (text: string) => {
    // Append user message
    const userMsg: ChatMessage = { id: Date.now(), sender: 'user', text };
    setMessages(prev => [...prev, userMsg]);
    setIsProcessing(true);

    try {
      const botResponse = await mlService.query(text, config);
      setMessages(prev => [...prev, botResponse]);
      await refreshData(config.isStaticSpace);
    } catch (err) {
      console.error(err);
      setMessages(prev => [
        ...prev,
        { id: Date.now(), sender: 'bot', text: 'Error connecting to ML engine.', isError: true }
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: 'init_clear',
        sender: 'bot',
        text: 'Chat history cleared. How can I assist you today?'
      }
    ]);
  };

  const handleFeedback = async (logId: number, type: 'like' | 'dislike') => {
    await mlService.recordFeedback(logId, type, config.isStaticSpace);
    await refreshData(config.isStaticSpace);
  };

  const handleRunEval = async () => {
    setIsEvaluating(true);
    setTimeout(() => {
      setIsEvaluating(false);
      refreshData(config.isStaticSpace);
    }, 1500);
  };

  const handleReindex = async () => {
    await refreshData(config.isStaticSpace);
  };

  return (
    <div className="min-h-screen bg-bgDark text-slate-100 p-4 md:p-8 font-sans relative overflow-hidden">
      {/* Background Glowing Lights */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-neonPurple/15 rounded-full blur-[128px] pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-neonCyan/15 rounded-full blur-[128px] pointer-events-none" />

      <div className="max-w-7xl mx-auto space-y-6 relative z-10">
        {/* Header */}
        <Header config={config} />

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Control Panel & Telemetry (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            <ControlPanel
              config={config}
              setConfig={setConfig}
              onRunEval={handleRunEval}
              onReindex={handleReindex}
              isEvaluating={isEvaluating}
            />
            <Telemetry stats={stats} />
          </div>

          {/* Right Column: Chat Assistant (7 cols) */}
          <div className="lg:col-span-7">
            <ChatWidget
              messages={messages}
              onSendMessage={handleSendMessage}
              onClearChat={handleClearChat}
              onFeedback={handleFeedback}
              config={config}
              isProcessing={isProcessing}
            />
          </div>
        </div>

        {/* Bottom Section: Automated Relevance Logs Table */}
        <LogsTable logs={logs} onRefresh={() => refreshData(config.isStaticSpace)} />
      </div>
    </div>
  );
};

export default App;
