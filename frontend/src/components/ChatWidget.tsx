import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, ThumbsUp, ThumbsDown, Trash2, Clock, Target, Quote, Sparkles } from 'lucide-react';
import { ChatMessage, ConfigState } from '../types';

interface ChatWidgetProps {
  messages: ChatMessage[];
  onSendMessage: (msg: string) => void;
  onClearChat: () => void;
  onFeedback: (logId: number, type: 'like' | 'dislike') => void;
  config: ConfigState;
  isProcessing: boolean;
}

export const ChatWidget: React.FC<ChatWidgetProps> = ({
  messages,
  onSendMessage,
  onClearChat,
  onFeedback,
  config,
  isProcessing
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;
    onSendMessage(input.trim());
    setInput('');
  };

  return (
    <div className="bg-cardBg backdrop-blur-xl border border-cardBorder rounded-2xl flex flex-col h-[620px] shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-cardBorder bg-slate-900/50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-neonPurple/20 rounded-xl border border-neonPurple/40">
            <MessageSquare className="w-4 h-4 text-neonPurple" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">Bilingual Customer Support Assistant</h2>
            <p className="text-[11px] text-slate-400 font-mono">
              {config.isStaticSpace ? 'Engine: WebAssembly (HF Static Space)' : `Mode: ${config.precision.toUpperCase()} ONNX & ${config.engine.toUpperCase()}`}
            </p>
          </div>
        </div>

        <button
          onClick={onClearChat}
          className="p-2 text-slate-400 hover:text-rose-400 hover:bg-slate-800/80 rounded-xl transition-all"
          title="Clear Chat"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Messages Stream */}
      <div className="flex-1 p-4 overflow-y-auto space-y-4">
        {messages.map((msg, index) => (
          <div
            key={msg.id || index}
            className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'} space-y-1.5`}
          >
            {/* Metadata bar for bot response */}
            {msg.sender === 'bot' && msg.meta && (
              <div className="flex items-center gap-2 text-[10px] font-mono mb-1">
                <span
                  className={`px-2 py-0.5 rounded-full font-bold uppercase ${
                    msg.meta.status === 'contained_vetted'
                      ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-500/40'
                      : msg.meta.status === 'contained_rag'
                      ? 'bg-cyan-950/80 text-cyan-400 border border-cyan-500/40'
                      : 'bg-rose-950/80 text-rose-400 border border-rose-500/40'
                  }`}
                >
                  {msg.meta.status === 'contained_vetted' ? 'Vetted FAQ' : msg.meta.status === 'contained_rag' ? 'Gen RAG' : 'Escalated'}
                </span>

                <span className="px-2 py-0.5 bg-slate-900 border border-slate-800 rounded-full text-slate-300">
                  {msg.meta.language === 'ur' ? 'Urdu 🇵🇰' : 'English 🇬🇧'}
                </span>

                <span className="flex items-center gap-1 text-slate-400">
                  <Clock className="w-3 h-3 text-slate-500" />
                  {msg.meta.latencyMs.toFixed(1)}ms
                </span>

                {msg.meta.score !== null && msg.meta.status !== 'escalated' && (
                  <span className="flex items-center gap-1 text-slate-400">
                    <Target className="w-3 h-3 text-neonCyan" />
                    {(msg.meta.score * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            )}

            {/* Message Bubble */}
            <div
              className={`max-w-[85%] rounded-2xl p-3.5 text-xs leading-relaxed shadow-lg ${
                msg.sender === 'user'
                  ? 'bg-gradient-to-r from-neonPurple to-indigo-600 text-white rounded-br-none'
                  : msg.isError
                  ? 'bg-rose-950/80 border border-rose-500/50 text-rose-200 rounded-bl-none'
                  : 'bg-slate-900/90 border border-slate-800 text-slate-100 rounded-bl-none'
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.text}</div>

              {/* RAG Citation Card if present */}
              {msg.meta?.status === 'contained_rag' && msg.meta.faq && (
                <div className="mt-2.5 p-2.5 bg-slate-950/80 border border-cyan-500/30 rounded-xl text-[11px] text-slate-300 space-y-1">
                  <div className="flex items-center gap-1.5 text-cyan-400 font-semibold text-[10px]">
                    <Quote className="w-3 h-3" />
                    Source Reference [{msg.meta.matchedFaqId}]
                  </div>
                  <div className="italic text-slate-400">
                    "{msg.meta.faq.question_en}"
                  </div>
                </div>
              )}
            </div>

            {/* Footer with actions for bot */}
            {msg.sender === 'bot' && msg.meta && (
              <div className="flex items-center justify-between w-full max-w-[85%] px-1 text-[10px] text-slate-500">
                <span>Engine: {config.isStaticSpace ? 'WASM' : 'FastAPI'}</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => onFeedback(Number(msg.id), 'like')}
                    className="p-1 hover:text-emerald-400 transition-colors"
                    title="Thumbs Up"
                  >
                    <ThumbsUp className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => onFeedback(Number(msg.id), 'dislike')}
                    className="p-1 hover:text-rose-400 transition-colors"
                    title="Thumbs Down"
                  >
                    <ThumbsDown className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="flex items-center gap-2 p-3 bg-slate-900/80 border border-slate-800 rounded-2xl w-max text-xs text-slate-400">
            <Sparkles className="w-4 h-4 text-neonCyan animate-spin" />
            <span>Computing embedding & searching vector index...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-cardBorder bg-slate-950/60 flex items-center gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask a question in English or Urdu (e.g. 'Where is my parcel?')..."
          className="flex-1 bg-slate-900 border border-slate-800 focus:border-neonPurple rounded-xl px-4 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none transition-all"
        />
        <button
          type="submit"
          disabled={!input.trim() || isProcessing}
          className="p-2.5 bg-gradient-to-r from-neonPurple to-neonCyan text-white rounded-xl shadow-lg shadow-neonPurple/20 hover:opacity-90 active:scale-95 transition-all disabled:opacity-50"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
};
