import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { createSession, sendMessage, getDownloadUrl } from './api';
import './App.css';

const FIELD_LABELS = {
  task_type: { icon: '🔨', name: 'Task Type' },
  workforce: { icon: '👷', name: 'Workforce' },
  materials: { icon: '🧱', name: 'Materials' },
  equipment_and_tools: { icon: '🔧', name: 'Equipment & Tools' },
  hazard: { icon: '⚠️', name: 'Hazard' },
};

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [reportData, setReportData] = useState(null);
  const [completion, setCompletion] = useState(0);
  const [initializing, setInitializing] = useState(true);
  const [selectedMsgIndex, setSelectedMsgIndex] = useState(null);
  const [sourcesPanelOpen, setSourcesPanelOpen] = useState(true);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    initSession();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function initSession() {
    try {
      const data = await createSession();
      setSessionId(data.session_id);
      setMessages([{ role: 'assistant', content: data.response, sources: [] }]);
      setReportData(data.report_data);
      setCompletion(data.completion);
    } catch {
      setMessages([{ role: 'assistant', content: 'Failed to connect to the server. Please make sure the backend is running.', sources: [] }]);
    } finally {
      setInitializing(false);
    }
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || loading || !sessionId) return;

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setLoading(true);

    try {
      const data = await sendMessage(sessionId, text);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response, sources: data.sources || [] },
      ]);
      setReportData(data.report_data);
      setCompletion(data.completion);
      // Auto-select the latest assistant message to show its sources
      setSelectedMsgIndex(null); // will be set after render via effect
    } catch {
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Something went wrong. Please try again.', sources: [] }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  // Auto-select latest assistant message with sources
  useEffect(() => {
    if (messages.length === 0) return;
    const lastIdx = messages.length - 1;
    const last = messages[lastIdx];
    if (last.role === 'assistant' && last.sources && last.sources.length > 0) {
      setSelectedMsgIndex(lastIdx);
    }
  }, [messages]);

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleNewReport() {
    setMessages([]);
    setReportData(null);
    setCompletion(0);
    setSelectedMsgIndex(null);
    setInitializing(true);
    initSession();
  }

  function handleMsgClick(index) {
    const msg = messages[index];
    if (msg.role === 'assistant' && msg.sources && msg.sources.length > 0) {
      setSelectedMsgIndex(selectedMsgIndex === index ? null : index);
    }
  }

  const selectedSources =
    selectedMsgIndex !== null && messages[selectedMsgIndex]
      ? messages[selectedMsgIndex].sources || []
      : [];

  const allFields = reportData
    ? [
        ...Object.entries(reportData.activity || {}),
        ...Object.entries(reportData.safety || {}),
      ]
    : [];

  const filledCount = allFields.filter(([, f]) => f.filled).length;

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1>Daily Report</h1>
        </div>
        <button className="btn-new" onClick={handleNewReport}>
          + New Report
        </button>
      </header>

      <div className="main">
        {/* Sources Panel */}
        <div className={`sources-panel ${sourcesPanelOpen ? '' : 'collapsed'}`}>
          <div className="sources-header">
            <h2>{'Relevant Sources' }</h2>
            <button
              className="sources-toggle"
              onClick={() => setSourcesPanelOpen(!sourcesPanelOpen)}
              title={sourcesPanelOpen ? 'Collapse panel' : 'Expand panel'}
            >
              {sourcesPanelOpen ? '◀' : '▶'}
            </button>
          </div>
          {selectedSources.length > 0 ? (
            <div className="sources-list">
              {selectedSources.map((src, i) => (
                <SourceCard key={i} source={src} index={i} />
              ))}
            </div>
          ) : (
            <div className="sources-empty">
              <div className="sources-empty-icon">📚</div>
              <p>Click on an AI response to view the supporting documents.</p>
            </div>
          )}
        </div>

        {/* Chat Panel */}
        <div className="chat-panel">
          <div className="messages">
            {initializing && (
              <div className="msg-row assistant">
                <div className="msg-bubble assistant">
                  <div className="typing-indicator">
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            )}
            {messages.map((msg, i) => {
              const isAssistant = msg.role === 'assistant';
              const hasSources = isAssistant && msg.sources && msg.sources.length > 0;
              const isSelected = selectedMsgIndex === i;
              return (
                <div key={i} className={`msg-row ${msg.role}`}>
                  {isAssistant && <div className="avatar assistant-avatar">AI</div>}
                  <div
                    className={`msg-bubble ${msg.role}${hasSources ? ' has-sources' : ''}${isSelected ? ' selected' : ''}`}
                    onClick={() => handleMsgClick(i)}
                    title={hasSources ? 'Click to view supporting documents' : undefined}
                  >
                    <ReactMarkdown>{String(msg.content || '')}</ReactMarkdown>
                    {hasSources && (
                      <div className="msg-sources-badge">
                         {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}
                      </div>
                    )}
                  </div>
                  {msg.role === 'user' && <div className="avatar user-avatar">You</div>}
                </div>
              );
            })}
            {loading && (
              <div className="msg-row assistant">
                <div className="avatar assistant-avatar">AI</div>
                <div className="msg-bubble assistant">
                  <div className="typing-indicator">
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe your day's work..."
              rows={1}
              disabled={loading || initializing}
            />
            <button
              className="btn-send"
              onClick={handleSend}
              disabled={loading || !input.trim() || initializing}
            >
              Send
            </button>
          </div>
        </div>

        {/* Report Panel */}
        <div className="report-panel">
          <div className="report-header">
            <h2>Report Progress</h2>
            <div className="progress-ring">
              <span className="progress-number">{filledCount}/{allFields.length}</span>
            </div>
          </div>

          <div className="progress-bar-container">
            <div
              className="progress-bar-fill"
              style={{ width: `${completion * 100}%` }}
            />
          </div>

          <div className="fields-list">
            <h3>Activity</h3>
            {reportData &&
              Object.entries(reportData.activity || {}).map(([key, field]) => (
                <FieldCard key={key} fieldKey={key} field={field} />
              ))}

            <h3>Safety</h3>
            {reportData &&
              Object.entries(reportData.safety || {}).map(([key, field]) => (
                <FieldCard key={key} fieldKey={key} field={field} />
              ))}
          </div>

          <button
            className="btn-download"
            disabled={filledCount === 0}
            onClick={() => window.open(getDownloadUrl(sessionId), '_blank')}
          >
            Download Report (PDF)
          </button>
        </div>
      </div>
    </div>
  );
}

function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);
  const displayName = source.file_name
    ? source.file_name.replace(/\.[^/.]+$/, '').replace(/_/g, ' ')
    : 'Unknown Document';
  const scorePercent = source.score != null ? Math.round(source.score * 100) : null;

  return (
    <div className={`source-card ${expanded ? 'expanded' : ''}`} onClick={() => setExpanded(!expanded)}>
      <div className="source-card-header">
        <span className="source-index">#{index + 1}</span>
        <span className="source-name" title={source.file_name}>{displayName}</span>
        {scorePercent !== null && (
          <span className="source-score">{scorePercent}%</span>
        )}
      </div>
      {source.page_label && (
        <div className="source-page">Page {source.page_label}</div>
      )}
      <div className={`source-text ${expanded ? 'expanded' : ''}`}>
        {source.text}
      </div>
      <div className="source-expand-hint">
        {expanded ? 'Click to collapse' : 'Click to expand'}
      </div>
    </div>
  );
}

function FieldCard({ fieldKey, field }) {
  const meta = FIELD_LABELS[fieldKey] || { icon: '📄', name: field.label };
  const displayValue = formatFieldValue(field.value);
  return (
    <div className={`field-card ${field.filled ? 'filled' : ''}`}>
      <div className="field-card-header">
        <span className="field-icon">{meta.icon}</span>
        <span className="field-name">{meta.name}</span>
        <span className={`field-status ${field.filled ? 'filled' : ''}`}>
          {field.filled ? 'Collected' : 'Pending'}
        </span>
      </div>
      {field.filled && <p className="field-value">{displayValue}</p>}
    </div>
  );
}

function formatFieldValue(value) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);

  if (Array.isArray(value)) {
    return value.map((item) => formatFieldValue(item)).join(', ');
  }

  if (typeof value === 'object') {
    return Object.entries(value)
      .map(([key, val]) => `${humanizeKey(key)}: ${formatFieldValue(val)}`)
      .join('; ');
  }

  return String(value);
}

function humanizeKey(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
