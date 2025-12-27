/**
 * LLM Safety Control Dashboard - Frontend JavaScript  
 * Enhanced with markdown rendering, connection status, and smart scrolling
 */

// === State ===
let sessionId = null;
let riskHistory = [];
let isLoading = false;
let isConnected = false;
let isUserScrolling = false;

// === DOM Elements ===
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const newSessionBtn = document.getElementById('new-session');
const sessionBadge = document.getElementById('session-id');
const turnCounter = document.getElementById('turn-counter');

// Risk meters
const instantRiskValue = document.getElementById('instant-risk-value');
const instantRiskBar = document.getElementById('instant-risk-bar');
const accumulatedRiskValue = document.getElementById('accumulated-risk-value');
const accumulatedRiskBar = document.getElementById('accumulated-risk-bar');
const effectiveRiskValue = document.getElementById('effective-risk-value');
const effectiveRiskBar = document.getElementById('effective-risk-bar');

// Control displays
const temperatureDisplay = document.getElementById('temperature-display');
const tempBar = document.getElementById('temp-bar');
const biasStatus = document.getElementById('bias-status');
const biasStrength = document.getElementById('bias-strength');
const biasContainer = document.getElementById('bias-container');

// Component bars
const toxicityBar = document.getElementById('toxicity-bar');
const toxicityValue = document.getElementById('toxicity-value');
const harmBar = document.getElementById('harm-bar');
const harmValue = document.getElementById('harm-value');
const manipulationBar = document.getElementById('manipulation-bar');
const manipulationValue = document.getElementById('manipulation-value');
const escalationBar = document.getElementById('escalation-bar');
const escalationValue = document.getElementById('escalation-value');

const historyChart = document.getElementById('history-chart');

// Laminar Framework elements
const flowStateEl = document.getElementById('flow-state');
const flowDescEl = document.getElementById('flow-desc');
const turbulenceBar = document.getElementById('turbulence-bar');
const turbulenceValue = document.getElementById('turbulence-value');
const energyBar = document.getElementById('energy-bar');
const energyValue = document.getElementById('energy-value');
const marginBar = document.getElementById('margin-bar');
const marginValue = document.getElementById('margin-value');
const riskSignalBar = document.getElementById('risk-signal-bar');
const riskSignalValue = document.getElementById('risk-signal-value');
const dcbfStatus = document.getElementById('dcbf-status');
const brakeStatus = document.getElementById('brake-status');
const decoderMode = document.getElementById('decoder-mode');

// Reynolds Number elements
const reynoldsValue = document.getElementById('reynolds-value');
const reynoldsBar = document.getElementById('reynolds-bar');
const riskLevelBadge = document.getElementById('risk-level-badge');
const flowRegimeBadge = document.getElementById('flow-regime-badge');


// === Smart Scrolling ===
function isNearBottom() {
    return chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight < 100;
}

function scrollToBottom() {
    if (!isUserScrolling) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

chatMessages.addEventListener('scroll', () => {
    isUserScrolling = !isNearBottom();
});


// === Markdown Rendering (Simple) ===
function renderMarkdown(text) {
    let html = text;
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/\n/g, '<br>');
    return html;
}


// === API Functions ===
async function sendMessage(message) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API request failed');
    }

    return await response.json();
}

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        return await response.json();
    } catch (e) {
        console.error('Health check failed:', e);
        return null;
    }
}


// === UI Functions ===
function addMessage(content, role, isMarkdown = false) {
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `<div class="message-content">${isMarkdown ? renderMarkdown(content) : escapeHtml(content)}</div>`;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function addTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message assistant typing';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    chatMessages.appendChild(indicator);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

function updateRiskDashboard(metrics) {
    updateMeter(instantRiskValue, instantRiskBar, metrics.instant_risk);
    updateMeter(accumulatedRiskValue, accumulatedRiskBar, metrics.accumulated_risk);
    updateMeter(effectiveRiskValue, effectiveRiskBar, metrics.effective_risk);

    temperatureDisplay.textContent = metrics.temperature.toFixed(2);
    tempBar.style.width = `${metrics.temperature * 100}%`;

    if (metrics.temperature > 0.7) {
        temperatureDisplay.style.color = '#22c55e';
        tempBar.style.background = 'linear-gradient(90deg, #22c55e, #10b981)';
    } else if (metrics.temperature > 0.4) {
        temperatureDisplay.style.color = '#f59e0b';
        tempBar.style.background = 'linear-gradient(90deg, #f59e0b, #eab308)';
    } else {
        temperatureDisplay.style.color = '#ef4444';
        tempBar.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
    }

    if (metrics.bias_applied) {
        biasStatus.textContent = 'ACTIVE';
        biasStatus.className = 'bias-status active';
        biasContainer.style.borderColor = '#ef4444';
        biasContainer.style.boxShadow = '0 0 20px rgba(239, 68, 68, 0.3)';
    } else {
        biasStatus.textContent = 'INACTIVE';
        biasStatus.className = 'bias-status inactive';
        biasContainer.style.borderColor = 'transparent';
        biasContainer.style.boxShadow = 'none';
    }
    biasStrength.textContent = `Strength: ${metrics.bias_strength.toFixed(1)}`;

    updateComponent(toxicityBar, toxicityValue, metrics.risk_components.toxicity);
    updateComponent(harmBar, harmValue, metrics.risk_components.harm_potential);
    updateComponent(manipulationBar, manipulationValue, metrics.risk_components.manipulation);
    updateComponent(escalationBar, escalationValue, metrics.risk_components.escalation);

    turnCounter.textContent = `Turn: ${metrics.turn_number}`;

    riskHistory.push(metrics.effective_risk);
    updateHistoryChart();

    // Update Laminar Framework metrics
    if (metrics.laminar) {
        updateLaminarDashboard(metrics.laminar);
    }
}

function updateLaminarDashboard(laminar) {
    // Update flow state indicator
    const flowStates = {
        'LAMINAR': { class: 'laminar', desc: 'Smooth, safe trajectory' },
        'TRANSITIONAL': { class: 'transitional', desc: 'Approaching turbulence - monitoring' },
        'TURBULENT': { class: 'turbulent', desc: 'Adversarial or chaotic trajectory detected' },
        'CRITICAL': { class: 'critical', desc: '⚠️ Immediate intervention required' },
        'UNKNOWN': { class: 'laminar', desc: 'State unknown' },
    };

    const state = flowStates[laminar.flow_state] || flowStates['UNKNOWN'];
    flowStateEl.textContent = laminar.flow_state;
    flowStateEl.className = 'flow-state ' + state.class;
    flowDescEl.textContent = state.desc;

    // Update metric bars
    // Turbulence: scale 0-0.1 to 0-100%
    const turbPct = Math.min(100, (laminar.turbulence / 0.1) * 100);
    turbulenceBar.style.width = `${turbPct}%`;
    turbulenceValue.textContent = laminar.turbulence.toFixed(4);

    // Energy: scale -20 to 0 (-20 is stable, 0 is unstable)
    const energyNorm = Math.min(100, Math.max(0, (laminar.energy + 20) / 20 * 100));
    energyBar.style.width = `${energyNorm}%`;
    energyValue.textContent = laminar.energy.toFixed(2);

    // Safety Margin: 0-1 (1 = safe)
    const marginPct = Math.max(0, Math.min(100, laminar.safety_margin * 100));
    marginBar.style.width = `${marginPct}%`;
    marginValue.textContent = laminar.safety_margin.toFixed(3);

    // Risk Signal: 0-1
    const riskPct = Math.min(100, laminar.risk_signal * 100);
    riskSignalBar.style.width = `${riskPct}%`;
    riskSignalValue.textContent = laminar.risk_signal.toFixed(4);

    // Reynolds Number: 0-1 (normalized)
    const reynoldsPct = Math.min(100, (laminar.reynolds_number || 0) * 100);
    if (reynoldsBar) {
        reynoldsBar.style.width = `${reynoldsPct}%`;
    }
    if (reynoldsValue) {
        reynoldsValue.textContent = (laminar.reynolds_number || 0).toFixed(3);
    }

    // Risk Level Badge
    if (riskLevelBadge) {
        const level = (laminar.risk_level || 'safe').toLowerCase();
        riskLevelBadge.textContent = level.toUpperCase();
        riskLevelBadge.className = 'badge risk-level ' + level;
    }

    // Flow Regime Badge
    if (flowRegimeBadge) {
        const regime = (laminar.flow_regime || 'laminar').toLowerCase();
        flowRegimeBadge.textContent = regime.toUpperCase();
        flowRegimeBadge.className = 'badge flow-regime ' + regime;
    }

    // Update safety control status
    updateControlStatus(dcbfStatus, 'DCBF Filter', laminar.dcbf_engaged);
    updateControlStatus(brakeStatus, 'Semantic Brake', laminar.brake_engaged);
    updateDecoderMode(decoderMode, laminar.decoder_mode);
}

function updateControlStatus(el, label, isActive) {
    const indicator = el.querySelector('.status-indicator');
    indicator.textContent = isActive ? 'ON' : 'OFF';
    indicator.className = 'status-indicator ' + (isActive ? 'active' : 'inactive');
}

function updateDecoderMode(el, mode) {
    const indicator = el.querySelector('.status-indicator');
    indicator.textContent = mode;
    indicator.className = 'status-indicator ' + mode.replace('-', '_');
}

function updateMeter(valueEl, barEl, value) {
    valueEl.textContent = value.toFixed(3);
    barEl.style.width = `${value * 100}%`;

    if (value > 0.7) {
        valueEl.style.color = '#ef4444';
    } else if (value > 0.4) {
        valueEl.style.color = '#f59e0b';
    } else {
        valueEl.style.color = '#6366f1';
    }
}

function updateComponent(barEl, valueEl, value) {
    barEl.style.width = `${value * 100}%`;
    valueEl.textContent = value.toFixed(2);

    if (value > 0.5) {
        barEl.style.background = '#ef4444';
    } else if (value > 0.3) {
        barEl.style.background = '#f59e0b';
    } else {
        barEl.style.background = '#6366f1';
    }
}

function updateHistoryChart() {
    historyChart.innerHTML = '';
    const displayHistory = riskHistory.slice(-20);

    displayHistory.forEach((risk) => {
        const bar = document.createElement('div');
        bar.className = 'history-bar';
        bar.style.height = `${Math.max(risk * 100, 5)}%`;

        if (risk > 0.7) bar.classList.add('danger');
        else if (risk > 0.4) bar.classList.add('warning');
        else bar.classList.add('safe');

        historyChart.appendChild(bar);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function setLoading(loading) {
    isLoading = loading;
    sendBtn.disabled = loading;
    chatInput.disabled = loading;
    sendBtn.innerHTML = loading ? '<span>Sending...</span>' : '<span>Send</span><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>';
}

function resetSession() {
    sessionId = null;
    riskHistory = [];
    isUserScrolling = false;

    chatMessages.innerHTML = `
        <div class="welcome-message">
            <h3>🛡️ Welcome to the Safety Control Demo</h3>
            <p>Chat with the AI and watch the risk metrics update in real-time.</p>
            <p>Try escalating your questions gradually to see how the system responds!</p>
            <div class="example-prompts">
                <p><strong>Try these:</strong></p>
                <button class="example-btn" data-prompt="Hello! How are you today?">👋 Normal greeting</button>
                <button class="example-btn" data-prompt="What's 5+5? Then multiply that by 2">🧮 Test memory</button>
                <button class="example-btn" data-prompt="Tell me about chemistry">🧪 Start neutral</button>
            </div>
        </div>
    `;

    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            chatInput.value = btn.dataset.prompt;
            chatInput.focus();
        });
    });

    updateMeter(instantRiskValue, instantRiskBar, 0);
    updateMeter(accumulatedRiskValue, accumulatedRiskBar, 0);
    updateMeter(effectiveRiskValue, effectiveRiskBar, 0);

    temperatureDisplay.textContent = '0.90';
    tempBar.style.width = '90%';
    tempBar.style.background = 'linear-gradient(90deg, #22c55e, #10b981)';
    temperatureDisplay.style.color = '#22c55e';

    biasStatus.textContent = 'INACTIVE';
    biasStatus.className = 'bias-status inactive';
    biasStrength.textContent = 'Strength: 0.0';

    updateComponent(toxicityBar, toxicityValue, 0);
    updateComponent(harmBar, harmValue, 0);
    updateComponent(manipulationBar, manipulationValue, 0);
    updateComponent(escalationBar, escalationValue, 0);

    turnCounter.textContent = 'Turn: 0';
    sessionBadge.textContent = 'No Session';
    historyChart.innerHTML = '<div class="chart-placeholder">Risk history will appear here</div>';

    // Reset Laminar metrics
    if (flowStateEl) {
        flowStateEl.textContent = 'LAMINAR';
        flowStateEl.className = 'flow-state laminar';
        flowDescEl.textContent = 'Smooth, safe trajectory';
        turbulenceBar.style.width = '0%';
        turbulenceValue.textContent = '0.000';
        energyBar.style.width = '0%';
        energyValue.textContent = '0.000';
        marginBar.style.width = '100%';
        marginValue.textContent = '1.000';
        riskSignalBar.style.width = '0%';
        riskSignalValue.textContent = '0.000';
        updateControlStatus(dcbfStatus, 'DCBF Filter', false);
        updateControlStatus(brakeStatus, 'Semantic Brake', false);
        updateDecoderMode(decoderMode, 'normal');
    }
}

function showConnectionStatus(backend) {
    const statusDiv = document.createElement('div');
    statusDiv.className = 'connection-status';
    statusDiv.innerHTML = `
        <div class="status-indicator ${isConnected ? 'connected' : 'disconnected'}"></div>
        <span>${isConnected ? '✅ Connected' : '❌ Disconnected'} (${backend || 'unknown'})</span>
    `;

    const existing = document.querySelector('.connection-status');
    if (existing) {
        existing.replaceWith(statusDiv);
    } else {
        document.querySelector('.header-actions').prepend(statusDiv);
    }
}


// === Event Handlers ===
async function handleSend() {
    const message = chatInput.value.trim();
    if (!message || isLoading) return;

    chatInput.value = '';
    chatInput.style.height = 'auto';
    isUserScrolling = false; // Reset so new messages scroll to bottom

    addMessage(message, 'user');
    setLoading(true);
    addTypingIndicator();

    try {
        const response = await sendMessage(message);
        removeTypingIndicator();

        sessionId = response.session_id;
        sessionBadge.textContent = sessionId.substring(0, 8) + '...';
        sessionBadge.title = sessionId;

        addMessage(response.response, 'assistant', true);
        updateRiskDashboard(response.risk_metrics);

    } catch (error) {
        removeTypingIndicator();

        let errorMessage = error.message;
        let errorDetails = '';

        if (errorMessage.includes('Failed to fetch')) {
            errorMessage = 'Cannot connect to server. Is it running?';
            errorDetails = 'Make sure the server is running on port 8000';
        } else if (errorMessage.includes('Ollama')) {
            errorMessage = 'Ollama is not responding';
            errorDetails = 'Make sure Ollama is running: ollama serve';
        }

        const errorDiv = document.createElement('div');
        errorDiv.className = 'message error-message';
        errorDiv.innerHTML = `
            <div class="message-content">
                <div class="error-icon">⚠️</div>
                <div class="error-text">
                    <strong>${errorMessage}</strong>
                    ${errorDetails ? `<p>${errorDetails}</p>` : ''}
                </div>
            </div>
        `;
        chatMessages.appendChild(errorDiv);
        scrollToBottom();

        isConnected = false;
        showConnectionStatus('error');
    }

    setLoading(false);
    chatInput.focus();
}


// === Initialize ===
document.addEventListener('DOMContentLoaded', async () => {
    const health = await checkHealth();
    if (health) {
        console.log('✅ API Health:', health);
        isConnected = true;
        showConnectionStatus(health.backend || 'ollama');

        if (health.backend === 'ollama') {
            console.log('🦙 Using Ollama backend');
        }
    } else {
        console.error('❌ API health check failed');
        isConnected = false;
        showConnectionStatus('disconnected');

        addMessage(
            '⚠️ Cannot connect to server. Make sure:\n' +
            '1. Server is running: python server.py\n' +
            '2. Ollama is running: ollama serve',
            'assistant'
        );
    }

    sendBtn.addEventListener('click', handleSend);

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
    });

    newSessionBtn.addEventListener('click', resetSession);

    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            chatInput.value = btn.dataset.prompt;
            chatInput.focus();
        });
    });

    chatInput.focus();
});
