/**
 * Laminar v5.0 - Frontend Application
 * Complete AI Safety System with all phases
 */

class LaminarChat {
    constructor() {
        // API endpoint
        this.API_BASE = '';

        // Elements
        this.chatMessages = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.chatForm = document.getElementById('chat-form');
        this.sendBtn = document.getElementById('send-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.latencyDisplay = document.getElementById('latency-display');
        this.sessionIdDisplay = document.getElementById('session-id');

        // Phase indicators
        this.phaseA = document.getElementById('phase-a');
        this.phaseT = document.getElementById('phase-t');
        this.phaseL = document.getElementById('phase-l');
        this.phaseUQ = document.getElementById('phase-uq');
        this.phase3 = document.getElementById('phase-3');

        // Risk display elements
        this.riskScore = document.getElementById('risk-score');
        this.riskZone = document.getElementById('risk-zone');
        this.riskBar = document.getElementById('risk-bar');
        this.actionBadge = document.getElementById('action-badge');

        // Component bars
        this.barAdaptive = document.getElementById('bar-adaptive');
        this.barSemantic = document.getElementById('bar-semantic');
        this.barShort = document.getElementById('bar-short');
        this.barLong = document.getElementById('bar-long');

        // Component values
        this.valAdaptive = document.getElementById('val-adaptive');
        this.valSemantic = document.getElementById('val-semantic');
        this.valShort = document.getElementById('val-short');
        this.valLong = document.getElementById('val-long');

        // Phase T elements
        this.escalationScore = document.getElementById('escalation-score');
        this.policyState = document.getElementById('policy-state');

        // Phase L elements
        this.driftScore = document.getElementById('drift-score');

        // Phase UQ elements
        this.predictionSet = document.getElementById('prediction-set');
        this.uqConfidence = document.getElementById('uq-confidence');

        // Stats
        this.statMessages = document.getElementById('stat-messages');
        this.statBlocked = document.getElementById('stat-blocked');
        this.statSteered = document.getElementById('stat-steered');
        this.statLatency = document.getElementById('stat-latency');

        // State
        this.sessionId = null;
        this.isLoading = false;
        this.stats = {
            messages: 0,
            blocked: 0,
            steered: 0,
            totalLatency: 0
        };

        // Initialize
        this.init();
    }

    init() {
        // Generate session ID
        this.sessionId = this.generateSessionId();
        this.sessionIdDisplay.textContent = this.sessionId.slice(0, 8) + '...';

        // Event listeners
        this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
        this.resetBtn.addEventListener('click', () => this.resetSession());

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });

        // Enter to send, Shift+Enter for new line
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSubmit(e);
            }
        });

        // Health check and phase status
        this.checkHealth();
    }

    generateSessionId() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.API_BASE}/health`);
            const data = await response.json();

            // Update phase indicators
            this.updatePhaseIndicators(data.phases || {});

            if (data.dolphin === 'unavailable') {
                this.addSystemMessage(
                    '⚠️ Dolphin LLM is not available. Please run: `ollama serve` and `ollama pull dolphin-llama3`'
                );
            }
        } catch (error) {
            this.addSystemMessage(
                '❌ Cannot connect to server. Please run: `python laminar_server.py`'
            );
        }
    }

    updatePhaseIndicators(phases) {
        // Phase A: Semantic
        if (phases.phase_a) {
            this.phaseA.classList.add('active');
        }

        // Phase T: Trajectory
        if (phases.phase_t) {
            this.phaseT.classList.add('active');
        }

        // Phase L: Long-Term
        if (phases.phase_l) {
            this.phaseL.classList.add('active');
        } else {
            this.phaseL.classList.add('degraded');
            this.phaseL.title = 'Phase L: Long-Term Memory (Degraded - No Redis)';
        }

        // Phase UQ: Uncertainty
        if (phases.phase_uq) {
            this.phaseUQ.classList.add('active');
        }

        // Phase 3: Governance (always active)
        this.phase3.classList.add('active');
    }

    async handleSubmit(e) {
        e.preventDefault();

        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;

        this.isLoading = true;
        this.sendBtn.disabled = true;
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';

        // Clear welcome message
        const welcomeMsg = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        // Add user message
        this.addMessage('user', message);

        // Show typing indicator
        const typingId = this.showTyping();

        try {
            const startTime = performance.now();

            const response = await fetch(`${this.API_BASE}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            const clientLatency = Math.round(performance.now() - startTime);

            // Remove typing indicator
            this.removeTyping(typingId);

            // Add assistant message
            this.addMessage('assistant', data.response, {
                zone: data.risk_zone,
                action: data.action,
                risk: data.risk_score,
                latency: data.latency_ms
            });

            // Update risk display
            this.updateRiskDisplay(data);

            // Update stats
            this.stats.messages++;
            if (data.action === 'block') this.stats.blocked++;
            if (data.action === 'steer') this.stats.steered++;
            this.stats.totalLatency += data.latency_ms;
            this.updateStats();

            // Update latency display
            this.latencyDisplay.textContent = `${Math.round(data.latency_ms)}ms pipeline, ${clientLatency}ms total`;

        } catch (error) {
            this.removeTyping(typingId);
            this.addSystemMessage(`Error: ${error.message}`);
        }

        this.isLoading = false;
        this.sendBtn.disabled = false;
        this.messageInput.focus();
    }

    addMessage(role, content, meta = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatar = role === 'user' ? '👤' : '🐬';

        let metaHTML = '';
        if (meta) {
            const zoneClass = meta.zone || 'green';
            metaHTML = `
                <div class="message-meta">
                    <span class="message-zone ${zoneClass}">${meta.zone?.toUpperCase() || 'GREEN'}</span>
                    <span>${Math.round(meta.latency || 0)}ms</span>
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(content)}</div>
                ${metaHTML}
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addSystemMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.innerHTML = `
            <div class="message-content" style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3);">
                <div class="message-text">${content}</div>
            </div>
        `;
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showTyping() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.id = id;
        typingDiv.className = 'message assistant';
        typingDiv.innerHTML = `
            <div class="message-avatar">🐬</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
        return id;
    }

    removeTyping(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    updateRiskDisplay(data) {
        // Update main gauge
        const riskPercent = Math.round(data.risk_score * 100);
        this.riskScore.textContent = data.risk_score.toFixed(2);
        this.riskBar.style.width = riskPercent + '%';

        // Update zone badge
        this.riskZone.textContent = data.risk_zone.toUpperCase();
        this.riskZone.className = `zone-badge zone-${data.risk_zone}`;

        // Update action badge
        this.actionBadge.textContent = data.action.toUpperCase();
        this.actionBadge.className = `action-badge action-${data.action}`;

        // Update risk bar color
        if (data.risk_zone === 'green') {
            this.riskBar.style.background = 'var(--zone-green)';
        } else if (data.risk_zone === 'yellow') {
            this.riskBar.style.background = 'var(--zone-yellow)';
        } else {
            this.riskBar.style.background = 'var(--zone-red)';
        }

        // Update component bars
        const components = data.components || {};
        this.setComponent('adaptive', components.adaptive || 0);
        this.setComponent('semantic', components.semantic || 0);
        this.setComponent('short', components.short_term || 0);
        this.setComponent('long', components.long_term || 0);

        // Update Phase T: Trajectory
        if (data.escalation_score !== undefined) {
            this.escalationScore.textContent = data.escalation_score.toFixed(2);
        }
        if (data.policy_state) {
            this.policyState.textContent = data.policy_state;
            this.policyState.className = `policy-badge policy-${data.policy_state}`;
        }

        // Update Phase L: Long-Term Drift
        if (data.long_term_drift !== undefined) {
            this.driftScore.textContent = data.long_term_drift.toFixed(2);
        }

        // Update Phase UQ: Uncertainty
        if (data.prediction_set) {
            const setStr = Array.isArray(data.prediction_set)
                ? data.prediction_set.join(', ')
                : JSON.stringify(data.prediction_set);
            this.predictionSet.textContent = setStr || '-';
        }
        if (data.uq_confidence !== undefined) {
            this.uqConfidence.textContent = data.uq_confidence.toFixed(2);
        }
    }

    setComponent(name, value) {
        const bar = document.getElementById(`bar-${name}`);
        const val = document.getElementById(`val-${name}`);

        if (bar) bar.style.width = (value * 100) + '%';
        if (val) val.textContent = value.toFixed(2);
    }

    updateStats() {
        this.statMessages.textContent = this.stats.messages;
        this.statBlocked.textContent = this.stats.blocked;
        this.statSteered.textContent = this.stats.steered;

        const avgLatency = this.stats.messages > 0
            ? Math.round(this.stats.totalLatency / this.stats.messages)
            : 0;
        this.statLatency.textContent = avgLatency + 'ms';
    }

    async resetSession() {
        if (this.isLoading) return;

        try {
            await fetch(`${this.API_BASE}/session/reset?session_id=${this.sessionId}`, {
                method: 'POST'
            });
        } catch (e) { }

        // Reset state
        this.sessionId = this.generateSessionId();
        this.sessionIdDisplay.textContent = this.sessionId.slice(0, 8) + '...';

        this.stats = {
            messages: 0,
            blocked: 0,
            steered: 0,
            totalLatency: 0
        };
        this.updateStats();

        // Reset risk display
        this.riskScore.textContent = '0.00';
        this.riskBar.style.width = '0%';
        this.riskZone.textContent = 'GREEN';
        this.riskZone.className = 'zone-badge zone-green';
        this.actionBadge.textContent = 'ALLOW';
        this.actionBadge.className = 'action-badge action-allow';

        this.setComponent('adaptive', 0);
        this.setComponent('semantic', 0);
        this.setComponent('short', 0);
        this.setComponent('long', 0);

        // Reset Phase T
        this.escalationScore.textContent = '0.00';
        this.policyState.textContent = 'benign';
        this.policyState.className = 'policy-badge policy-benign';

        // Reset Phase L
        this.driftScore.textContent = '0.00';

        // Reset Phase UQ
        this.predictionSet.textContent = '-';
        this.uqConfidence.textContent = '-';

        // Clear messages
        this.chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">🐬</div>
                <h2>Session Reset</h2>
                <p>New session started. Memory cleared.</p>
            </div>
        `;

        this.latencyDisplay.textContent = 'Ready';
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.laminar = new LaminarChat();
});
