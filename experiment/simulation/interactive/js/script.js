/**
 * LSTM Sentiment Analysis Simulation - script.js
 * Logic for pedagogical deep learning visualizations
 */

document.addEventListener('DOMContentLoaded', () => {
    initSim1();
    initSim2();
    initSimulation3();
    initSimulation4();
    initSimulation5();
    initSimulation6();

    // Setup tab navigation
    setupTabs();
});

// --- TAB SYSTEM ---
function setupTabs() {
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            switchToTab(targetTab);
        });
    });
}

function switchToTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    // Remove active from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    // Show target tab
    const targetTab = document.getElementById(tabId);
    if (targetTab) {
        targetTab.classList.add('active');
    }
    
    // Activate button
    const targetBtn = document.querySelector(`[data-tab="${tabId}"]`);
    if (targetBtn) {
        targetBtn.classList.add('active');
    }

    // Fix Chart.js visibility issues when switching tabs
    setTimeout(() => {
        if (retentionChart) retentionChart.resize();
        if (sentimentChart) sentimentChart.resize();
        if (rnnLossChart) rnnLossChart.resize();
        if (rnnAccChart) rnnAccChart.resize();
        if (lstmLossChart) lstmLossChart.resize();
        if (lstmAccChart) lstmAccChart.resize();
        if (cellStateChartA) cellStateChartA.resize();
        if (cellStateChartB) cellStateChartB.resize();
        if (compareBarChart) compareBarChart.resize();
        if (rocChart) rocChart.resize();
        if (prChart) prChart.resize();
        // Render sim-specific content
        if (tabId === 'sim3') renderSim3();
        if (tabId === 'sim5') { renderSim5A(); renderSim5B(); }
        if (tabId === 'sim6') renderSim6();
    }, 200);

    // Scroll main content to top
    const content = targetTab.querySelector('.sim-main');
    if (content) content.scrollTo({ top: 0, behavior: 'smooth' });
}

// Keep old switchTab function for backward compatibility
function switchTab(index) {
    const tabMap = ['sim0', 'sim1', 'sim2', 'sim3', 'sim4', 'sim5', 'sim6'];
    if (index >= 0 && index < tabMap.length) {
        switchToTab(tabMap[index]);
    }
}

// --- HELPER FUNCTIONS ---
const getWords = (str) => str.split(' ').filter(w => w.length > 0);

// --- SIMULATION 1: MEMORY LOSS ---
let sim1Model = 'rnn';
let retentionChart;
// Elements will be initialized in initSim1
let sim1Input, rnnTimeline, lstmTimeline;

// New experiment controls
const rnnAlphaSlider = () => document.getElementById('rnn-alpha-slider');
const lstmBiasSlider = () => document.getElementById('lstm-bias-slider');

function initSim1() {
    // Initialize elements here after DOM is ready
    sim1Input = document.getElementById('sim1-input');
    rnnTimeline = document.getElementById('rnn-timeline');
    lstmTimeline = document.getElementById('lstm-timeline');

    if (!sim1Input) return; // Guard against missing elements

    initRetentionChart();

    // Add listeners for new sliders
    const rAlpha = rnnAlphaSlider();
    const lBias = lstmBiasSlider();

    if (rAlpha) {
        rAlpha.addEventListener('input', (e) => {
            document.getElementById('val-rnn-alpha').innerText = (e.target.value / 100).toFixed(2);
            renderSim1();
        });
    }
    if (lBias) {
        lBias.addEventListener('input', (e) => {
            document.getElementById('val-lstm-bias').innerText = (e.target.value / 100).toFixed(2);
            renderSim1();
        });
    }
    sim1Input.addEventListener('input', renderSim1);

    renderSim1();
}

function initRetentionChart() {
    const ctx = document.getElementById('retentionChart').getContext('2d');
    retentionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'RNN Retention',
                    borderColor: '#4a4ae1',
                    backgroundColor: 'rgba(74, 74, 225, 0.1)',
                    data: [],
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'LSTM Retention',
                    borderColor: '#9c27b0',
                    backgroundColor: 'rgba(156, 39, 176, 0.1)',
                    data: [],
                    fill: true,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 1, title: { display: true, text: 'Signal Strength' } },
                x: { title: { display: true, text: 'Word Position' } }
            }
        }
    });
}

function setSim1Model(model) {
    sim1Model = model;
    document.querySelectorAll('#sim1 .toggle-group button').forEach(btn => {
        btn.classList.toggle('active', btn.innerText.toLowerCase() === model);
    });
    renderSim1();
}

function renderSim1() {
    const text = sim1Input.value || "The movie started great but the ending was terrible";
    const words = getWords(text);
    const rnnAlpha = (rnnAlphaSlider()?.value || 85) / 100;
    const lstmBias = (lstmBiasSlider()?.value || 98) / 100;

    // Toggle row visibility
    document.querySelector('.rnn-row').style.display = (sim1Model === 'rnn' || sim1Model === 'both') ? 'flex' : 'none';
    document.querySelector('.lstm-row').style.display = (sim1Model === 'lstm' || sim1Model === 'both') ? 'flex' : 'none';

    rnnTimeline.innerHTML = '';
    lstmTimeline.innerHTML = '';

    let rnnMemory = 1.0;
    let lstmMemory = 1.0;
    let rnnHistory = [];
    let lstmHistory = [];

    words.forEach((word, i) => {
        const isTurningPoint = word.toLowerCase() === 'but' || word.toLowerCase() === 'however';

        // Realistic Decay Math
        rnnMemory *= rnnAlpha;
        if (isTurningPoint) rnnMemory *= 0.5; // Context shift penalty

        lstmMemory *= lstmBias;
        if (isTurningPoint) lstmMemory *= 0.95; // LSTMs are stable

        rnnHistory.push(rnnMemory);
        lstmHistory.push(lstmMemory);

        const rnnToken = createToken(word, rnnMemory, 'rnn');
        const lstmToken = createToken(word, lstmMemory, 'lstm');

        rnnTimeline.appendChild(rnnToken);
        lstmTimeline.appendChild(lstmToken);
    });

    updateRetentionChart(words, rnnHistory, lstmHistory);
}

function updateRetentionChart(labels, rnnData, lstmData) {
    if (!retentionChart) return;
    retentionChart.data.labels = labels.map((_, i) => i + 1);

    // Toggle dataset visibility
    retentionChart.data.datasets[0].hidden = (sim1Model === 'lstm');
    retentionChart.data.datasets[1].hidden = (sim1Model === 'rnn');

    retentionChart.data.datasets[0].data = rnnData;
    retentionChart.data.datasets[1].data = lstmData;
    retentionChart.update();
}

async function runBackpropPulse() {
    const rnnTokens = rnnTimeline.querySelectorAll('.token');
    const lstmTokens = lstmTimeline.querySelectorAll('.token');

    // Animate backward (Right to Left)
    for (let i = rnnTokens.length - 1; i >= 0; i--) {
        if (rnnTokens[i]) rnnTokens[i].classList.add('pulse');
        if (lstmTokens[i]) lstmTokens[i].classList.add('pulse');

        await new Promise(r => setTimeout(r, 100));

        if (rnnTokens[i]) setTimeout(() => rnnTokens[i].classList.remove('pulse'), 500);
        if (lstmTokens[i]) setTimeout(() => lstmTokens[i].classList.remove('pulse'), 500);
    }
}

function loadStressTest() {
    const btn = document.getElementById('stress-test-btn');
    if (btn) btn.disabled = true;
    sim1Input.value = "The beginning was promising and the middle had some great depth but eventually the plot became so convoluted and extremely long that by the time we reached the final scene I had completely forgotten why I even started watching this movie in the first place";
    renderSim1();
}

function createToken(text, strength, type) {
    const el = document.createElement('div');
    el.className = 'token';
    el.innerText = text;
    // Visually fade color based on memory strength
    const opacity = 0.1 + (strength * 0.9);
    const color = type === 'rnn' ? `rgba(74, 74, 225, ${opacity})` : `rgba(156, 39, 176, ${opacity})`;
    el.style.backgroundColor = color;
    el.style.color = opacity > 0.4 ? 'white' : 'black';
    el.title = `Strength: ${(strength * 100).toFixed(1)}%`;
    return el;
}

function resetSim1() {
    const btn = document.getElementById('stress-test-btn');
    if (btn) btn.disabled = false;
    sim1Input.value = "The movie started great but the ending was terrible";
    document.getElementById('rnn-alpha-slider').value = 85;
    document.getElementById('lstm-bias-slider').value = 98;
    document.getElementById('val-rnn-alpha').innerText = "0.85";
    document.getElementById('val-lstm-bias').innerText = "0.98";
    setSim1Model('rnn');
}

function initSim2() {
    const container = document.getElementById('lstm-cell-svg-container');
    container.innerHTML = `
        <svg width="100%" viewBox="0 20 1100 470" preserveAspectRatio="xMidYMid meet" id="lstm-pipeline-svg">
            <defs>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>

            <!-- Labels -->
            <text x="20" y="55" font-size="17" font-weight="bold" fill="#666">Cell Memory (C<tspan dy="5" font-size="13">t-1</tspan>)</text>
            <text x="810" y="55" font-size="17" font-weight="bold" fill="#666">Updated Memory (C<tspan dy="5" font-size="13">t</tspan>)</text>
            <text x="20" y="470" font-size="17" font-weight="bold" fill="#666">Input (p<tspan dy="5" font-size="13">t</tspan>)</text>
            <text x="950" y="470" font-size="17" font-weight="bold" fill="#666">Output (h<tspan dy="5" font-size="13">t</tspan>)</text>

            <!-- Main Pathways -->
            <path id="path-cell" d="M 0 120 L 1100 120" stroke="#eee" stroke-width="14" fill="none" />
            <path id="path-input" d="M 100 480 L 100 360 L 740 360" stroke="#eee" stroke-width="10" fill="none" />
            <path id="path-update" d="M 645 360 L 645 250 L 790 250 L 790 180" stroke="#eee" stroke-width="7" fill="none" />
            <path id="path-hidden-branch" d="M 920 120 L 920 240 L 1020 240 L 1020 330" stroke="#eee" stroke-width="7" fill="none" />
            <path id="path-output-sig-to-prod" d="M 740 360 L 960 360" stroke="#eee" stroke-width="7" fill="none" />
            
            <!-- Nodes: Forget Gate -->
            <circle id="node-forget" cx="230" cy="120" r="66" fill="#f8d7da" stroke="#dc3545" stroke-width="3" class="gate-activation"/>
            <text x="230" y="138" text-anchor="middle" font-size="50" class="math-op" font-family="Arial">×</text>
            <text x="230" y="210" text-anchor="middle" font-size="15" fill="#d32f2f" font-weight="bold">Forget Gate (σ)</text>

            <!-- Nodes: Input Gate Pipeline -->
            <circle cx="340" cy="360" r="60" fill="#d4edda" stroke="#28a745" stroke-width="2" class="gate-activation" id="node-input-sig"/>
            <text x="340" y="378" text-anchor="middle" font-size="44" class="math-op">σ</text>
            
            <circle cx="490" cy="360" r="60" fill="#d4edda" stroke="#28a745" stroke-width="2" class="gate-activation" id="node-input-tanh"/>
            <text x="490" y="378" text-anchor="middle" font-size="36" class="math-op">tanh</text>
            
            <circle cx="645" cy="360" r="60" fill="#d4edda" stroke="#28a745" stroke-width="2" class="gate-activation" id="node-input-prod"/>
            <text x="645" y="378" text-anchor="middle" font-size="44" class="math-op" font-family="Arial">×</text>

            <!-- Node: Memory Update Adder -->
            <circle cx="790" cy="120" r="62" fill="#fff" stroke="#aaa" stroke-width="3" class="gate-activation" id="node-update"/>
            <text x="790" y="138" text-anchor="middle" font-size="46" class="math-op">+</text>
            <text x="790" y="205" text-anchor="middle" font-size="15" fill="#666" font-weight="bold">Cell Update</text>

            <!-- Node: Output Gate Sigmoid -->
            <circle cx="830" cy="360" r="60" fill="#cce5ff" stroke="#007bff" stroke-width="2" class="gate-activation" id="node-output-sig"/>
            <text x="830" y="378" text-anchor="middle" font-size="44" class="math-op">σ</text>
            
            <!-- Node: Tanh Activation for Cell State -->
            <circle cx="920" cy="240" r="60" fill="#fff" stroke="#aaa" stroke-width="2" class="gate-activation" id="node-cell-tanh"/>
            <text x="920" y="258" text-anchor="middle" font-size="36" class="math-op">tanh</text>

            <!-- Final multiplier for h_t -->
            <circle cx="1020" cy="360" r="68" fill="#cce5ff" stroke="#007bff" stroke-width="4" class="gate-activation" id="node-hidden-prod"/>
            <text x="1020" y="380" text-anchor="middle" font-size="54" class="math-op" font-family="Arial">×</text>
            <text x="1020" y="452" text-anchor="middle" font-size="15" fill="#0069d9" font-weight="bold">Output Filter (h<tspan dy="4" font-size="11">t</tspan>)</text>

            <!-- Particles -->
            <circle r="14" fill="#e74c3c" id="p1" style="filter: url(#glow)">
                <animateMotion id="anim1" dur="4s" repeatCount="indefinite" path="M0 120 L1100 120" />
            </circle>
            <circle r="14" fill="#2ecc71" id="p2" style="filter: url(#glow)">
                <animateMotion id="anim2" dur="3s" repeatCount="indefinite" path="M100 480 L100 360 L645 360 L645 250 L790 250 L790 180" />
            </circle>
            <circle r="14" fill="#3498db" id="p3" style="filter: url(#glow)">
                <animateMotion id="anim3" dur="5s" repeatCount="indefinite" path="M920 120 L920 240 L1020 240 L1020 360 L1100 360" />
            </circle>
        </svg>
    `;
    const sliders = ['forget-gate', 'input-gate', 'output-gate'];
    sliders.forEach(id => {
        document.getElementById(id).addEventListener('input', updateCellViz);
    });
    updateCellViz();
}

function updateCellViz() {
    const f = document.getElementById('forget-gate').value;
    const i = document.getElementById('input-gate').value;
    const o = document.getElementById('output-gate').value;

    document.getElementById('val-forget-gate').innerText = f + '%';
    document.getElementById('val-input-gate').innerText = i + '%';
    document.getElementById('val-output-gate').innerText = o + '%';

    document.getElementById('node-forget').style.fillOpacity = 0.2 + (f / 100 * 0.8);
    document.getElementById('node-input-sig').style.fillOpacity = 0.2 + (i / 100 * 0.8);
    document.getElementById('node-output-sig').style.fillOpacity = 0.2 + (o / 100 * 0.8);
    document.getElementById('node-hidden-prod').style.fillOpacity = 0.2 + (o / 100 * 0.8);

    // Adjust particle speed based on gate activity
    const anim1 = document.getElementById('anim1');
    const anim2 = document.getElementById('anim2');
    const anim3 = document.getElementById('anim3');
    if (anim1) anim1.setAttribute('dur', (5 - (f / 100 * 4.5)) + 's');
    if (anim2) anim2.setAttribute('dur', (5 - (i / 100 * 4.5)) + 's');
    if (anim3) anim3.setAttribute('dur', (6 - (o / 100 * 5.0)) + 's');
}

function injectSentiment(type) {
    const log = document.getElementById('pipeline-log');
    const placeholder = log.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    const words = type === 'pos'
        ? ['excellent', 'amazing', 'great', 'fantastic', 'wonderful', 'brilliant', 'perfect', 'outstanding', 'superb', 'exceptional', 'delightful', 'inspiring', 'recommend']
        : ['terrible', 'awful', 'bad', 'horrible', 'dreadful', 'disastrous', 'weak', 'failed', 'poor', 'disappointing', 'garbage', 'broken', 'clunky'];
    const word = words[Math.floor(Math.random() * words.length)];
    const vector = Array.from({ length: 4 }, () => (Math.random() * 2 - 1).toFixed(2));

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<strong>${word}</strong> &rarr; [${vector.join(', ')}]`;
    log.prepend(entry);

    // Trigger Pulse in SVG
    const nodes = document.querySelectorAll('.gate-activation');
    nodes.forEach((n, idx) => {
        setTimeout(() => {
            n.style.strokeWidth = "5px";
            n.style.boxShadow = "0 0 10px gold";
            setTimeout(() => {
                n.style.strokeWidth = "1px";
            }, 500);
        }, idx * 100);
    });
}

function applyPreset(name) {
    const configs = {
        'negation': [20, 90, 80],
        'sarcasm': [50, 40, 90],
        'contrast': [80, 70, 40]
    };
    const vals = configs[name];
    document.getElementById('forget-gate').value = vals[0];
    document.getElementById('input-gate').value = vals[1];
    document.getElementById('output-gate').value = vals[2];
    updateCellViz();
}

function resetSim2() {
    document.getElementById('forget-gate').value = 70;
    document.getElementById('input-gate').value = 50;
    document.getElementById('output-gate').value = 80;
    document.getElementById('pipeline-log').innerHTML = '<div class="log-entry placeholder">Waiting for input...</div>';
    updateCellViz();
}

// ============================================================
// --- SIMULATION 3: SENTIMENT TIMELINE ---
// ============================================================
let sentimentChart = null;
let sim3ShowRNN = true, sim3ShowLSTM = true;

function initSimulation3() {
    const canvas = document.getElementById('sentimentChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    sentimentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'RNN', borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.05)', borderWidth: 2, data: [], fill: true, tension: 0.3 },
                { label: 'LSTM', borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.05)', borderWidth: 2, data: [], fill: true, tension: 0.4 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { min: -1, max: 1, title: { display: true, text: 'Sentiment Score' } }, x: { title: { display: true, text: 'Word Index' } } },
            plugins: { legend: { display: true } }
        }
    });
    const input = document.getElementById('sim3-input');
    if (input) input.addEventListener('input', renderSim3);
    renderSim3();
}

function getSentimentScore(word) {
    const pos = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'beautiful', 'fantastic', 'love', 'best', 'happy', 'brilliant', 'outstanding', 'superb', 'perfect', 'delightful', 'inspiring', 'recommend', 'enjoyed'];
    const neg = ['bad', 'terrible', 'awful', 'horrible', 'boring', 'disappointing', 'worst', 'hate', 'poor', 'ugly', 'dreadful', 'disastrous', 'weak', 'failed', 'garbage', 'broken', 'clunky', 'mediocre'];
    const w = word.toLowerCase().replace(/[^a-z]/g, '');
    if (pos.includes(w)) return 0.6;
    if (neg.includes(w)) return -0.6;
    return 0;
}

function renderSim3() {
    const text = document.getElementById('sim3-input')?.value || '';
    const words = text.split(' ').filter(w => w.length > 0);
    if (words.length === 0) return;

    let rnnScore = 0, lstmScore = 0;
    const rnnScores = [], lstmScores = [], heatData = [];
    let negateNext = false;

    words.forEach((word, i) => {
        let base = getSentimentScore(word);
        if (word.toLowerCase() === 'not' || word.toLowerCase() === "n't") {
            negateNext = true;
            rnnScores.push(rnnScore);
            lstmScores.push(lstmScore);
            heatData.push({ word, score: 0, role: 'negator' });
            return;
        }
        if (negateNext && base !== 0) { base = -base; negateNext = false; }
        else { negateNext = false; }

        // RNN: sharp oscillation
        rnnScore = rnnScore * 0.6 + base * (0.8 + Math.random() * 0.4);
        rnnScore = Math.max(-1, Math.min(1, rnnScore));
        // LSTM: smooth accumulation
        lstmScore = lstmScore * 0.85 + base * 0.4;
        lstmScore = Math.max(-1, Math.min(1, lstmScore));

        rnnScores.push(rnnScore);
        lstmScores.push(lstmScore);
        heatData.push({ word, score: base, rnn: rnnScore, lstm: lstmScore });
    });

    // Update chart
    if (sentimentChart) {
        sentimentChart.data.labels = words.map((_, i) => i);
        sentimentChart.data.datasets[0].data = rnnScores;
        sentimentChart.data.datasets[0].hidden = !sim3ShowRNN;
        sentimentChart.data.datasets[1].data = lstmScores;
        sentimentChart.data.datasets[1].hidden = !sim3ShowLSTM;
        sentimentChart.update();
    }

    // Update gauge
    const finalScore = sim3ShowLSTM ? lstmScore : rnnScore;
    const needle = document.getElementById('gauge-needle');
    const valueEl = document.getElementById('gauge-value');
    if (needle) needle.style.transform = 'translateX(-50%) rotate(' + (finalScore * 90) + 'deg)';
    if (valueEl) valueEl.textContent = finalScore.toFixed(2);

    // Update heatmap
    const heatmap = document.getElementById('sim3-heatmap');
    if (heatmap) {
        heatmap.innerHTML = '';
        heatData.forEach((d, i) => {
            const token = document.createElement('span');
            token.className = 'heat-token';
            token.textContent = d.word;
            const intensity = Math.abs(d.score);
            if (d.score > 0) token.style.background = 'rgba(34,197,94,' + (0.15 + intensity) + ')';
            else if (d.score < 0) token.style.background = 'rgba(239,68,68,' + (0.15 + intensity) + ')';
            else if (d.role === 'negator') { token.style.background = '#fef3c7'; token.style.border = '2px solid #f59e0b'; }
            token.onclick = function () { showWordTooltip(d, i); };
            heatmap.appendChild(token);
        });
    }
}

function showWordTooltip(data, index) {
    const tooltip = document.getElementById('sim3-tooltip');
    if (!tooltip) return;
    var msg = '<strong>' + data.word + '</strong> (Position ' + index + ')<br>';
    if (data.role === 'negator') msg += 'Role: <em>Negator</em> — flips next word\'s polarity';
    else if (data.score > 0) msg += 'Sentiment: <span style="color:#22c55e;">Positive (+' + data.score.toFixed(1) + ')</span>';
    else if (data.score < 0) msg += 'Sentiment: <span style="color:#ef4444;">Negative (' + data.score.toFixed(1) + ')</span>';
    else msg += 'Sentiment: <em>Neutral (0.0)</em>';
    if (data.rnn !== undefined) msg += '<br>RNN running: ' + data.rnn.toFixed(3) + ' | LSTM running: ' + data.lstm.toFixed(3);
    tooltip.innerHTML = msg;
    tooltip.style.display = 'block';
}

function toggleSim3Model(model) {
    var btns = document.querySelectorAll('#sim3 .toggle-btn');
    btns.forEach(function (b) {
        if (b.textContent.trim().toLowerCase() === model) b.classList.toggle('active');
    });
    sim3ShowRNN = document.querySelector('#sim3 .toggle-btn.rnn') && document.querySelector('#sim3 .toggle-btn.rnn').classList.contains('active');
    sim3ShowLSTM = document.querySelector('#sim3 .toggle-btn.lstm') && document.querySelector('#sim3 .toggle-btn.lstm').classList.contains('active');
    renderSim3();
}

function insertNot() {
    var input = document.getElementById('sim3-input');
    if (!input) return;
    var words = input.value.split(' ');
    var sentimentIdx = words.findIndex(function (w) { return Math.abs(getSentimentScore(w)) > 0; });
    if (sentimentIdx > 0) words.splice(sentimentIdx, 0, 'not');
    else words.splice(Math.max(1, Math.floor(words.length / 2)), 0, 'not');
    input.value = words.join(' ');
    renderSim3();
}

async function processSim3Animated() {
    var tokens = document.querySelectorAll('#sim3-heatmap .heat-token');
    for (var i = 0; i < tokens.length; i++) {
        tokens[i].classList.add('processing');
        await new Promise(function (r) { setTimeout(r, 300); });
        tokens[i].classList.remove('processing');
    }
}

function resetSim3() {
    document.getElementById('sim3-input').value = 'I thought the movie would be good but it was disappointing';
    sim3ShowRNN = true; sim3ShowLSTM = true;
    document.querySelectorAll('#sim3 .toggle-btn').forEach(function (b) { b.classList.add('active'); });
    var tooltip = document.getElementById('sim3-tooltip');
    if (tooltip) tooltip.style.display = 'none';
    renderSim3();
}

// ============================================================
// --- SIMULATION 4: TRAINING DYNAMICS LAB ---
// ============================================================
var rnnLossChart = null, rnnAccChart = null, lstmLossChart = null, lstmAccChart = null;
var trainingInterval = null;

function initSimulation4() {
    rnnLossChart = createTrainingChart('rnnLossChart', 'Loss', '#ef4444');
    rnnAccChart = createTrainingChart('rnnAccChart', 'Accuracy (%)', '#ef4444');
    lstmLossChart = createTrainingChart('lstmLossChart', 'Loss', '#3b82f6');
    lstmAccChart = createTrainingChart('lstmAccChart', 'Accuracy (%)', '#3b82f6');
    // Wire up sliders
    var lrSlider = document.getElementById('sim4-lr');
    if (lrSlider) lrSlider.addEventListener('input', function (e) {
        var v = parseInt(e.target.value);
        document.getElementById('val-lr').textContent = v >= 10 ? (v / 10).toFixed(1) + 'e-4' : v + 'e-5';
    });
    var seqSlider = document.getElementById('sim4-seqlen');
    if (seqSlider) seqSlider.addEventListener('input', function (e) {
        document.getElementById('val-seqlen').textContent = e.target.value;
    });
    // Render static confusion matrices
    renderCM('rnn-cm', 600, 583, 167, 150);
    renderCM('lstm-cm', 619, 653, 97, 131);
}

function createTrainingChart(canvasId, label, color) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    return new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: [], datasets: [
                { label: 'Train ' + label, borderColor: color, backgroundColor: 'transparent', data: [], borderWidth: 2, pointRadius: 0 },
                { label: 'Val ' + label, borderColor: color, borderDash: [5, 5], backgroundColor: 'transparent', data: [], borderWidth: 2, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: label } } },
            plugins: { legend: { labels: { font: { size: 10 } } } }
        }
    });
}

function startSim4Training() {
    if (trainingInterval) clearInterval(trainingInterval);
    // Reset charts
    [rnnLossChart, rnnAccChart, lstmLossChart, lstmAccChart].forEach(function (c) {
        if (c) { c.data.labels = []; c.data.datasets.forEach(function (d) { d.data = []; }); c.update(); }
    });
    var epoch = 0;
    var maxEpoch = 150;
    trainingInterval = setInterval(function () {
        epoch++;
        if (epoch > maxEpoch) { clearInterval(trainingInterval); trainingInterval = null; return; }
        var t = epoch / maxEpoch;
        // RNN curves — plateaus around 79% val, noisy
        var rnnTrainAcc = 50 + 40.76 * (1 - Math.exp(-3 * t)) + Math.random() * 2;
        var rnnValAcc = 50 + 29.47 * (1 - Math.exp(-2.5 * t)) + Math.sin(epoch * 0.3) * 2;
        var rnnTrainLoss = 0.7 * Math.exp(-2.5 * t) + 0.05 + Math.random() * 0.03;
        var rnnValLoss = 0.7 * Math.exp(-1.8 * t) + 0.15 + Math.sin(epoch * 0.2) * 0.03;
        // LSTM curves — smooth convergence to 87% val
        var lstmTrainAcc = 50 + 46.17 * (1 - Math.exp(-3.5 * t));
        var lstmValAcc = 50 + 37.33 * (1 - Math.exp(-3 * t));
        var lstmTrainLoss = 0.7 * Math.exp(-3 * t) + 0.02;
        var lstmValLoss = 0.7 * Math.exp(-2.2 * t) + 0.08;

        addChartPoint(rnnAccChart, epoch, Math.min(rnnTrainAcc, 90.76), Math.min(rnnValAcc, 79.47));
        addChartPoint(lstmAccChart, epoch, Math.min(lstmTrainAcc, 96.17), Math.min(lstmValAcc, 87.33));
        addChartPoint(rnnLossChart, epoch, Math.max(rnnTrainLoss, 0.04), Math.max(rnnValLoss, 0.12));
        addChartPoint(lstmLossChart, epoch, Math.max(lstmTrainLoss, 0.01), Math.max(lstmValLoss, 0.06));

        // Gradient bars — RNN shrinks, LSTM stable
        var rnnGrad = document.getElementById('rnn-grad-bar');
        var lstmGrad = document.getElementById('lstm-grad-bar');
        if (rnnGrad) rnnGrad.style.width = Math.max(5, 100 * (1 - t * 0.85)) + '%';
        if (lstmGrad) lstmGrad.style.width = Math.max(60, 100 * (1 - t * 0.15)) + '%';
    }, 40);
}

function addChartPoint(chart, epoch, trainVal, valVal) {
    if (!chart) return;
    chart.data.labels.push(epoch);
    chart.data.datasets[0].data.push(trainVal);
    chart.data.datasets[1].data.push(valVal);
    chart.update();
}

function renderCM(containerId, tp, tn, fp, fn) {
    var el = document.getElementById(containerId);
    if (!el) return;
    var total = tp + tn + fp + fn;
    el.innerHTML = '<div class="cm-cell cm-tp" title="True Positive"><span class="cm-label">TP</span><span class="cm-val">' + tp + '</span></div>' +
        '<div class="cm-cell cm-fp" title="False Positive"><span class="cm-label">FP</span><span class="cm-val">' + fp + '</span></div>' +
        '<div class="cm-cell cm-fn" title="False Negative"><span class="cm-label">FN</span><span class="cm-val">' + fn + '</span></div>' +
        '<div class="cm-cell cm-tn" title="True Negative"><span class="cm-label">TN</span><span class="cm-val">' + tn + '</span></div>' +
        '<div class="cm-accuracy">Accuracy: ' + ((tp + tn) / total * 100).toFixed(1) + '%</div>';
}

function resetSim4() {
    if (trainingInterval) { clearInterval(trainingInterval); trainingInterval = null; }
    [rnnLossChart, rnnAccChart, lstmLossChart, lstmAccChart].forEach(function (c) {
        if (c) { c.data.labels = []; c.data.datasets.forEach(function (d) { d.data = []; }); c.update(); }
    });
    var rnnGrad = document.getElementById('rnn-grad-bar');
    var lstmGrad = document.getElementById('lstm-grad-bar');
    if (rnnGrad) rnnGrad.style.width = '100%';
    if (lstmGrad) lstmGrad.style.width = '100%';
}

// ============================================================
// --- SIMULATION 5: FEATURE ATTRIBUTION EXPLORER ---
// ============================================================
var attModel = 'both';
var cellStateChartA = null, cellStateChartB = null;

function initSimulation5() {
    var inputA = document.getElementById('sim5-input-a');
    var inputB = document.getElementById('sim5-input-b');
    var threshold = document.getElementById('sim5-threshold');
    if (inputA) inputA.addEventListener('input', renderSim5A);
    if (inputB) inputB.addEventListener('input', renderSim5B);
    if (threshold) threshold.addEventListener('input', function (e) {
        document.getElementById('val-threshold').textContent = (e.target.value / 100).toFixed(2);
        renderSim5A(); renderSim5B();
    });
    renderSim5A();
    renderSim5B();
}

function computeSaliency(words, model) {
    return words.map(function (word, i) {
        var base = Math.abs(getSentimentScore(word));
        var positionFactor;
        if (model === 'rnn') {
            positionFactor = Math.pow(0.7, words.length - 1 - i);
        } else {
            positionFactor = 0.5 + 0.5 * Math.sin(Math.PI * i / words.length);
        }
        var noise = 0.05 + Math.random() * 0.1;
        return Math.min(1, (base > 0 ? base + 0.3 : 0.08 + noise) * positionFactor + noise * 0.3);
    });
}

function renderAttribution(containerId, words, model, chartCanvasId, existingChart) {
    var container = document.getElementById(containerId);
    if (!container) return existingChart;
    var threshold = (document.getElementById('sim5-threshold') ? document.getElementById('sim5-threshold').value : 10) / 100;
    container.innerHTML = '';

    var rnnScores = computeSaliency(words, 'rnn');
    var lstmScores = computeSaliency(words, 'lstm');
    var scores = model === 'rnn' ? rnnScores : lstmScores;
    var maxScore = Math.max.apply(null, scores.concat([0.01]));

    words.forEach(function (word, i) {
        var div = document.createElement('div');
        div.className = 'att-word';
        var normalized = scores[i] / maxScore;
        if (normalized < threshold) { div.style.opacity = '0.2'; }
        var bar = document.createElement('div');
        bar.className = 'att-bar';
        bar.style.height = (normalized * 250) + 'px';
        if (model === 'rnn') {
            bar.style.background = 'rgba(239,68,68,' + (0.4 + normalized * 0.6) + ')';
        } else {
            bar.style.background = 'rgba(59,130,246,' + (0.4 + normalized * 0.6) + ')';
        }
        bar.setAttribute('data-score', scores[i].toFixed(2));
        var span = document.createElement('span');
        span.textContent = word;
        div.appendChild(bar);
        div.appendChild(span);
        container.appendChild(div);
    });

    // Cell state chart (LSTM only)
    var section = chartCanvasId === 'cellStateChartA' ? document.getElementById('cell-state-section-a') : document.getElementById('cell-state-section-b');
    if (section) section.style.display = (model === 'lstm' || attModel === 'both') ? 'block' : 'none';

    var canvas = document.getElementById(chartCanvasId);
    if (canvas) {
        if (existingChart) existingChart.destroy();
        var cellStates = words.map(function (_, i) {
            var cs = 0;
            for (var j = 0; j <= i; j++) cs = cs * 0.9 + lstmScores[j] * 0.3;
            return cs;
        });
        existingChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { labels: words, datasets: [{ label: 'Cell State |C_t|', borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.1)', data: cellStates, fill: true, tension: 0.4, borderWidth: 2 }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { title: { display: true, text: 'Magnitude' } } }, plugins: { legend: { display: false } } }
        });
    }
    return existingChart;
}

function renderSim5A() {
    var text = document.getElementById('sim5-input-a') ? document.getElementById('sim5-input-a').value : '';
    var words = text.split(' ').filter(function (w) { return w.length > 0; });
    var model = attModel === 'both' ? 'lstm' : attModel;
    cellStateChartA = renderAttribution('sim5-attribution-a', words, model, 'cellStateChartA', cellStateChartA);
}

function renderSim5B() {
    var text = document.getElementById('sim5-input-b') ? document.getElementById('sim5-input-b').value : '';
    var words = text.split(' ').filter(function (w) { return w.length > 0; });
    var model = attModel === 'both' ? 'lstm' : attModel;
    cellStateChartB = renderAttribution('sim5-attribution-b', words, model, 'cellStateChartB', cellStateChartB);
}

function toggleAttModel(model) {
    var btns = document.querySelectorAll('#sim5 .toggle-btn');
    btns.forEach(function (b) {
        if (b.textContent.trim().toLowerCase() === model) b.classList.toggle('active');
    });
    var rActive = document.querySelector('#sim5 .toggle-btn.rnn') && document.querySelector('#sim5 .toggle-btn.rnn').classList.contains('active');
    var lActive = document.querySelector('#sim5 .toggle-btn.lstm') && document.querySelector('#sim5 .toggle-btn.lstm').classList.contains('active');
    if (rActive && lActive) attModel = 'both';
    else if (rActive) attModel = 'rnn';
    else if (lActive) attModel = 'lstm';
    else attModel = 'both';
    renderSim5A(); renderSim5B();
}

function resetSim5() {
    document.getElementById('sim5-input-a').value = 'The visuals were beautiful but the story was boring';
    document.getElementById('sim5-input-b').value = 'The story was boring but the visuals were beautiful';
    document.getElementById('sim5-threshold').value = 10;
    document.getElementById('val-threshold').textContent = '0.10';
    attModel = 'both';
    document.querySelectorAll('#sim5 .toggle-btn').forEach(function (b) { b.classList.add('active'); });
    renderSim5A(); renderSim5B();
}

// ============================================================
// --- SIMULATION 6: MODEL COMPARISON DASHBOARD ---
// ============================================================
var compareBarChart = null, rocChart = null, prChart = null;

function initSimulation6() {
    // Charts will render when tab is first shown
}

function renderSim6() {
    renderCompareBar();
    renderROC();
    renderPR();
    animateF1Bars();
    animateCounters();
}

function renderCompareBar() {
    var canvas = document.getElementById('compareBarChart');
    if (!canvas) return;
    if (compareBarChart) compareBarChart.destroy();
    compareBarChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Train', 'Validation', 'Test'],
            datasets: [
                { label: 'RNN', data: [90.76, 79.47, 78.87], backgroundColor: 'rgba(239,68,68,0.8)', borderRadius: 6 },
                { label: 'LSTM', data: [96.17, 87.33, 84.80], backgroundColor: 'rgba(59,130,246,0.8)', borderRadius: 6 }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 60, max: 100, title: { display: true, text: 'Accuracy (%)' } } }, plugins: { legend: { display: true } } }
    });
}

function renderROC() {
    var canvas = document.getElementById('rocChart');
    if (!canvas) return;
    if (rocChart) rocChart.destroy();
    var rnnFPR = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
    var rnnTPR = [0, 0.15, 0.35, 0.52, 0.62, 0.70, 0.78, 0.83, 0.87, 0.91, 0.94, 0.97, 0.99, 1];
    var lstmFPR = [0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.6, 0.75, 0.9, 1];
    var lstmTPR = [0, 0.25, 0.48, 0.65, 0.76, 0.83, 0.88, 0.92, 0.95, 0.97, 0.98, 0.99, 1, 1];
    rocChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: rnnFPR,
            datasets: [
                { label: 'RNN (AUC ~ 0.86)', data: rnnFPR.map(function (_, i) { return { x: rnnFPR[i], y: rnnTPR[i] }; }), borderColor: '#ef4444', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 },
                { label: 'LSTM (AUC ~ 0.92)', data: lstmFPR.map(function (_, i) { return { x: lstmFPR[i], y: lstmTPR[i] }; }), borderColor: '#3b82f6', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 },
                { label: 'Random', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: '#ccc', borderDash: [5, 5], borderWidth: 1, fill: false, pointRadius: 0 }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 }, y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 } } }
    });
}

function renderPR() {
    var canvas = document.getElementById('prChart');
    if (!canvas) return;
    if (prChart) prChart.destroy();
    var recall = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
    var rnnPrec = [0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.74, 0.70, 0.65, 0.58, 0.50];
    var lstmPrec = [0.98, 0.96, 0.93, 0.91, 0.88, 0.85, 0.82, 0.78, 0.73, 0.67, 0.55];
    prChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            datasets: [
                { label: 'RNN', data: recall.map(function (r, i) { return { x: r, y: rnnPrec[i] }; }), borderColor: '#ef4444', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 },
                { label: 'LSTM', data: recall.map(function (r, i) { return { x: r, y: lstmPrec[i] }; }), borderColor: '#3b82f6', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', title: { display: true, text: 'Recall' }, min: 0, max: 1 }, y: { title: { display: true, text: 'Precision' }, min: 0.4, max: 1 } } }
    });
}

function animateF1Bars() {
    setTimeout(function () {
        var rnnBar = document.getElementById('f1-rnn-bar');
        var lstmBar = document.getElementById('f1-lstm-bar');
        if (rnnBar) rnnBar.style.width = '78.5%';
        if (lstmBar) lstmBar.style.width = '84.2%';
    }, 300);
}

function animateCounters() {
    document.querySelectorAll('#sim6 .counter').forEach(function (counter) {
        var target = parseFloat(counter.dataset.target);
        var current = 0;
        var step = target / 60;
        var timer = setInterval(function () {
            current += step;
            if (current >= target) { current = target; clearInterval(timer); }
            counter.textContent = current.toFixed(2);
        }, 25);
    });
}
