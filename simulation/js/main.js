/**
 * LSTM Sentiment Analysis Simulation - Main JavaScript
 * Handles cell execution, step state management, and visualization dropdown
 */

// ============================================
// Configuration
// ============================================

const CONFIG = {
    executionDelay: 1500, // 1.5 second delay for realistic execution simulation
    totalSteps: 8
};

// ============================================
// Code Templates for Different Visualization Types
// ============================================

const VISUALIZATION_CODE = {
    combined: `<pre><code class="language-python"><span class="comment"># Graph Visualization - Combined (RNN vs LSTM)</span>

<span class="builtin">print</span>(<span class="string">"\\nCombined Training Curves (RNN vs LSTM)-"</span>)
plot_side_by_side_epoch_metrics(
    history_rnn, history_lstm,
    <span class="string">"accuracy"</span>, <span class="string">"loss"</span>,
    <span class="string">"Accuracy"</span>, <span class="string">"Loss"</span>,
    <span class="string">"Training Accuracy (RNN vs LSTM)"</span>,
    <span class="string">"Training Loss (RNN vs LSTM)"</span>
)

<span class="builtin">print</span>(<span class="string">"\\nCombined Validation Curves (RNN vs LSTM)-"</span>)
plot_side_by_side_epoch_metrics(
    history_rnn, history_lstm,
    <span class="string">"val_accuracy"</span>, <span class="string">"val_loss"</span>,
    <span class="string">"Accuracy"</span>, <span class="string">"Loss"</span>,
    <span class="string">"Validation Accuracy (RNN vs LSTM)"</span>,
    <span class="string">"Validation Loss (RNN vs LSTM)"</span>
)

<span class="builtin">print</span>(<span class="string">"\\nCombined ROC & PR Curves"</span>)
plot_roc_pr_curves(best_rnn, best_lstm, X_test_pad, y_test, X_val_pad, y_val)</code></pre>`,

    lstm: `<pre><code class="language-python"><span class="comment"># Graph Visualization - LSTM Only</span>

<span class="builtin">print</span>(<span class="string">"\\nLearning Curves for LSTM-"</span>)
plot_per_model_metrics(history_lstm, <span class="string">"LSTM"</span>, gap=<span class="number">0.2</span>)

<span class="builtin">print</span>(<span class="string">"\\nConfusion Matrix - LSTM"</span>)
plt.figure(figsize=(<span class="number">5</span>,<span class="number">4</span>))
sns.heatmap(confusion_matrix(y_test, y_pred_lstm), annot=<span class="keyword">True</span>, fmt=<span class="string">'d'</span>, cmap=<span class="string">'Blues'</span>)
plt.title(<span class="string">"Confusion Matrix - LSTM"</span>)
plt.xlabel(<span class="string">"Predicted"</span>)
plt.ylabel(<span class="string">"Actual"</span>)
plt.show()

<span class="builtin">print</span>(<span class="string">"\\nROC & PR Curves for LSTM-"</span>)
plot_model_roc_pr(best_lstm, <span class="string">"LSTM"</span>, X_test_pad, y_test, X_val_pad, y_val)</code></pre>`,

    rnn: `<pre><code class="language-python"><span class="comment"># Graph Visualization - RNN Only</span>

<span class="builtin">print</span>(<span class="string">"\\nLearning Curves for RNN-"</span>)
plot_per_model_metrics(history_rnn, <span class="string">"RNN"</span>, gap=<span class="number">0.2</span>)

<span class="builtin">print</span>(<span class="string">"\\nConfusion Matrix - RNN"</span>)
plt.figure(figsize=(<span class="number">5</span>,<span class="number">4</span>))
sns.heatmap(confusion_matrix(y_test, y_pred_rnn), annot=<span class="keyword">True</span>, fmt=<span class="string">'d'</span>, cmap=<span class="string">'Blues'</span>)
plt.title(<span class="string">"Confusion Matrix - RNN"</span>)
plt.xlabel(<span class="string">"Predicted"</span>)
plt.ylabel(<span class="string">"Actual"</span>)
plt.show()

<span class="builtin">print</span>(<span class="string">"\\nROC & PR Curves for RNN-"</span>)
plot_model_roc_pr(best_rnn, <span class="string">"RNN"</span>, X_test_pad, y_test, X_val_pad, y_val)</code></pre>`
};

// ============================================
// State Management
// ============================================

const state = {
    currentStep: 1,
    completedSteps: new Set(),
    runningStep: null,
    isRunningAll: false,
    visualizationType: 'combined' // 'lstm', 'rnn', or 'combined'
};

// ============================================
// DOM Elements
// ============================================

const elements = {
    stepItems: document.querySelectorAll('.step-item'),
    cells: document.querySelectorAll('.notebook-cell'),
    resetBtn: document.getElementById('resetBtn'),
    completionMessage: document.getElementById('completionMessage'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    visualizationSelect: document.getElementById('visualizationSelect'),
    downloadBtn: document.getElementById('downloadBtn'),
    visualizationCodeContainer: document.getElementById('visualizationCodeContainer')
};

// ============================================
// Helper Functions
// ============================================

function updateStepState(stepNumber, status) {
    const stepItem = document.querySelector(`.step-item[data-step="${stepNumber}"]`);
    const cell = document.querySelector(`.notebook-cell[data-step="${stepNumber}"]`);
    
    if (!stepItem) return;
    
    stepItem.classList.remove('active', 'running', 'completed');
    if (cell) {
        cell.classList.remove('running', 'completed');
    }
    
    switch (status) {
        case 'active':
            stepItem.classList.add('active');
            break;
        case 'running':
            stepItem.classList.add('running');
            if (cell) cell.classList.add('running');
            state.runningStep = stepNumber;
            break;
        case 'completed':
            stepItem.classList.add('completed');
            if (cell) cell.classList.add('completed');
            state.completedSteps.add(stepNumber);
            break;
    }
}

function setNextStepActive() {
    const nextStep = Math.min(state.currentStep + 1, CONFIG.totalSteps);
    if (!state.completedSteps.has(nextStep)) {
        updateStepState(nextStep, 'active');
        state.currentStep = nextStep;
    }
}

function canRunCell(cellNumber) {
    // Cell 8 (visualization) can be re-run if cells 1-7 are completed
    if (cellNumber === 8) {
        for (let i = 1; i < 8; i++) {
            if (!state.completedSteps.has(i)) {
                return false;
            }
        }
        return true;
    }
    
    // For other cells, all previous cells must be completed
    for (let i = 1; i < cellNumber; i++) {
        if (!state.completedSteps.has(i)) {
            return false;
        }
    }
    return true;
}

function updateRunButtonStates() {
    elements.cells.forEach((cell, index) => {
        const cellNumber = index + 1;
        const runBtn = cell.querySelector('.run-btn');
        
        if (!runBtn) return;
        
        // Special handling for cell 8 - allow rerunning
        if (cellNumber === 8) {
            if (canRunCell(8)) {
                runBtn.disabled = false;
                runBtn.title = '';
                // Reset button to "Run" state to allow re-running
                runBtn.innerHTML = `
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                    Run
                `;
            } else {
                runBtn.disabled = true;
                runBtn.title = 'Run previous cells first';
            }
            return;
        }
        
        if (state.completedSteps.has(cellNumber)) {
            return;
        }
        
        if (canRunCell(cellNumber)) {
            runBtn.disabled = false;
            runBtn.title = '';
        } else {
            runBtn.disabled = true;
            runBtn.title = 'Run previous cells first';
        }
    });
}

// ============================================
// Visualization Handling
// ============================================

function updateVisualizationCode() {
    const type = state.visualizationType;
    const codeContainer = elements.visualizationCodeContainer;
    
    if (!codeContainer) return;
    
    codeContainer.innerHTML = VISUALIZATION_CODE[type] || VISUALIZATION_CODE.combined;
}

function updateVisualization() {
    const type = state.visualizationType;
    const container = document.getElementById('visualizationContainer');
    
    if (!container) return;
    
    let html = '';
    
    if (type === 'lstm') {
        html = `
            <div class="visualization-grid">
                <div class="viz-item">
                    <h4>Learning Curves - LSTM</h4>
                    <img src="${EXPERIMENT_DATA.images.lstm.learning_curves}" alt="LSTM Learning Curves">
                </div>
                <div class="viz-item">
                    <h4>Confusion Matrix - LSTM</h4>
                    <img src="${EXPERIMENT_DATA.images.lstm.confusion_matrix}" alt="LSTM Confusion Matrix">
                </div>
                <div class="viz-item full-width">
                    <h4>ROC & Precision-Recall Curves - LSTM</h4>
                    <img src="${EXPERIMENT_DATA.images.lstm.roc_pr}" alt="LSTM ROC PR Curves">
                </div>
            </div>
        `;
    } else if (type === 'rnn') {
        html = `
            <div class="visualization-grid">
                <div class="viz-item">
                    <h4>Learning Curves - RNN</h4>
                    <img src="${EXPERIMENT_DATA.images.rnn.learning_curves}" alt="RNN Learning Curves">
                </div>
                <div class="viz-item">
                    <h4>Confusion Matrix - RNN</h4>
                    <img src="${EXPERIMENT_DATA.images.rnn.confusion_matrix}" alt="RNN Confusion Matrix">
                </div>
                <div class="viz-item full-width">
                    <h4>ROC & Precision-Recall Curves - RNN</h4>
                    <img src="${EXPERIMENT_DATA.images.rnn.roc_pr}" alt="RNN ROC PR Curves">
                </div>
            </div>
        `;
    } else { // combined
        html = `
            <div class="visualization-grid">
                <div class="viz-item">
                    <h4>Combined Training Curves (RNN vs LSTM)</h4>
                    <img src="${EXPERIMENT_DATA.images.combined.training_curves}" alt="Combined Training Curves">
                </div>
                <div class="viz-item">
                    <h4>Combined Validation Curves (RNN vs LSTM)</h4>
                    <img src="${EXPERIMENT_DATA.images.combined.validation_curves}" alt="Combined Validation Curves">
                </div>
                <div class="viz-item full-width">
                    <h4>Combined ROC & Precision-Recall Curves</h4>
                    <img src="${EXPERIMENT_DATA.images.combined.roc_pr}" alt="Combined ROC PR Curves">
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function handleVisualizationChange(newValue) {
    state.visualizationType = newValue;
    
    // Always update the code display when selection changes
    updateVisualizationCode();
    
    // If cell 8 has been run at least once (cells 1-7 completed), 
    // automatically update the visualization output
    if (canRunCell(8)) {
        const cell = document.querySelector('.notebook-cell[data-cell="8"]');
        const output = cell.querySelector('.cell-output');
        
        // If output is already visible, update visualization immediately
        if (output && !output.classList.contains('hidden')) {
            updateVisualization();
        }
        
        // Update button states to allow rerunning
        updateRunButtonStates();
    }
}

// ============================================
// Cell Execution
// ============================================

async function executeCell(cellNumber) {
    const cell = document.querySelector(`.notebook-cell[data-cell="${cellNumber}"]`);
    const stepNumber = parseInt(cell.dataset.step);
    const output = cell.querySelector('.cell-output');
    const runBtn = cell.querySelector('.run-btn');
    
    // For cells 1-7, skip if already completed
    if (cellNumber !== 8 && state.completedSteps.has(stepNumber)) {
        return;
    }
    
    if (!canRunCell(cellNumber)) {
        alert(`Please run all previous cells first before running Step ${cellNumber}.`);
        return;
    }
    
    // Update UI to show running state
    updateStepState(stepNumber, 'running');
    runBtn.classList.add('running');
    runBtn.disabled = true;
    runBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" class="spinning">
            <path d="M12 4V2A10 10 0 0 0 2 12h2a8 8 0 0 1 8-8z"/>
        </svg>
        Running...
    `;
    
    const spinner = runBtn.querySelector('.spinning');
    if (spinner) {
        spinner.style.animation = 'spin 1s linear infinite';
    }
    
    // Simulate execution delay
    await new Promise(resolve => setTimeout(resolve, CONFIG.executionDelay));
    
    // Special handling for visualization step
    if (cellNumber === 8) {
        updateVisualization();
    }
    
    // Show output
    output.classList.remove('hidden');
    
    // Update UI to show completed state
    updateStepState(stepNumber, 'completed');
    runBtn.classList.remove('running');
    
    // For cell 8, keep Run button enabled for re-running
    if (cellNumber === 8) {
        runBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
            </svg>
            Run
        `;
        runBtn.disabled = false;
    } else {
        runBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
            </svg>
            Done
        `;
        runBtn.disabled = true;
    }
    
    // Update run button states for other cells
    updateRunButtonStates();
    
    if (state.completedSteps.size === CONFIG.totalSteps) {
        showCompletionMessage();
    } else if (cellNumber !== 8) {
        setNextStepActive();
    }
}

// ============================================
// Reset Functionality
// ============================================

function resetSimulation() {
    state.currentStep = 1;
    state.completedSteps.clear();
    state.runningStep = null;
    state.isRunningAll = false;
    state.visualizationType = 'combined';
    
    // Reset step items
    elements.stepItems.forEach((item, index) => {
        item.classList.remove('active', 'running', 'completed');
        if (index === 0) {
            item.classList.add('active');
        }
    });
    
    // Reset cells
    elements.cells.forEach(cell => {
        cell.classList.remove('running', 'completed');
        const output = cell.querySelector('.cell-output');
        const runBtn = cell.querySelector('.run-btn');
        
        if (output) output.classList.add('hidden');
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.classList.remove('running');
            runBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z"/>
                </svg>
                Run
            `;
        }
    });
    
    // Reset visualization dropdown
    if (elements.visualizationSelect) {
        elements.visualizationSelect.value = 'combined';
    }
    
    // Update visualization code
    updateVisualizationCode();
    
    // Hide completion message
    elements.completionMessage.classList.add('hidden');
    
    // Update button states
    updateRunButtonStates();
}

// ============================================
// Completion Message
// ============================================

function showCompletionMessage() {
    elements.completionMessage.classList.remove('hidden');
    elements.completionMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ============================================
// Download Functionality
// ============================================

function downloadExperiment() {
    // Download the experiment PDF
    const link = document.createElement('a');
    link.href = './assets/LSTM for Sentiment Analysis.pdf';
    link.download = 'LSTM_Sentiment_Analysis_Experiment.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ============================================
// Event Listeners
// ============================================

function initEventListeners() {
    // Run buttons on individual cells
    elements.cells.forEach((cell, index) => {
        const runBtn = cell.querySelector('.run-btn');
        if (runBtn) {
            runBtn.addEventListener('click', () => executeCell(index + 1));
        }
    });
    
    // Download button
    if (elements.downloadBtn) {
        elements.downloadBtn.addEventListener('click', downloadExperiment);
    }
    
    // Reset button
    if (elements.resetBtn) {
        elements.resetBtn.addEventListener('click', resetSimulation);
    }
    
    // Visualization dropdown
    if (elements.visualizationSelect) {
        elements.visualizationSelect.addEventListener('change', (e) => {
            handleVisualizationChange(e.target.value);
        });
    }
    
    // Step item clicks (scroll to cell)
    elements.stepItems.forEach(item => {
        item.addEventListener('click', () => {
            const stepNumber = parseInt(item.dataset.step);
            const targetCell = document.querySelector(`.notebook-cell[data-step="${stepNumber}"]`);
            if (targetCell) {
                targetCell.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    });
}

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    
    // Set first step as active
    updateStepState(1, 'active');
    
    // Initialize visualization code with default (combined)
    updateVisualizationCode();
    
    // Update button states
    updateRunButtonStates();
    
    console.log('LSTM Sentiment Analysis Simulation initialized');
});
