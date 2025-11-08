// Global variables
let lossChart = null;
let accuracyChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    checkStatus();
    loadModelInfo();
    loadTrainingHistory();

    // Update input stats on typing
    document.getElementById('inputText').addEventListener('input', updateInputStats);
});

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');

    // Activate button
    event.target.classList.add('active');
}

// Check model status
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        const badge = document.getElementById('statusBadge');
        const statusText = document.getElementById('statusText');

        if (data.model_loaded) {
            badge.classList.add('ready');
            statusText.textContent = 'Model Ready';
        } else {
            badge.classList.add('error');
            statusText.textContent = 'Model Not Loaded';
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

// Load example text
function loadExample(exampleId) {
    const text = exampleTexts[exampleId];
    document.getElementById('inputText').value = text;
    updateInputStats();
    showAlert('Example loaded successfully!', 'success');
}

// Clear input
function clearInput() {
    document.getElementById('inputText').value = '';
    updateInputStats();
}

// Update input word count
function updateInputStats() {
    const text = document.getElementById('inputText').value.trim();
    const wordCount = text ? text.split(/\s+/).length : 0;
    document.getElementById('inputStats').textContent = `Words: ${wordCount}`;
}

// Generate summary
async function generateSummary() {
    const inputText = document.getElementById('inputText').value.trim();

    if (!inputText) {
        showAlert('Please enter some text to summarize.', 'error');
        return;
    }

    const btn = document.getElementById('generateBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');

    // Show loading state
    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';

    try {
        const response = await fetch('/api/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: inputText })
        });

        const data = await response.json();

        if (data.success) {
            // Display summary
            const outputDiv = document.getElementById('outputText');
            outputDiv.innerHTML = `<p>${data.summary}</p>`;

            // Update stats
            document.getElementById('outputStats').textContent =
                `Words: ${data.summary_length}`;

            showAlert('Summary generated successfully!', 'success');
        } else {
            showAlert('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error generating summary:', error);
        showAlert('Failed to generate summary. Please try again.', 'error');
    } finally {
        // Reset button
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Copy summary to clipboard
function copySummary() {
    const outputText = document.getElementById('outputText').textContent.trim();

    if (!outputText || outputText === 'Your summary will appear here...') {
        showAlert('No summary to copy!', 'error');
        return;
    }

    navigator.clipboard.writeText(outputText).then(() => {
        showAlert('Summary copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Error copying text:', err);
        showAlert('Failed to copy summary.', 'error');
    });
}

// Show alert message
function showAlert(message, type) {
    const alertArea = document.getElementById('alertArea');
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    alertArea.innerHTML = '';
    alertArea.appendChild(alertDiv);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        alertDiv.style.opacity = '0';
        setTimeout(() => alertDiv.remove(), 300);
    }, 5000);
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();

        if (data.success) {
            const infoHtml = `
                <div class="info-item">
                    <strong>Architecture</strong>
                    <span>${data.architecture}</span>
                </div>
                <div class="info-item">
                    <strong>Embedding Dimension</strong>
                    <span>${data.embedding_dim}</span>
                </div>
                <div class="info-item">
                    <strong>Latent Dimension</strong>
                    <span>${data.latent_dim}</span>
                </div>
                <div class="info-item">
                    <strong>Max Text Length</strong>
                    <span>${data.max_text_len} tokens</span>
                </div>
                <div class="info-item">
                    <strong>Max Summary Length</strong>
                    <span>${data.max_summary_len} tokens</span>
                </div>
                <div class="info-item">
                    <strong>Text Vocabulary</strong>
                    <span>${data.vocab_size_text.toLocaleString()}</span>
                </div>
                <div class="info-item">
                    <strong>Summary Vocabulary</strong>
                    <span>${data.vocab_size_summary.toLocaleString()}</span>
                </div>
                <div class="info-item">
                    <strong>Total Parameters</strong>
                    <span>${data.total_params.toLocaleString()}</span>
                </div>
            `;

            document.getElementById('modelInfo').innerHTML = infoHtml;
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('modelInfo').innerHTML =
            '<p>Model information not available. Please ensure the model is loaded.</p>';
    }
}

// Load training history and create charts
async function loadTrainingHistory() {
    try {
        const response = await fetch('/api/training-history');
        const data = await response.json();

        if (data.success) {
            createLossChart(data);
            createAccuracyChart(data);
            displayMetricsSummary(data.metrics);
        } else {
            showNoHistoryMessage();
        }
    } catch (error) {
        console.error('Error loading training history:', error);
        showNoHistoryMessage();
    }
}

// Create loss chart
function createLossChart(data) {
    const ctx = document.getElementById('lossChart').getContext('2d');

    if (lossChart) {
        lossChart.destroy();
    }

    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [
                {
                    label: 'Training Loss',
                    data: data.train_loss,
                    borderColor: 'rgb(52, 152, 219)',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.3
                },
                {
                    label: 'Validation Loss',
                    data: data.val_loss,
                    borderColor: 'rgb(231, 76, 60)',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Loss',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Create accuracy chart
function createAccuracyChart(data) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');

    if (accuracyChart) {
        accuracyChart.destroy();
    }

    // Convert to percentages
    const trainAccPercent = data.train_acc.map(val => val * 100);
    const valAccPercent = data.val_acc.map(val => val * 100);

    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: trainAccPercent,
                    borderColor: 'rgb(52, 152, 219)',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.3
                },
                {
                    label: 'Validation Accuracy',
                    data: valAccPercent,
                    borderColor: 'rgb(231, 76, 60)',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Display metrics summary
function displayMetricsSummary(metrics) {
    const summaryHtml = `
        <h3 style="margin-bottom: 15px;">📈 Training Performance Summary</h3>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Final Training Loss</div>
                <div class="value">${metrics.final_train_loss}</div>
            </div>
            <div class="metric-card">
                <div class="label">Final Validation Loss</div>
                <div class="value">${metrics.final_val_loss}</div>
            </div>
            <div class="metric-card">
                <div class="label">Training Accuracy</div>
                <div class="value">${metrics.final_train_acc}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Validation Accuracy</div>
                <div class="value">${metrics.final_val_acc}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Best Epoch</div>
                <div class="value">${metrics.best_epoch}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Epochs</div>
                <div class="value">${metrics.total_epochs}</div>
            </div>
        </div>
    `;

    document.getElementById('metricsSummary').innerHTML = summaryHtml;
}

// Show message when no training history
function showNoHistoryMessage() {
    document.getElementById('metricsSummary').innerHTML = `
        <p style="text-align: center; color: white;">
            No training history available. Train the model first using:<br>
            <code style="background: rgba(0,0,0,0.2); padding: 5px 10px; border-radius: 5px; display: inline-block; margin-top: 10px;">
                python quick_demo_train.py
            </code>
        </p>
    `;
}
