let globalEquityCurve = null;
let chartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    fetch('data.json?t=' + new Date().getTime())
        .then(response => response.json())
        .then(data => {
            renderMetrics(data.metrics);
            globalEquityCurve = data.equity_curve;
            updatePeriod('all'); // Default view
        })
        .catch(error => console.error('Error loading data:', error));
});

function updatePeriod(mode) {
    // Update active button state
    document.querySelectorAll('.btn').forEach(btn => {
        btn.classList.remove('active');
        const text = btn.textContent.toLowerCase();
        if (mode === 'all' && text === 'all') btn.classList.add('active');
        if (mode === 'test' && text.includes('testing')) btn.classList.add('active');
        if (mode === 'valid_test' && text.includes('valid')) btn.classList.add('active');
    });

    if (!globalEquityCurve) return;

    let indices = [];
    const segments = globalEquityCurve.segments;

    // Determine indices explicitly based on segments
    if (mode === 'all') {
        indices = segments.map((_, i) => i);
    } else if (mode === 'test') {
        indices = segments.map((s, i) => s === 'Test' ? i : -1).filter(i => i !== -1);
    } else if (mode === 'valid_test') {
        indices = segments.map((s, i) => (s === 'Valid' || s === 'Test') ? i : -1).filter(i => i !== -1);
    }

    if (indices.length === 0) return;

    // Filter and Re-Base Data
    // For Arithmetic Cumulative Returns (Sum of Returns), we subtract the base value.
    const baseStrategy = globalEquityCurve.strategy[indices[0]];
    const baseBenchmark = globalEquityCurve.benchmark[indices[0]];

    // Guard against undefined/null
    const safeSub = (val, base) => (val !== undefined && val !== null) ? val - base : 0;

    const filteredCurve = {
        dates: indices.map(i => globalEquityCurve.dates[i]),
        strategy: indices.map(i => safeSub(globalEquityCurve.strategy[i], baseStrategy)),
        benchmark: indices.map(i => safeSub(globalEquityCurve.benchmark[i], baseBenchmark)),
        segments: indices.map(i => globalEquityCurve.segments[i])
    };

    renderEquityChart(filteredCurve);
}

function renderMetrics(metrics) {
    const formatPercent = (val) => (val * 100).toFixed(2) + '%';
    const formatNumber = (val) => val.toFixed(4);

    if (metrics.annualized_return !== undefined) {
        setMetric('metric-return', metrics.annualized_return, formatPercent);
    }
    if (metrics.sharpe_ratio !== undefined) {
        setMetric('metric-sharpe', metrics.sharpe_ratio, formatNumber);
    }
    if (metrics.max_drawdown !== undefined) {
        setMetric('metric-mdd', metrics.max_drawdown, formatPercent);
    }
    if (metrics.information_ratio !== undefined) {
        setMetric('metric-ir', metrics.information_ratio, formatNumber);
    }
}

function setMetric(id, value, formatter) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = formatter(value);

    // Color coding for returns
    if (id === 'metric-return' || id === 'metric-sharpe') {
        el.style.color = value >= 0 ? 'var(--accent)' : 'var(--negative)';
    }
}

function renderEquityChart(equityCurve) {
    const ctx = document.getElementById('equityChart').getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    // Plugin to draw detailed background segments
    const segmentPlugin = {
        id: 'segmentBackground',
        beforeDraw: (chart) => {
            const ctx = chart.ctx;
            const xAxis = chart.scales.x;
            const yAxis = chart.scales.y;
            const segments = equityCurve.segments;

            if (!segments) return;

            const colors = {
                'Train': 'rgba(255, 255, 255, 0.05)',
                'Valid': 'rgba(255, 193, 7, 0.1)',
                'Test': 'rgba(0, 0, 0, 0)' // Transparent
            };

            let startIdx = 0;
            let currentSegment = segments[0];

            // Helper to fill rect
            const fillSegment = (start, end, segment) => {
                const startX = xAxis.getPixelForValue(start);
                const endX = xAxis.getPixelForValue(end);
                // Only draw if within current view (simple check)
                if (endX < 0 || startX > chart.width) return;

                const color = colors[segment] || 'rgba(0,0,0,0)';
                ctx.fillStyle = color;
                ctx.fillRect(startX, yAxis.top, endX - startX, yAxis.bottom - yAxis.top);

                // Label
                if (endX - startX > 50) {
                    ctx.fillStyle = 'rgba(255,255,255,0.3)';
                    ctx.font = '12px ' + Chart.defaults.font.family;
                    ctx.textAlign = 'center';
                    ctx.fillText(segment, (startX + endX) / 2, yAxis.top + 20);
                }
            };

            for (let i = 1; i <= segments.length; i++) {
                if (i === segments.length || segments[i] !== currentSegment) {
                    fillSegment(startIdx, i - 1, currentSegment);
                    startIdx = i;
                    if (i < segments.length) currentSegment = segments[i];
                }
            }
        }
    };

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: equityCurve.dates,
            datasets: [
                {
                    label: 'Strategy',
                    data: equityCurve.strategy,
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'Benchmark',
                    data: equityCurve.benchmark,
                    borderColor: '#aaaaaa',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    tension: 0.1,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    ticks: { color: '#aaaaaa', maxTicksLimit: 10 },
                    grid: { color: '#333333' }
                },
                y: {
                    ticks: { color: '#aaaaaa' },
                    grid: { color: '#333333' }
                }
            }
        },
        plugins: [segmentPlugin]
    });
}
