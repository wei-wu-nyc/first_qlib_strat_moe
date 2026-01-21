let globalEquityCurve = null;
let chartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    fetch('data.json?t=' + new Date().getTime())
        .then(response => response.json())
        .then(data => {
            renderMetrics(data.metrics);
            globalEquityCurve = data.equity_curve;
            globalEquityCurve = data.equity_curve;
            updatePeriod('test'); // Default to Test period to show the MoE result
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
    // Geometric Equity (Compound): We divide by the base value.
    const baseStrategy = globalEquityCurve.strategy[indices[0]];
    const baseBenchmark = globalEquityCurve.benchmark[indices[0]];

    // Guard against zero division
    const safeDiv = (val, base) => (base !== 0 && base !== undefined && base !== null) ? val / base : 1;

    // Create filtered arrays
    const fDates = indices.map(i => globalEquityCurve.dates[i]);
    const fSegment = indices.map(i => globalEquityCurve.segments[i]);
    const fRegime = indices.map(i => globalEquityCurve.regimes ? globalEquityCurve.regimes[i] : 0); // Default 0

    // Subtract 1 to show as "Return" (1.5 -> 0.5 i.e. 50%)
    const fStrategy = indices.map(i => safeDiv(globalEquityCurve.strategy[i], baseStrategy) - 1);
    const fBenchmark = indices.map(i => safeDiv(globalEquityCurve.benchmark[i], baseBenchmark) - 1);

    // Re-base MAs using the BENCHMARK Base (Since they are MAs of the Benchmark)
    const fMA20 = indices.map(i => globalEquityCurve.bench_ma20 ? safeDiv(globalEquityCurve.bench_ma20[i], baseBenchmark) - 1 : null);
    const fMA60 = indices.map(i => globalEquityCurve.bench_ma60 ? safeDiv(globalEquityCurve.bench_ma60[i], baseBenchmark) - 1 : null);

    const filteredCurve = {
        dates: fDates,
        strategy: fStrategy,
        benchmark: fBenchmark,
        ma20: fMA20,
        ma60: fMA60,
        regimes: fRegime,
        segments: fSegment
    };

    renderEquityChart(filteredCurve);

    // Calculate metrics for this period immediately
    const metrics = calculateMetrics(fDates, fStrategy, fBenchmark);
    renderMetrics(metrics);
}

function resetZoom() {
    if (chartInstance) {
        chartInstance.resetZoom();
        // Recalculate based on current full range
        // Or just re-trigger the current period update
        const activeBtn = document.querySelector('.btn.active');
        const mode = activeBtn ? activeBtn.textContent.toLowerCase() : 'all';
        // Map button text to mode key
        let modeKey = 'all';
        if (mode.includes('testing')) modeKey = 'test';
        else if (mode.includes('valid')) modeKey = 'valid_test';

        updatePeriod(modeKey);
    }
}

// Client-side Metrics Calculation
function calculateMetrics(dates, strategyRets, benchmarkRets) {
    if (!strategyRets || strategyRets.length < 2) return {
        annualized_return: 0,
        max_drawdown: 0,
        sharpe_ratio: 0,
        information_ratio: 0
    };

    // strategyRets are Cumulative Returns (e.g. 0.0, 0.01, 0.05...)
    // We need Daily Returns: R_t = (1 + Cum_t) / (1 + Cum_t-1) - 1

    const dailyRets = [];
    for (let i = 1; i < strategyRets.length; i++) {
        const valPrev = 1 + strategyRets[i - 1];
        const valCurr = 1 + strategyRets[i];
        dailyRets.push(valCurr / valPrev - 1);
    }

    // 1. Annualized Return
    // Total Return = (1 + FinalCum) - 1
    const totalRet = strategyRets[strategyRets.length - 1]; // This IS the cumulative return
    const days = dates.length; // Approximate trading days
    // Ann Ret = (1 + Total)^(252/days) - 1
    const years = days / 252;
    const annRet = years > 0 ? Math.pow(1 + totalRet, 1 / years) - 1 : 0;

    // 2. Max Drawdown
    let mdd = 0;
    let peak = -99999;

    // Iterate through CUMULATIVE values (1 + ret)
    // strategyRets is (Limit - 1). So (1 + x) restores the Equity Index.
    for (let x of strategyRets) {
        let val = 1 + x;
        if (val > peak) peak = val;
        let dd = (val - peak) / peak;
        if (dd < mdd) mdd = dd;
    }

    // 3. Sharpe Ratio
    // Mean(Daily) / Std(Daily) * sqrt(252)
    const n = dailyRets.length;
    if (n === 0) return { annualized_return: annRet, max_drawdown: mdd, sharpe_ratio: 0, information_ratio: 0 };

    const mean = dailyRets.reduce((a, b) => a + b, 0) / n;
    // Let's stick to simple Sharpe
    const variance = dailyRets.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
    const std = Math.sqrt(variance);
    const sharpe = std !== 0 ? (mean / std) * Math.sqrt(252) : 0;

    // 4. Information Ratio (vs Benchmark)
    // IR = Mean(ActiveRet) / Std(ActiveRet)
    // Active Ret = DailyStrat - DailyBench
    // Only if benchmark exists
    let ir = 0;
    if (benchmarkRets && benchmarkRets.length === strategyRets.length) {
        const activeDaily = [];
        for (let i = 1; i < strategyRets.length; i++) {
            const sDaily = (1 + strategyRets[i]) / (1 + strategyRets[i - 1]) - 1;
            const bDaily = (1 + benchmarkRets[i]) / (1 + benchmarkRets[i - 1]) - 1;
            activeDaily.push(sDaily - bDaily);
        }
        const activeMean = activeDaily.reduce((a, b) => a + b, 0) / activeDaily.length;
        const activeVar = activeDaily.reduce((a, b) => a + Math.pow(b - activeMean, 2), 0) / activeDaily.length;
        const activeStd = Math.sqrt(activeVar);
        ir = activeStd !== 0 ? (activeMean / activeStd) * Math.sqrt(252) : 0;
    }

    return {
        annualized_return: annRet,
        max_drawdown: mdd,
        sharpe_ratio: sharpe,
        information_ratio: ir
    };
}

function renderMetrics(metrics) {
    const formatPercent = (val) => (val * 100).toFixed(2) + '%';
    const formatNumber = (val) => val.toFixed(4);

    setMetric('metric-return', metrics.annualized_return, formatPercent);
    setMetric('metric-sharpe', metrics.sharpe_ratio, formatNumber);
    setMetric('metric-mdd', metrics.max_drawdown, formatPercent);
    setMetric('metric-ir', metrics.information_ratio, formatNumber);
}

function setMetric(id, value, formatter) {
    const el = document.getElementById(id);
    if (!el || value === undefined) return;
    el.textContent = formatter(value);

    // Color coding for returns
    if (id === 'metric-return' || id === 'metric-sharpe') {
        el.style.color = value >= 0 ? 'var(--accent)' : 'var(--negative)';
    } else if (id === 'metric-mdd') {
        el.style.color = 'var(--negative)';
    }
}

function renderEquityChart(equityCurve) {
    const ctx = document.getElementById('equityChart').getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    // Plugin to draw detailed background REGIMES
    const regimePlugin = {
        id: 'regimeBackground',
        beforeDraw: (chart) => {
            const ctx = chart.ctx;
            const xAxis = chart.scales.x;
            const yAxis = chart.scales.y;
            const regimes = equityCurve.regimes; // [1, 0, -1, ...]

            if (!regimes) return;

            const colors = {
                '1': 'rgba(76, 175, 80, 0.2)',    // Green (Bull) - More visible
                '-1': 'rgba(244, 67, 54, 0.25)',   // Red (Bear) - More visible
                '0': 'rgba(158, 158, 158, 0.1)'  // Grey (Choppy)
            };

            let startIdx = 0;
            let currentRegime = regimes[0];

            // Helper to fill rect
            const fillRegime = (start, end, regime) => {
                const startX = xAxis.getPixelForValue(start);
                const endX = xAxis.getPixelForValue(end);

                if (endX < 0 || startX > chart.width) return;

                const color = colors[regime] || 'rgba(0,0,0,0)';
                ctx.fillStyle = color;
                ctx.fillRect(startX, yAxis.top, endX - startX, yAxis.bottom - yAxis.top);
            };

            for (let i = 1; i <= regimes.length; i++) {
                if (i === regimes.length || regimes[i] !== currentRegime) {
                    fillRegime(startIdx, i - 1, currentRegime);
                    startIdx = i;
                    if (i < regimes.length) currentRegime = regimes[i];
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
                    borderColor: '#00bcd4', // Cyan
                    backgroundColor: 'rgba(0, 188, 212, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: { value: 0 },
                    pointRadius: 0
                },
                {
                    label: 'Benchmark',
                    data: equityCurve.benchmark,
                    borderColor: '#aaaaaa',
                    borderWidth: 1.5,
                    borderDash: [0, 0],
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: 'MA20',
                    data: equityCurve.ma20,
                    borderColor: '#ff9800', // Orange
                    borderWidth: 1.5,
                    borderDash: [2, 2],
                    tension: 0.4,
                    pointRadius: 0,
                    hidden: false // Visible by default
                },
                {
                    label: 'MA60',
                    data: equityCurve.ma60,
                    borderColor: '#9c27b0', // Purple
                    borderWidth: 1.5,
                    tension: 0.4,
                    pointRadius: 0,
                    hidden: false // Visible by default
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
                legend: { labels: { color: '#ffffff' } },
                zoom: {
                    pan: { enabled: true, mode: 'x' },
                    zoom: {
                        wheel: { enabled: true },
                        drag: { enabled: true },
                        mode: 'x', // Zoom X axis only
                        onZoomComplete: ({ chart }) => {
                            // Dynamic Metrics Calculation on Zoom
                            const scales = chart.scales;
                            const minIdx = Math.max(0, Math.floor(scales.x.min));
                            const maxIdx = Math.min(equityCurve.dates.length - 1, Math.ceil(scales.x.max));

                            // Slice data
                            const dates = equityCurve.dates.slice(minIdx, maxIdx + 1);
                            const strat = equityCurve.strategy.slice(minIdx, maxIdx + 1);
                            const bench = equityCurve.benchmark.slice(minIdx, maxIdx + 1);

                            // Re-normalize for Metric Calculation
                            const baseS = 1 + strat[0];
                            const normStrat = strat.map(v => (1 + v) / baseS - 1);

                            const baseB = 1 + bench[0];
                            const normBench = bench.map(v => (1 + v) / baseB - 1);

                            const m = calculateMetrics(dates, normStrat, normBench);
                            renderMetrics(m);
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y !== null) label += (context.parsed.y * 100).toFixed(2) + '%';
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#aaaaaa', maxTicksLimit: 10 },
                    grid: { color: '#333333' }
                },
                y: {
                    ticks: {
                        color: '#aaaaaa',
                        callback: function (value) { return (value * 100).toFixed(0) + '%'; }
                    },
                    grid: { color: '#333333' }
                }
            }
        },
        plugins: [regimePlugin]
    });
}
