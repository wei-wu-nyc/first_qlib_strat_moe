import json
import pandas as pd
import numpy as np

TRADING_DAYS = 250

def calculate_metrics_user_formula(daily_ret_strat, daily_ret_bench):
    if len(daily_ret_strat) < 2:
        return [np.nan] * 9

    # 1. Annualized Return (Geometric)
    # User said "Strategy annual return". Geometric is standard.
    # (1 + Total)^(250/N) - 1
    total_strat = (1 + daily_ret_strat).prod()
    n_days = len(daily_ret_strat)
    ann_ret_strat = (total_strat ** (TRADING_DAYS / n_days)) - 1
    
    total_bench = (1 + daily_ret_bench).prod()
    ann_ret_bench = (total_bench ** (TRADING_DAYS / n_days)) - 1
    
    # 2. Annualized Outperformance
    outperf = ann_ret_strat - ann_ret_bench
    
    # 3. Strat/Bench Volatility
    # stdev(daily) * sqrt(250)
    # Using ddof=1 for sample stdev
    strat_vol = daily_ret_strat.std(ddof=1) * np.sqrt(TRADING_DAYS)
    bench_vol = daily_ret_bench.std(ddof=1) * np.sqrt(TRADING_DAYS)
    
    # 4. Sharpe Ratio (Return / Vol)
    # User formula: "Strategy annual return/Strategy Annualized Volatility"
    # Note: Usually Sharpe is (Rp - Rf) / Vol. Assuming Rf=0 as per user formula implication.
    sharpe_strat = ann_ret_strat / strat_vol if strat_vol != 0 else np.nan
    sharpe_bench = ann_ret_bench / bench_vol if bench_vol != 0 else np.nan
    
    # 5. Information Ratio
    # Formula: (Strat Ann - Bench Ann) / (stdev(diff) * sqrt(250))
    diff_daily = daily_ret_strat - daily_ret_bench
    tracking_error = diff_daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    ir = (ann_ret_strat - ann_ret_bench) / tracking_error if tracking_error != 0 else np.nan
    
    # 6. Winning Ratios
    strat_pos = (daily_ret_strat > 0).mean()
    bench_pos = (daily_ret_bench > 0).mean()
    beat_bench = (daily_ret_strat > daily_ret_bench).mean()
    
    return (
        ann_ret_strat, ann_ret_bench,
        outperf,
        strat_vol, bench_vol,
        sharpe_strat, sharpe_bench,
        ir,
        strat_pos, bench_pos, beat_bench
    )

def main():
    base_path = r"d:\Work\Antigravity\qlib_strategy_test"
    with open(f"{base_path}\\dashboard\\data.json", 'r') as f:
        data = json.load(f)
    
    ec = data['equity_curve']
    
    df = pd.DataFrame({
        'date': ec['dates'],
        'strategy_cum': ec['strategy'],
        'benchmark_cum': ec['benchmark'],
        'regime': ec['regimes'],
        'segment': ec['segments']
    })
    
    # Calculate Daily Returns
    df['strat_ret'] = df['strategy_cum'].pct_change().fillna(0)
    df['bench_ret'] = df['benchmark_cum'].pct_change().fillna(0)
    
    segments = ['Train', 'Valid', 'Test']
    regimes = [
        ('Uptrend', 1),
        ('Downtrend', -1),
        ('Choppy', 0),
        ('All', None)
    ]
    
    # HTML Header
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Performance Analysis</title>
        <link rel="stylesheet" href="style.css">
        <style>
            body { padding: 40px; background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            table { width: auto; border-collapse: collapse; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); } 
            th, td { padding: 12px 10px; text-align: right; border-bottom: 1px solid #333; font-size: 0.95rem; }
            th { background-color: #1f1f1f; color: #00bcd4; text-align: right; font-weight: 600; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.5px; }
            .text-left { text-align: left; padding-right: 20px; }
            tr:hover { background-color: #1e1e1e; }
            .pos { color: #4caf50; }
            .neg { color: #f44336; }
            h1 { color: #ffffff; font-weight: 300; margin-bottom: 10px; }
            .back-link { display: inline-block; margin-bottom: 20px; color: #00bcd4; text-decoration: none; border: 1px solid #00bcd4; padding: 8px 16px; border-radius: 4px; transition: all 0.2s; }
            .back-link:hover { background: rgba(0, 188, 212, 0.1); }
            tr.sep { border-bottom: 2px solid #444; }
        </style>
    </head>
    <body>
        <a href="index.html" class="back-link">&larr; Back to Dashboard</a>
        <h1>Regime Performance Analysis</h1>
        <table>
            <thead>
                <tr>
                    <th colspan="2" class="text-left">Context</th>
                    <th colspan="3">Annualized Returns</th>
                    <th colspan="2">Risk (Vol)</th>
                    <th colspan="3">Risk-Adjusted</th>
                    <th colspan="3">Winning Ratios</th>
                </tr>
                <tr>
                    <th class="text-left">Regime</th>
                    <th class="text-left">Period</th>
                    <th style="min-width: 60px;">Strat</th>
                    <th style="min-width: 60px;">Bench</th>
                    <th style="min-width: 60px;">OutPerf</th>
                    <th style="min-width: 60px;">Strat</th>
                    <th style="min-width: 60px;">Bench</th>
                    <th style="min-width: 60px;">Strat<br>Sharpe</th>
                    <th style="min-width: 60px;">Bench<br>Sharpe</th>
                    <th style="min-width: 60px;">InfoRatio</th>
                    <th style="min-width: 60px;">Strat<br>Pos</th>
                    <th style="min-width: 60px;">Bench<br>Pos</th>
                    <th style="min-width: 60px;">Beat<br>Bench</th>
                </tr>
            </thead>
            <tbody>
    """
    
    def fmt_pct(x): 
        if np.isnan(x): return "-"
        cls = 'pos' if x >= 0 else 'neg'
        return f'<span class="{cls}">{x:.2%}</span>'
        
    def fmt_num(x):
        if np.isnan(x): return "-"
        cls = 'pos' if x >= 0 else 'neg'
        return f'<span class="{cls}">{x:.2f}</span>'

    for regime_name, regime_val in regimes:
        for i, seg in enumerate(segments):
            mask = (df['segment'] == seg)
            if regime_val is not None:
                mask = mask & (df['regime'] == regime_val)
            
            subset = df[mask]
            
            metrics = calculate_metrics_user_formula(subset['strat_ret'], subset['bench_ret'])
            
            # Use explicit class text-left for the text columns
            
            row_class = "sep" if i == 2 else ""
            
            # HTML Row Construction
            if i == 0:
                html += f'<tr class="{row_class}"><td rowspan="3" class="text-left" style="vertical-align: middle; font-weight: bold; border-right: 1px solid #333;">{regime_name}</td>'
            else:
                html += f'<tr class="{row_class}">'
                
            html += f'<td class="text-left">{seg}</td>'
            html += f'<td>{fmt_pct(metrics[0])}</td>' # Strat Ann
            html += f'<td>{fmt_pct(metrics[1])}</td>' # Bench Ann
            html += f'<td>{fmt_pct(metrics[2])}</td>' # Outperf
            html += f'<td>{fmt_pct(metrics[3])}</td>' # Strat Vol
            html += f'<td>{fmt_pct(metrics[4])}</td>' # Bench Vol
            html += f'<td>{fmt_num(metrics[5])}</td>' # Strat Sharpe
            html += f'<td>{fmt_num(metrics[6])}</td>' # Bench Sharpe
            html += f'<td>{fmt_num(metrics[7])}</td>' # IR
            html += f'<td>{fmt_pct(metrics[8])}</td>' # Strat Pos
            html += f'<td>{fmt_pct(metrics[9])}</td>' # Bench Pos
            html += f'<td>{fmt_pct(metrics[10])}</td>' # Beat Bench
            html += '</tr>'
            
    html += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open(f"{base_path}\\dashboard\\analysis.html", "w", encoding='utf-8') as f:
        f.write(html)
        
    print(f"Generated analysis.html")

if __name__ == "__main__":
    main()
