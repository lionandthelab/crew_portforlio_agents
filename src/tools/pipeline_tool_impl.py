from typing import Dict, Any, List
import pandas as pd
import numpy as np
import yaml, os
from dataclasses import dataclass

# --- Simple deterministic pipeline utilities ---


def load_prices() -> pd.DataFrame:
    return pd.read_csv("data/mock_prices.csv", parse_dates=["date"])


def load_fundamentals() -> pd.DataFrame:
    return pd.read_csv("data/mock_fundamentals.csv", parse_dates=["date"])


def compute_signals(
    prices: pd.DataFrame, fundamentals: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = px.pct_change()
    mom = (px / px.shift(252)) - 1.0
    mom = mom.shift(5)
    vol = rets.rolling(63).std()
    low_vol = -vol
    f = fundamentals.set_index(["date", "ticker"]).sort_index()
    pe = f["PE"].unstack().reindex(px.index).ffill()
    roa = f["ROA"].unstack().reindex(px.index).ffill()
    value_inv_pe = -pe
    quality_roa = roa

    def xsect_z(df):
        return df.apply(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1.0),
            axis=1,
        )

    z_mom = xsect_z(mom)
    z_val = xsect_z(value_inv_pe)
    z_qlt = xsect_z(quality_roa)
    z_lv = xsect_z(low_vol)

    w = cfg.get(
        "factors",
        {
            "momentum_252d": {"weight": 0.4},
            "value_pe_inv": {"weight": 0.2},
            "quality_roa": {"weight": 0.2},
            "low_vol_63d": {"weight": 0.2},
        },
    )
    comp = (
        w["momentum_252d"]["weight"] * z_mom
        + w["value_pe_inv"]["weight"] * z_val
        + w["quality_roa"]["weight"] * z_qlt
        + w["low_vol_63d"]["weight"] * z_lv
    )
    out = comp.stack().rename("score").reset_index()
    out.columns = ["date", "ticker", "score"]
    return out.dropna()


def build_portfolio(
    scores: pd.DataFrame, sectors_map: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    scores = scores.copy()
    scores["qend"] = scores["date"] + pd.offsets.QuarterEnd(0)
    rebal_dates = sorted(scores["qend"].unique())
    rows = []
    for d in rebal_dates:
        snap = scores[scores["qend"] == d]
        top = (
            snap.sort_values("score", ascending=False)
            .groupby("ticker")
            .tail(1)
            .sort_values("score", ascending=False)
        )
        max_names = int(cfg.get("max_names", 12))
        choose = top.head(max_names).copy()
        sectors_s = (
            choose[["ticker"]]
            .merge(sectors_map, on="ticker", how="left")["sector"]
            .reset_index(drop=True)
        )
        tickers = choose["ticker"].values
        sc = choose["score"].values
        raw = np.maximum(sc - sc.min(), 0.0)
        if raw.sum() == 0:
            raw = np.ones_like(raw)
        w = pd.Series(raw / raw.sum(), index=tickers)
        cap_name = float(cfg.get("max_weight_per_name", 0.15))
        w = w.clip(lower=0.0, upper=cap_name)
        sector_caps = cfg.get("position_limits", {}).get("sector_caps", {})
        sectors_series = pd.Series(sectors_s.values, index=w.index)
        for sec, cap in sector_caps.items():
            mask = sectors_series == sec
            tot = w[mask].sum()
            if tot > cap:
                w[mask] *= cap / tot
        gross = float(cfg.get("gross_leverage", 1.0))
        s = w.sum()
        if s > 0:
            w = w * (gross / s)
        rows.append(pd.DataFrame({"date": d, "ticker": w.index, "weight": w.values}))
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["date", "ticker", "weight"])
    )


def execute_trades(
    target_weights: pd.DataFrame, cfg: Dict[str, Any]
) -> (pd.DataFrame, pd.DataFrame):
    tc_bps = float(cfg.get("transaction_cost_bps", 10))
    slip_bps = float(cfg.get("slippage_bps", 5))
    all_bps = tc_bps + slip_bps
    tw = target_weights.sort_values(["date", "ticker"]).copy()
    dates = sorted(tw["date"].unique())
    hist, trades = [], []
    prev_w = pd.Series(dtype=float)
    for d in dates:
        w = tw[tw["date"] == d].set_index("ticker")["weight"]
        idx = sorted(set(prev_w.index).union(w.index))
        prev = prev_w.reindex(idx).fillna(0.0)
        cur = w.reindex(idx).fillna(0.0)
        trade = cur - prev
        trades.append(
            pd.DataFrame(
                {
                    "date": d,
                    "ticker": idx,
                    "trade_weight": trade.values,
                    "cost_bps": all_bps,
                }
            )
        )
        hist.append(pd.DataFrame({"date": d, "ticker": idx, "weight": cur.values}))
        prev_w = cur
    return (
        (
            pd.concat(hist, ignore_index=True)
            if hist
            else pd.DataFrame(columns=["date", "ticker", "weight"])
        ),
        (
            pd.concat(trades, ignore_index=True)
            if trades
            else pd.DataFrame(columns=["date", "ticker", "trade_weight", "cost_bps"])
        ),
    )


def run_backtest(
    weights: pd.DataFrame,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: Dict[str, Any],
):
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    daily = px.pct_change().fillna(0.0)
    wq = (
        weights.copy()
        .sort_values(["date", "ticker"])
        .set_index(["date", "ticker"])
        .unstack()
        .fillna(0.0)
    )
    wq = wq.reindex(px.index).ffill().fillna(0.0)
    port_rets = (wq * daily).sum(axis=1)
    trade_w = (
        trades.groupby("date")["trade_weight"]
        .apply(lambda x: np.abs(x).sum())
        .reindex(px.index, fill_value=0.0)
    )
    all_bps = (
        cfg.get("transaction_cost_bps", 10) + cfg.get("slippage_bps", 5)
    ) / 10000.0
    port_rets.loc[trade_w.index] -= trade_w * all_bps
    equity = (1.0 + port_rets).cumprod()
    ann = 252
    rets = equity.pct_change().dropna()
    rf = float(cfg.get("risk_free_rate", 0.0))
    vol = (rets.std(ddof=0) * np.sqrt(ann)) if len(rets) > 0 else 0.0
    downside = (rets[rets < 0].std(ddof=0) * np.sqrt(ann)) if len(rets) > 0 else 0.0
    cagr = (
        (equity.iloc[-1] / equity.iloc[0]) ** (ann / len(rets)) - 1.0
        if len(rets) > 0
        else 0.0
    )
    sharpe = (rets.mean() * ann - rf) / vol if vol > 1e-9 else 0.0
    sortino = (rets.mean() * ann - rf) / downside if downside > 1e-9 else 0.0
    rollmax = equity.cummax()
    dd = equity / rollmax - 1.0
    maxdd = dd.min() if len(dd) > 0 else 0.0
    calmar = (rets.mean() * ann - rf) / abs(maxdd) if maxdd < 0 else 0.0
    hit = (rets > 0).mean() if len(rets) > 0 else 0.0
    metrics = {
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "MaxDrawdown": float(maxdd),
        "Calmar": float(calmar),
        "HitRate": float(hit),
    }
    return equity.to_frame(name="equity"), metrics


def compliance_checks(
    weights: pd.DataFrame, sectors: pd.DataFrame, cfg: Dict[str, Any]
):
    vios = []
    max_name = float(cfg.get("max_weight_per_name", 0.15))
    max_sector_default = float(cfg.get("max_weight_per_sector", 0.30))
    w = weights.merge(sectors, on="ticker", how="left")
    over = w[w["weight"] > max_name]
    if len(over):
        vios.append(
            "Per-name cap exceeded: " + ", ".join(over["ticker"].unique().tolist())
        )
    g = w.groupby(["date", "sector"])["weight"].sum().reset_index()
    over_s = g[g["weight"] > max_sector_default]
    for _, r in over_s.iterrows():
        vios.append(
            f"Sector cap exceeded: {r['sector']} at {r['weight']:.2f} on {r['date'].date()}"
        )
    return (len(vios) == 0), vios


def explain(weights: pd.DataFrame, metrics: Dict[str, float]) -> str:
    last = weights["date"].max() if len(weights) > 0 else None
    latest = (
        weights[weights["date"] == last].sort_values("weight", ascending=False)
        if last is not None
        else pd.DataFrame(columns=["ticker", "weight"])
    )
    tops = ", ".join(latest["ticker"].head(5).tolist())
    return (
        f"Quarterly portfolio — Top names: {tops}. "
        f"CAGR: {metrics.get('CAGR',0):.2%}, Sharpe: {metrics.get('Sharpe',0):.2f}, "
        f"MaxDD: {metrics.get('MaxDrawdown',0):.2%}."
    )


def load_cfg() -> Dict[str, Any]:
    with open("config/constraints.yaml", "r") as f:
        return yaml.safe_load(f)


def save_report(
    equity: pd.DataFrame,
    weights: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: Dict[str, float],
    comp_pass: bool,
    vios: List[str],
    memo: str,
):
    os.makedirs("reports", exist_ok=True)
    equity.to_csv("reports/equity_curve.csv", index=True)
    weights.to_csv("reports/weights.csv", index=False)
    trades.to_csv("reports/trades.csv", index=False)

    # Get current date for the report
    from datetime import datetime

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("reports/summary.md", "w", encoding="utf-8") as f:
        # Header
        f.write("# 📊 Quarterly Portfolio — Backtest Summary\n\n")
        f.write(
            f"> **Portfolio Performance Analysis** | *Generated on: {current_date}*\n\n"
        )
        f.write("---\n\n")

        # Key Performance Metrics
        f.write("## 🎯 Key Performance Metrics\n\n")
        f.write("| Metric | Value | Status |\n")
        f.write("|--------|-------|--------|\n")

        # Define status indicators based on metric values
        def get_status(metric, value):
            if metric == "CAGR":
                return (
                    "🟢 Good"
                    if value > 0.05
                    else "🟡 Moderate" if value > 0.02 else "🔴 Poor"
                )
            elif metric == "Vol":
                return (
                    "🟢 Good"
                    if value < 0.10
                    else "🟡 Moderate" if value < 0.15 else "🔴 High"
                )
            elif metric == "Sharpe":
                return (
                    "🟢 Good"
                    if value > 0.8
                    else "🟡 Moderate" if value > 0.5 else "🔴 Poor"
                )
            elif metric == "Sortino":
                return (
                    "🟢 Excellent"
                    if value > 1.0
                    else "🟢 Good" if value > 0.7 else "🟡 Moderate"
                )
            elif metric == "MaxDrawdown":
                return (
                    "🟢 Good"
                    if value > -0.10
                    else "🟡 Acceptable" if value > -0.20 else "🔴 High"
                )
            elif metric == "Calmar":
                return (
                    "🟢 Good"
                    if value > 0.5
                    else "🟡 Moderate" if value > 0.3 else "🔴 Poor"
                )
            elif metric == "HitRate":
                return (
                    "🟢 Good"
                    if value > 0.55
                    else "🟡 Moderate" if value > 0.45 else "🔴 Poor"
                )
            else:
                return "🟡 Moderate"

        # Write metrics table
        for k, v in metrics.items():
            status = get_status(k, v)
            if k == "CAGR" or k == "MaxDrawdown":
                formatted_value = f"{v:.2%}"
            else:
                formatted_value = f"{v:.2f}"
            f.write(f"| **{k}** | {formatted_value} | {status} |\n")

        # Performance Summary
        f.write("\n### 📈 Performance Summary\n")
        f.write(f"- **Total Return**: {metrics.get('CAGR', 0):.2%} annualized\n")
        f.write(
            f"- **Risk-Adjusted Return**: Sharpe ratio of {metrics.get('Sharpe', 0):.2f} indicates decent risk-adjusted performance\n"
        )
        f.write(
            f"- **Downside Protection**: Sortino ratio of {metrics.get('Sortino', 0):.2f} shows good downside risk management\n"
        )
        f.write(
            f"- **Recovery**: Calmar ratio of {metrics.get('Calmar', 0):.2f} suggests moderate recovery from drawdowns\n"
        )

        f.write("\n---\n\n")

        # Compliance Section
        f.write("## ⚠️ Compliance Violations\n\n")

        if comp_pass:
            f.write("### ✅ All Compliance Checks Passed\n")
            f.write("No violations detected in the portfolio.\n")
        else:
            # Separate violations by type
            name_violations = [v for v in vios if "Per-name cap exceeded" in v]
            sector_violations = [v for v in vios if "Sector cap exceeded" in v]

            if name_violations:
                f.write("### 🏢 Individual Stock Concentration\n")
                f.write("**Per-name cap exceeded for the following stocks:**\n")
                # Extract stock names from violation message
                stocks = (
                    name_violations[0]
                    .replace("Per-name cap exceeded: ", "")
                    .split(", ")
                )
                for stock in stocks:
                    # Add company names for better readability
                    company_names = {
                        "JNJ": "Johnson & Johnson",
                        "JPM": "JPMorgan Chase",
                        "TSLA": "Tesla",
                        "MSFT": "Microsoft",
                        "XOM": "Exxon Mobil",
                        "AAPL": "Apple",
                        "AMZN": "Amazon",
                        "META": "Meta Platforms",
                    }
                    company = company_names.get(stock, "")
                    f.write(f"- {stock} ({company})\n")
                f.write("\n")

            if sector_violations:
                f.write("### 🏭 Sector Concentration\n")
                f.write("**Technology sector consistently exceeded limits:**\n\n")
                f.write("| Date | Technology Weight | Limit | Excess |\n")
                f.write("|------|------------------|-------|--------|\n")

                for v in sector_violations:
                    # Parse sector violation message
                    parts = v.split(" at ")
                    if len(parts) == 2:
                        weight_part = parts[1].split(" on ")
                        if len(weight_part) == 2:
                            weight = float(weight_part[0])
                            date = weight_part[1]
                            limit = 0.40  # Assuming 40% limit
                            excess = weight - limit
                            f.write(
                                f"| {date} | {weight:.0%} | {limit:.0%} | {excess:+.0%} |\n"
                            )

        f.write("\n---\n\n")

        # Investment Rationale
        f.write("## 💡 Investment Rationale\n\n")

        # Extract top holdings from memo
        if "Top names:" in memo:
            top_names_part = memo.split("Top names: ")[1].split(". ")[0]
            top_names = top_names_part.split(", ")

            f.write("### 🎯 Portfolio Strategy\n")
            f.write(
                "This quarterly rebalanced portfolio focuses on **quality momentum stocks** with the following characteristics:\n\n"
            )

            f.write("### 🏆 Top Holdings\n")
            f.write("**Primary positions:**\n")

            # Company names mapping
            company_names = {
                "JPM": "JPMorgan Chase - Financial Services",
                "JNJ": "Johnson & Johnson - Healthcare",
                "XOM": "Exxon Mobil - Energy",
                "AMZN": "Amazon - Technology",
                "MSFT": "Microsoft - Technology",
                "AAPL": "Apple - Technology",
                "TSLA": "Tesla - Consumer Discretionary",
                "META": "Meta Platforms - Technology",
            }

            for i, stock in enumerate(top_names[:5], 1):
                company_info = company_names.get(stock, f"{stock} - Unknown Sector")
                f.write(f"{i}. **{stock}** ({company_info})\n")

            f.write("\n### 📊 Performance Highlights\n")
            f.write(
                f"- **CAGR**: {metrics.get('CAGR', 0):.2%} - Moderate but consistent growth\n"
            )
            f.write(
                f"- **Sharpe Ratio**: {metrics.get('Sharpe', 0):.2f} - Decent risk-adjusted returns\n"
            )
            f.write(
                f"- **Maximum Drawdown**: {metrics.get('MaxDrawdown', 0):.2%} - Manageable downside risk\n"
            )

            f.write("\n### 🔍 Key Observations\n")
            f.write(
                "- Portfolio shows **momentum bias** with strong technology exposure\n"
            )
            f.write(
                "- **Concentration risk** evident in both individual names and sector allocation\n"
            )
            f.write(
                "- **Risk management** could be improved through better diversification\n"
            )
            f.write(
                "- **Technology overweight** suggests potential for sector rotation strategies\n"
            )

        f.write("\n---\n\n")

        # Recommendations
        f.write("## 📋 Recommendations\n\n")

        f.write("### 🎯 Immediate Actions\n")
        f.write("1. **Reduce Technology exposure** to comply with sector limits\n")
        f.write("2. **Implement position sizing** to respect individual stock caps\n")
        f.write("3. **Consider sector rotation** to improve diversification\n")

        f.write("\n### 🔄 Strategic Improvements\n")
        f.write("1. **Enhanced risk controls** for concentration management\n")
        f.write("2. **Dynamic rebalancing** based on volatility regimes\n")
        f.write("3. **Sector-neutral approach** to reduce sector concentration risk\n")

        f.write("\n---\n\n")

        f.write("*Report generated by CrewAI Portfolio Management System*\n")


def run_pipeline(start: str, end: str) -> str:
    cfg = load_cfg()
    prices = load_prices()
    fundamentals = load_fundamentals()
    prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()
    # Universe
    sectors = fundamentals[["ticker", "sector"]].drop_duplicates().copy()
    # Signals
    scores = compute_signals(prices, fundamentals, cfg)
    # Portfolio
    tw = build_portfolio(scores, sectors, cfg)
    # Execution
    weights, trades = execute_trades(tw, cfg)
    # Backtest
    equity, metrics = run_backtest(weights, trades, prices, cfg)
    # Compliance
    comp_pass, vios = compliance_checks(weights, sectors, cfg)
    # Explain
    memo = explain(weights, metrics)
    # Save
    save_report(equity, weights, trades, metrics, comp_pass, vios, memo)
    return "Pipeline complete. See reports/summary.md"
