import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from scipy.optimize import minimize


ASSET_COLS = ["SPY", "QQQ", "TLT", "GLD", "EEM", "IWM", "VNQ", "LQD"]


def sharpe_ratio(r, annualization=252):
    r = pd.Series(r).dropna()
    if r.std() == 0:
        return 0.0
    return float(np.sqrt(annualization) * r.mean() / r.std())


def max_drawdown(r):
    r = pd.Series(r).dropna()
    cumulative = (1 + r).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    return float(drawdown.min())


def annual_volatility(r, annualization=252):
    r = pd.Series(r).dropna()
    return float(r.std() * np.sqrt(annualization))


def cumulative_return(r):
    r = pd.Series(r).dropna()
    return float((1 + r).cumprod().iloc[-1] - 1)


def min_variance_weights(cov_matrix):
    n = cov_matrix.shape[0]
    init_weights = np.ones(n) / n

    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights

    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    })

    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(
        portfolio_variance,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        return result.x
    return init_weights


def run_app(start_date, end_date, transaction_cost):
    data = pd.read_csv("data/processed_data.csv", index_col=0, parse_dates=True).sort_index()
    ai_weights = pd.read_csv("results/ai_weights.csv", index_col=0, parse_dates=True)
    regime_summary = pd.read_csv("results/regime_summary.csv", index_col=0)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered = data.loc[start_date:end_date].copy()
    if filtered.empty:
        raise gr.Error("No data available in that date range.")

    returns = filtered[ASSET_COLS].copy()

    # equal weight
    n_assets = len(ASSET_COLS)
    ew_weights = np.ones(n_assets) / n_assets
    ew_returns = returns.dot(ew_weights)

    # mean variance rolling
    lookback = 252
    mv_returns = []
    mv_dates = []

    full_returns = data[ASSET_COLS].copy()

    for date in returns.index:
        full_loc = full_returns.index.get_loc(date)
        if full_loc < lookback:
            continue

        train_window = full_returns.iloc[full_loc - lookback:full_loc]
        cov = train_window.cov().values
        weights = min_variance_weights(cov)

        day_ret = returns.loc[date].values @ weights
        turnover_penalty = transaction_cost * np.sum(np.abs(weights - ew_weights))
        mv_returns.append(day_ret - turnover_penalty)
        mv_dates.append(date)

    mv_series = pd.Series(mv_returns, index=mv_dates, name="Mean Variance")

    # AI allocation from saved weights
    ai_weights_filtered = ai_weights.loc[start_date:end_date].copy()
    common_idx = returns.index.intersection(ai_weights_filtered.index).intersection(mv_series.index)

    if len(common_idx) == 0:
        raise gr.Error("No overlapping dates available for comparison.")

    ew_series = ew_returns.loc[common_idx].copy()
    mv_series = mv_series.loc[common_idx].copy()
    ai_weights_filtered = ai_weights_filtered.loc[common_idx].copy()
    ai_returns = (ai_weights_filtered.values * returns.loc[common_idx].values).sum(axis=1)
    ai_series = pd.Series(ai_returns, index=common_idx, name="AI Allocation")

    comparison = pd.concat(
        [
            ew_series.rename("Equal Weight"),
            mv_series.rename("Mean Variance"),
            ai_series.rename("AI Allocation")
        ],
        axis=1
    ).dropna()

    comparison_total = (1 + comparison).cumprod()

    # metrics
    metrics = pd.DataFrame({
        col: {
            "Sharpe": sharpe_ratio(comparison[col]),
            "MaxDrawdown": max_drawdown(comparison[col]),
            "Volatility": annual_volatility(comparison[col]),
            "CumulativeReturn": cumulative_return(comparison[col])
        }
        for col in comparison.columns
    }).T.reset_index().rename(columns={"index": "Strategy"})

    # plot 1: cumulative performance
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for col in comparison_total.columns:
        ax1.plot(comparison_total.index, comparison_total[col], label=col)
    ax1.set_title("Out-of-Sample Strategy Comparison")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value")
    ax1.grid(True)
    ax1.legend()

    # plot 2: average AI allocation
    avg_weights = ai_weights_filtered.mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    avg_weights.plot(kind="bar", ax=ax2)
    ax2.set_title("Average AI Portfolio Allocation")
    ax2.set_ylabel("Average Weight")
    ax2.grid(True, axis="y")

    regime_out = regime_summary.copy().reset_index()

    return fig1, metrics, fig2, regime_out


demo = gr.Interface(
    fn=run_app,
    inputs=[
        gr.Textbox(value="2018-01-01", label="Start Date (YYYY-MM-DD)"),
        gr.Textbox(value="2024-12-31", label="End Date (YYYY-MM-DD)"),
        gr.Slider(minimum=0.0, maximum=0.01, value=0.001, step=0.0005, label="Transaction Cost")
    ],
    outputs=[
        gr.Plot(label="Strategy Comparison"),
        gr.Dataframe(label="Metrics"),
        gr.Plot(label="Average AI Allocation"),
        gr.Dataframe(label="Regime Summary")
    ],
    title="PortfolioPilot",
    description="Regime-aware AI portfolio allocation under market frictions."
)

if __name__ == "__main__":
    demo.launch()