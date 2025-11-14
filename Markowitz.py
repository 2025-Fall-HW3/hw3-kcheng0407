import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
import sys

warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

start = "2019-01-01"
end = "2024-04-01"

df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust=False)
    df[asset] = raw["Adj Close"]

df_returns = df.pct_change().fillna(0)


class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        n_assets = len(assets)
        self.portfolio_weights.loc[:, :] = 0.0
        self.portfolio_weights[assets] = 1.0 / n_assets

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]

        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        self.portfolio_weights.loc[:, :] = 0.0

        for i in range(self.lookback + 1, len(df)):
            window = df_returns[assets].iloc[i - self.lookback : i]

            sigma = window.std()

            sigma = sigma.replace(0, np.nan)
            sigma = sigma.fillna(sigma.mean())
            sigma = sigma.replace(0, 1e-6)

            inv_sigma = 1.0 / sigma
            weights = inv_sigma / inv_sigma.sum()

            self.portfolio_weights.loc[df.index[i], assets] = weights.values

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]

        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]
            self.portfolio_weights.loc[df.index[i], assets] = self.mv_opt(
                R_n, self.gamma
            )

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                w = model.addMVar(n, lb=0.0, ub=1.0, name="w")

                model.addConstr(w.sum() == 1.0, name="budget")

                ret = gp.quicksum(float(mu[j]) * w[j] for j in range(n))

                risk = gp.quicksum(
                    float(Sigma[i, j]) * w[i] * w[j]
                    for i in range(n)
                    for j in range(n)
                )

                obj = ret - 0.5 * gamma * risk
                model.setObjective(obj, gp.GRB.MAXIMIZE)

                model.optimize()

                if model.status == gp.GRB.INF_OR_UNBD:
                    print(
                        "Model status is INF_OR_UNBD. Reoptimizing with DualReductions set to 0."
                    )
                elif model.status == gp.GRB.INFEASIBLE:
                    print("Model is infeasible.")
                elif model.status == gp.GRB.INF_OR_UNBD:
                    print("Model is infeasible or unbounded.")

                solution = []
                if model.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                    for i in range(n):
                        var = model.getVarByName(f"w[{i}]")
                        solution.append(var.X)
                else:
                    solution = [1.0 / n] * n

        return solution

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )
    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )
    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )
    parser.add_argument(
        "--report",
        action="append",
        help="Report for evaluation metric",
    )

    args = parser.parse_args()
    judge = AssignmentJudge()
    judge.run_grading(args)
