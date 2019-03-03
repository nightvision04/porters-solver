# porters-solver
A tool to optimize business decisions for profit using constraints.

# Requirements

- python3.x
- scikit-learn
- scipy
- pandas
- numpy

It is recommended to use a stable package of Anaconda as your python client.

# Quickstart

For normal-sized datasets (samples > 10000), run:

```python3
cd porters-solver
python scripts/porters-analysis.py
```

For tiny datasets (samples < 200), run:

```python3
cd porters-solver
python scripts/porters-analysis-light.py
```

# How It Works

This library combines scikit-learn's Linear Regression Model and BFGS algorithm to predict which decisions are expected to maximize business profitability. Decisions are first assigned significance values in the form of coefficients after fitting the dataset model.

Next, budgets and constraints are set to focus optimization within feasible bounds. Price elasticity is tested, and the marginal product of included expenses are maximized.

A decisions list, as well as the predicted net profit is displayed. It is encouraged to use this information to perform gap analysis and improve feature selection after each business cycle.

![image](/images/overview.png)

![image](/images/volumemodel.png)

![image](/images/constraints.png)
