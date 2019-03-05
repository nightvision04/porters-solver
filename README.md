# porters-solver
A tool for optimizing profit-driven business decisions using regression, objective minimization, and constraints.

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
cd porters-solver/scripts
python scripts/porters-analysis.py
```

For tiny datasets (samples < 200), run:

```python3
cd porters-solver/scripts
python scripts/porters-analysis-light.py
```

# How It Works

This library combines scikit-learn's Linear Regression Model and BFGS algorithm to predict which decisions are expected to maximize business profitability.

First, decisions are assigned significance while fitting the dataset model. Next, budgets and constraints are set to focus optimization within feasible bounds. Price elasticity is tested, and the marginal product of included expenses are maximized. Finally, a decisions list - as well as the predicted net profit - is displayed. It is encouraged to use this information to perform gap analysis and improve feature selection after each business cycle.

<img src="/images/overview.png" width="680">

<img src="/images/volumemodel.png" width="850">

<img src="/images/constraints.png" width="900">

I encourage any feedback or contributions to the project.
