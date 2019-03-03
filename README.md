# porters-solver
A tool to optimize business decisions for profit using constraints.

# Requirements

- scikit-learn
- python3.x
- scipy
- pandas
- numpy

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

![image](/images/overview.png)

![image](/images/volumemodel.png)

![image](/images/constraints.png)
