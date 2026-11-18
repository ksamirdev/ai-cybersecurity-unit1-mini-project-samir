"""Task 1: confirm the virtual environment has every required package and the dataset loads."""
import sys

import matplotlib
import numpy
import pandas
import seaborn
import sklearn

from common import load_data

print(f"Python       {sys.version.split()[0]}  ({sys.prefix})")
for mod in (numpy, pandas, sklearn, matplotlib, seaborn):
    print(f"{mod.__name__:<12} {mod.__version__}")

df = load_data()
print(f"Dataset      {df.shape[0]} rows x {df.shape[1]} columns - setup OK")
