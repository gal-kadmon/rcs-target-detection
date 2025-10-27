from oct2py import octave
import os
from src import train_random_forest
import pandas as pd

# -----------------------------
# Paths setup
# -----------------------------
base_dir = os.path.dirname(__file__)
os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
octave.addpath(os.path.join(base_dir, 'src'))

output_filename = os.path.join(base_dir, 'data', 'rcs_output.csv')

# -----------------------------
# Generate synthetic RCS data with Octave
# -----------------------------
N = 5000
octave.generate_rcs_data(output_filename, N, nout=0)

# -----------------------------
# Read the generated data
# -----------------------------
df = pd.read_csv(output_filename, header=None)
df.columns = ['Range', 'SNR', 'target_class']

# -----------------------------
# Train Random Forest model
# -----------------------------
train_random_forest.train_random_forest(output_filename, n_trials=30)

