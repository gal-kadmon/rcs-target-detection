from oct2py import octave
import os

from sklearn.metrics import f1_score
from src import train_random_forest
from src import train_mlp

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths setup
# -----------------------------
base_dir = os.path.dirname(__file__)
os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
octave.addpath(os.path.join(base_dir, 'src'))

data_file = os.path.join(base_dir, 'data', 'rcs_output.csv')

# -----------------------------
# Generate synthetic RCS data with Octave
# -----------------------------
N = 5000
#octave.generate_rcs_data(data_file, N, nout=0)

# -----------------------------
# Read the generated data
# -----------------------------
df = pd.read_csv(data_file, header=None)
df.columns = ['Range', 'SNR', 'target_class']

# -----------------------------
# Graph: KDE distribution of SNR per class
# -----------------------------
plt.figure(figsize=(10,6))
for cls, color, label in zip([1,2,3], ['blue','green','red'], ['Small','Medium','Large']):
    cls_data = df[df['target_class'] == cls]
    sns.kdeplot(x=cls_data['SNR'], fill=True, color=color, alpha=0.3, label=f'{label}')

plt.xlabel('SNR')
plt.ylabel('Density')
plt.title('SNR Distribution per Target Class')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_dir = "/home/gal/Desktop/Radars/rcs-target-detection/graphs"
os.makedirs(output_dir, exist_ok=True)


plt.savefig(os.path.join(output_dir, "SNR_Distribution.png"), dpi=300)
# plt.show()

# -----------------------------
# Train Random Forest model
# -----------------------------
rf_results = train_random_forest.train_random_forest(data_file, n_trials=30)
mlp_results = train_mlp.train_mlp(data_file, n_trials=30)

# compare performance
print("RF F1 Macro:", f1_score(rf_results['y_test'], rf_results['y_pred'], average='macro'))
print("MLP F1 Macro:", f1_score(mlp_results['y_test'], mlp_results['y_pred'], average='macro'))

