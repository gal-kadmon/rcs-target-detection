from oct2py import octave
import os

base_dir = os.path.dirname(__file__)
os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)

octave.addpath(os.path.join(base_dir, 'src'))

output_filename = os.path.join(base_dir, 'data', 'rcs_output.csv')

N = 1000
target_ratio = 0.3

try:
    octave.generate_rcs_data(output_filename, N, target_ratio, nout=0)
    print(f"CSV file created at: {output_filename}")

except Exception as e:
    print("Error calling Octave function:", e)
