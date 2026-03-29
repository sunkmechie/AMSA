import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

scripts = [
    "challenge1_triangle_area.py",
    "challenge2_orientation_batch.py",
    "challenge3_signed_volume.py",
    "challenge4_product_decomposition.py",
    "challenge5_sparse_support.py",
    "challenge6_even_odd_decomposition.py",
]

for s in scripts:

    print("\n==============================")
    print("Running", s)
    print("==============================")

    subprocess.run([sys.executable, str(ROOT/s)], check=True)
