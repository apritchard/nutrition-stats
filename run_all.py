"""
run_all.py — Run the full pipeline in order.

  pip install -r requirements.txt
  python run_all.py

Steps:
  1. aggregate.py  → daily_unified.csv
  2. tdee.py       → tdee_results.csv
  3. visualize.py  → chart_weight.html, chart_tdee.html, chart_dashboard.html
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = ['aggregate.py', 'tdee.py', 'visualize.py', 'visualize_extra.py']

for script in SCRIPTS:
    path = Path(__file__).parent / script
    print(f'\n{"="*50}')
    print(f'Running {script}...')
    print('='*50)
    result = subprocess.run([sys.executable, str(path)], capture_output=False)
    if result.returncode != 0:
        print(f'\nERROR: {script} failed (exit code {result.returncode}). Stopping.')
        sys.exit(result.returncode)

print('\nAll done. Open charts/chart_full_dashboard.html to view results.')
