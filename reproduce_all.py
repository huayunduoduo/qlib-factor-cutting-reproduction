"""Run the complete data-supported reproduction; use --workers 1 on low-RAM PCs."""
import argparse,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(*args):
    print('RUN',*args,flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
    if not (ROOT/'data/research_2009_2025/calendars/day.txt').exists():raise FileNotFoundError('Extract the supplied data/research_2009_2025 directory first')
    run('verify_data.py')
    run('-m','unittest','discover','-s','tests')
    run('test_timing.py')
    run('prepare_verified_inputs.py')
    run('run_raw_factors.py')
    for out,extra in [('results_2015_2025',[]),('results_with_reversal',['--wind','verified_wind_join'])]:
        run('run_research.py','--model','lightgbm','--features','alpha158','--start','2015-01-01','--end','2025-12-31','--out',out,*extra)
    run('run_neural_grid.py','--workers',str(a.workers))
    run('evaluate_replication.py')
    run('run_stacking.py')
    run('enrich_metrics.py')
    run('finalize_statistics.py')
    run('verify_current.py')
    run('build_reproduction_report.py')
