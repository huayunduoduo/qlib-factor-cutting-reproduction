"""Package code, frozen inputs, trained models and auditable results; omit caches."""
import json,hashlib,zipfile
from pathlib import Path
from research.data import ROOT
verify=json.loads((ROOT/'verification_current.json').read_text())
assert verify['completed_neural_models']==44
assert len(verify['portfolio_checks'])>=21
required=['results_2015_2025','results_with_reversal','results_raw_factors','results_replication','results_stacking_2021_2025']
for d in required:assert (ROOT/d/'summary.csv').exists()
scripts=['run_research.py','train_neural.py','run_neural_grid.py','prepare_verified_inputs.py','run_raw_factors.py','evaluate_replication.py','run_stacking.py','verify_current.py','verify_data.py','enrich_metrics.py','reprice_saved_predictions.py','finalize_statistics.py','build_reproduction_report.py','reproduce_all.py','package_deliverable.py','test_timing.py','import_counts.py','import_wind_provider.py','download_data.py','prepare_data.py','make_snapshot.py']
files=[ROOT/s for s in scripts]
for s in ['README.md','README_CURRENT.md','requirements.txt','requirements_reproduction.txt','environment_current.txt','replication_matrix.csv','pending_data.csv','planned_folds_2015_2025.csv','coverage_2009_2025.csv','amount_audit.json','model_parameters.json','bootstrap_sensitivity.csv','verification_current.json','data_verification.log','tests_final.log','tests_execution.log','reproduction_report.html','QLIB_LICENSE']:
    files.append(ROOT/s)
for d in ['research','tests','configs','data/research_2009_2025','wind_normalized','verified_wind_join','results_neural_grid',*required]:
    for p in (ROOT/d).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='factor_panel.parquet':continue
        if p.suffix=='.pyc' or p.name.endswith('.pending.gz'):continue
        files.append(p)
for s in ['source_manifest.json','source_audit.json','initial_count_audit.json','run_research_before_memory_crop.py']:
    files.append(ROOT/'provenance'/s)
files=sorted(set(files),key=lambda p:str(p.relative_to(ROOT)))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(ROOT/'DELIVERY_SHA256.json').write_text(json.dumps(manifest,indent=2))
target=ROOT.parent/'qlib_paper_reproduction_2009_2025.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for p in files+[ROOT/'DELIVERY_SHA256.json']:z.write(p,Path('qlib_paper_reproduction')/p.relative_to(ROOT))
with zipfile.ZipFile(target) as z:assert z.testzip() is None
print(json.dumps({'file':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':len(files)+1}))
