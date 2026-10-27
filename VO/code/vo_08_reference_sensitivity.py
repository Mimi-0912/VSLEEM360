"""Reproduce VO reference-speed sensitivity using saved per-sequence metrics.

Filtering is a diagnostic; it does not certify the retained reference accuracy.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--speed-threshold',type=float,default=20)
    a=p.parse_args();rows=list(csv.DictReader(a.metrics.open()));scenes=sorted({r['scene'] for r in rows});diagnostics=[]
    for s in scenes:
        z=np.genfromtxt(a.data_root/s/f'{s}_trajectory.csv',delimiter=',',names=True)
        xyz=np.column_stack([z[k]/1000 for k in ['x','y','z']]);dt=np.diff(z['time']);good=dt>0
        speed=np.linalg.norm(np.diff(xyz,axis=0),axis=1)[good]/dt[good]
        diagnostics.append({'scene':s,'intervals_over_threshold':int((speed>a.speed_threshold).sum()),'max_speed_mps':float(speed.max())})
    flags={r['scene'] for r in diagnostics if r['intervals_over_threshold']}
    groups={m:{r['scene']:r for r in rows if r['method']==m and r['status']=='ok'} for m in ['orbslam3','droid','360dvo']}
    common=set.intersection(*(set(g) for g in groups.values()));result={}
    for label,ids in [('common',common),('without_speed_flags',common-flags)]:
        result[label]={'n':len(ids),'scene_ids':sorted(ids),'methods':{}}
        for m,g in groups.items():result[label]['methods'][m]={k:float(np.median([float(g[s][k]) for s in ids])) for k in ['ate_rmse_m','rpe_pos_time1s_m']}
        mm=result[label]['methods'];result[label]['360dvo_vs_droid_ate_reduction_percent']=100*(1-mm['360dvo']['ate_rmse_m']/mm['droid']['ate_rmse_m'])
    report={'schema':1,'speed_threshold_mps':a.speed_threshold,'diagnostic_only':True,'screened_sequence_count':len(flags),'diagnostics':diagnostics,'subsets':result}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print({k:(v['n'],v['360dvo_vs_droid_ate_reduction_percent']) for k,v in result.items()})
if __name__=='__main__':main()
