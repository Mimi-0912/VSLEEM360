"""Image-derived diagnostic on the exact saved VSR clips (not clock ground truth)."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from check_alignment import compare

def quant(xs):
    a=np.asarray(xs,dtype=float)
    return {k:float(np.percentile(a,q)) for k,q in [('min',0),('median',50),('p95',95),('max',100)]} if len(a) else {}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clips',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();cv2.setNumThreads(1)
    manifest=a.clips/'clips_manifest.jsonl'
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    a.output.mkdir(parents=True,exist_ok=True)
    def read(path):
        raw=path.read_bytes();im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if im is None:raise ValueError(f'Cannot decode {path}')
        im=cv2.cvtColor(cv2.resize(im,(480,240),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        return im,hashlib.sha256(raw).hexdigest()
    def check(r):
        folder=a.clips/r['seq'];center=r.get('center_offset',7)
        gt,gtsha=read(folder/'gt'/f'{center:04d}.png');cs=[];hashes={}
        for i in range(r['window']):
            lr,h=read(folder/'lr'/f'{i:04d}.png');hashes[f'{i:04d}.png']=h
            cs.append({'offset_frames':i-center,**compare(gt,lr)})
        best=max(cs,key=lambda c:c['response'])
        return {'id':r['id'],'clip_id':r.get('clip_id',r['id']),'scene':r['scene'],
            'segment':r.get('segment'),'hd_index':r['hd_index'],'center_index':r['center_index'],
            'pre_correction_pairing_delta_ms':r['pts_delta_ms'],
            'center':cs[center],'best':best,'best_at_search_boundary':best['offset_frames'] in [-center,r['window']-1-center],
            'best_response_ge_0_8':best['response']>=0.8,'candidates':cs,
            'image_sha256':{'gt':gtsha,'lr':hashes}}
    records=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool, (a.output/'records.jsonl').open('w') as log:
        for r in pool.map(check,rows):
            records.append(r);log.write(json.dumps(r)+'\n');log.flush()
            if len(records)%10==0 or len(records)==len(rows):
                print(f'{len(records)}/{len(rows)} checked; latest {r["clip_id"]} best={r["best"]["offset_frames"]:+d}, response={r["best"]["response"]:.3f}',flush=True)
    good=[r for r in records if r['best_response_ge_0_8'] and not r['best_at_search_boundary']]
    summary={'schema':1,'manifest':str(manifest.resolve()),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'diagnostic_only':True,'audit_resolution':[480,240],'n_clips':len(records),
        'n_sequences':len({r['scene'] for r in records}),'segment_counts':dict(Counter(r['segment'] for r in records)),
        'best_offset_counts':dict(Counter(str(r['best']['offset_frames']) for r in records)),
        'within_one_frame_n':sum(abs(r['best']['offset_frames'])<=1 for r in records),
        'boundary_wins':sum(r['best_at_search_boundary'] for r in records),
        'response_ge_0_8_n':sum(r['best_response_ge_0_8'] for r in records),'qualified_n':len(good),
        'pre_correction_pairing_delta_ms':quant([r['pre_correction_pairing_delta_ms'] for r in records]),
        'center_abs_dx_preview_px':quant([abs(r['center']['dx_audit_px'])*4 for r in records]),
        'center_abs_dy_preview_px':quant([abs(r['center']['dy_audit_px'])*4 for r in records]),
        'best_abs_dx_preview_px':quant([abs(r['best']['dx_audit_px'])*4 for r in records]),
        'best_abs_dy_preview_px':quant([abs(r['best']['dy_audit_px'])*4 for r in records]),
        'limitations':'Whole-image phase correlation after downsampling; argmax need not identify physical exposure offset, especially for static/ambiguous content; no calibration of clock error or local pixel registration.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
