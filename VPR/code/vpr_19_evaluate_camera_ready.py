"""Canonical camera-ready VPR evaluation from frozen features and geographic split.

Float64 local geographic coordinates, inclusive <=25 m positives, exact inner
product retrieval, identical query denominators across all reported configurations.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import faiss
import numpy as np
from scipy.spatial import cKDTree
from vpr_geo_protocol import local_xy

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def csvrows(path):
    with path.open() as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cpu',action='store_true')
    a=ap.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    root=a.root;fr=root/'vpr_frames';features=root/'VPR/features'
    rows=csvrows(fr/'geosplit_outdoor.csv');db=[r for r in rows if r['split']=='new_database'];qs=[r for r in rows if r['split']=='new_query']
    dbxy,qxy=local_xy(db,qs);nearest=cKDTree(dbxy).query(qxy,k=1)[0];answerable=nearest<=25.0
    metas=[csvrows(fr/f'{s}_metadata.csv') for s in ['database','query']]
    loc={r['frame_name']:(origin,i) for origin,rs in enumerate(metas) for i,r in enumerate(rs)}
    configs=[]
    def config(model,stream,size,dbfile,qfile,role):configs.append(dict(key=f'{model}_{stream}_{size}'+('_course' if role=='course' else ''),model=model,stream=stream,input=size,db_feature=dbfile,q_feature=qfile,role=role))
    for stream in ['hd','preview']:
        for model,size in [('cosplace','224x448'),('netvlad','224x448'),('eigenplaces','224x448'),('mixvpr','320x320')]:
            if model=='cosplace':fmt='cosplace_features_{split}_'+stream+'.npy'
            elif model=='netvlad':fmt='netvlad_ablation_{split}_'+stream+'_'+size+'.npy'
            else:fmt=model+'_official_features_{split}_'+stream+'_'+size+'.npy'
            config(model,stream,size,fmt.format(split='database'),fmt.format(split='query'),'main')
    for model in ['netvlad','eigenplaces']:
        for size in ['256x512','384x768','512x1024']:
            if model=='netvlad':fmt='netvlad_features_{split}_preview.npy' if size=='256x512' else 'netvlad_ablation_{split}_preview_'+size+'.npy'
            else:fmt='eigenplaces_official_features_{split}_preview_'+size+'.npy'
            config(model,'preview',size,fmt.format(split='database'),fmt.format(split='query'),'resolution')
    config('cosplace','hd','224x448','cosplace_features_database_hd_aligned.npy','cosplace_features_query_hd_aligned.npy','course')
    gpu=not a.cpu and faiss.get_num_gpus()>0
    faiss.omp_set_num_threads(4)
    resources=faiss.StandardGpuResources() if gpu else None
    if resources is not None:resources.setTempMemory(256<<20)
    results={'schema':2,'protocol':'geo-split-outdoor-float64-inclusive25','threshold_m':25.0,
        'comparison':'distance <= threshold; local float64 coordinates, same 111000 m/degree approximation',
        'retrieval':'FAISS GpuIndexFlatIP float32' if gpu else 'FAISS IndexFlatIP float32',
        'split_sha256':sha(fr/'geosplit_outdoor.csv'),'database_frames':len(db),'total_query':len(qs),
        'answerable':int(answerable.sum()),'unanswerable':int((~answerable).sum()),'runs':{}}
    topdir=a.out.with_suffix('');topdir.mkdir(exist_ok=True)
    np.savez(topdir/'geography.npz',database_xy=dbxy,query_xy=qxy,answerable=answerable,
             database_names=np.array([r['frame_name'] for r in db]),query_names=np.array([r['frame_name'] for r in qs]))
    for cfg in configs:
        paths=[features/cfg['db_feature'],features/cfg['q_feature']]
        caches=[np.load(p,mmap_mode='r') for p in paths]
        for m,c in zip(metas,caches):
            if len(m)!=len(c):raise ValueError(f'Feature row count does not match metadata: {paths}')
        def gather(rs):return np.array([caches[loc[r['frame_name']][0]][loc[r['frame_name']][1]] for r in rs],dtype='float32')
        dv,qv=gather(db),gather(qs)
        if not np.isfinite(dv).all() or not np.isfinite(qv).all():raise ValueError(f'Nonfinite features: {cfg}')
        dim=dv.shape[1];index=faiss.GpuIndexFlatIP(resources,dim) if gpu else faiss.IndexFlatIP(dim);index.add(dv)
        top=np.concatenate([index.search(qv[i:i+512],20)[1] for i in range(0,len(qv),512)])
        dist=np.linalg.norm(dbxy[top]-qxy[:,None,:],axis=2)
        hit={str(k):int((dist[:,:k]<=25).any(1).sum()) for k in [1,5,10,20]}
        r={**cfg,'descriptor_dim':dim,'feature_sha256':[sha(p) for p in paths],
           'hit_counts':hit,'recalls':{f'Recall@{k}':v/int(answerable.sum())*100 for k,v in hit.items()},
           'all_query_recalls':{f'Recall@{k}':v/len(qs)*100 for k,v in hit.items()}}
        results['runs'][cfg['key']]=r
        np.save(topdir/(cfg['key']+'_top20.npy'),top)
        a.out.write_text(json.dumps(results,indent=2)+'\n')
        print(cfg['key'], 'answerable',int(answerable.sum()), 'recalls',r['recalls'],flush=True)
        del index,dv,qv,caches
    print('Written',a.out,flush=True)
if __name__=='__main__':main()
