"""Shared geographic metric: float64 metres and an inclusive 25 m threshold.

Feature vectors remain float32; geographic coordinates must not be quantized to
float32 at the roughly 10^7 m absolute-coordinate scale.
"""
import numpy as np

def to_xy(lats, lons, ref):
    lat=np.asarray(lats,dtype=np.float64)
    lon=np.asarray(lons,dtype=np.float64)
    return np.column_stack((lon*111000.0*np.cos(np.radians(ref)),lat*111000.0))

def local_xy(db_rows, query_rows):
    ref=float(np.mean([float(r['lat']) for r in db_rows]))
    db=to_xy([float(r['lat']) for r in db_rows],[float(r['lon']) for r in db_rows],ref)
    query=to_xy([float(r['lat']) for r in query_rows],[float(r['lon']) for r in query_rows],ref)
    origin=db.mean(axis=0)
    return db-origin,query-origin
