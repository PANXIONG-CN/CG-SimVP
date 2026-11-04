"""Corrected SZA + REAL QD / subsolar-relative MLT; same loader function signatures.
Requires JPLD_GRID_NPZ and (for QD) JPLD_GEOMETRY_CACHE. Never falls back to legacy proxy.
"""
import calendar
import datetime
import json
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
from scenario_core import load_grid,digest

FEATURE_VERSION='noaa_sza_apex_qd_mlt_v2'


@lru_cache(maxsize=4)
def _grid(path):
    lat,lon,_=load_grid(path)
    return lat,lon,digest(path)


def coordinates(height,width):
    path=os.environ.get('JPLD_GRID_NPZ')
    if not path:raise ValueError('Set JPLD_GRID_NPZ to verified actual TEC grid')
    la,lo,sha=_grid(path)
    if (len(la),len(lo))!=(height,width):raise ValueError('Actual TEC / physical feature grid mismatch')
    return la,lo,sha


def frame_to_datetime(year,frame_idx,frames_per_day=96):
    if frames_per_day<=0 or 86400%frames_per_day:raise ValueError('Invalid frames_per_day')
    return datetime.datetime(year,1,1)+datetime.timedelta(seconds=frame_idx*86400/frames_per_day)


def compute_sza_sequence(year,start_frame,num_frames,height=180,width=360,frames_per_day=96):
    la,lo,_=coordinates(height,width)
    if start_frame<0 or num_frames<1 or frames_per_day<=0:raise ValueError('Invalid SZA sequence')
    times=[frame_to_datetime(year,start_frame+i,frames_per_day) for i in range(num_frames)]
    doy=np.array([t.timetuple().tm_yday for t in times],dtype=float)
    hour=np.array([t.hour+t.minute/60+t.second/3600 for t in times])
    days=np.array([366 if __import__('calendar').isleap(t.year) else 365 for t in times])
    g=2*np.pi/days*(doy-1+(hour-12)/24)
    eq=229.18*(.000075+.001868*np.cos(g)-.032077*np.sin(g)-.014615*np.cos(2*g)-.040849*np.sin(2*g))
    dec=(.006918-.399912*np.cos(g)+.070257*np.sin(g)-.006758*np.cos(2*g)
         +.000907*np.sin(2*g)-.002697*np.cos(3*g)+.00148*np.sin(3*g))
    ha=np.deg2rad(15*(hour[:,None]+lo[None,:]/15+eq[:,None]/60-12))
    lat=np.deg2rad(la)
    cosz=(np.sin(lat)[None,:,None]*np.sin(dec)[:,None,None]
          +np.cos(lat)[None,:,None]*np.cos(dec)[:,None,None]*np.cos(ha)[:,None,:])
    return np.clip(cosz,-1,1).astype('f4')


def compute_sza_grid(dt,height=180,width=360):
    if dt.tzinfo is not None:dt=dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    # Accept arbitrary UTC seconds rather than rounding to a 15-minute tick.
    from scenario_core import solar_geometry
    la,lo,_=coordinates(height,width)
    angle,_,_=solar_geometry(dt,la,lo)
    return np.cos(np.deg2rad(angle)).astype('f4')


@lru_cache(maxsize=12)
def _year_cache(root,year,grid_sha,fpd):
    base=Path(root)/str(year)
    meta=json.loads((base/'metadata.json').read_text(encoding='utf-8'))
    if meta['feature_version']!=FEATURE_VERSION or meta['grid_sha256']!=grid_sha or meta['frames_per_day']!=fpd:
        raise ValueError('QD cache version / grid / cadence mismatch; regenerate explicit cache')
    for name in ('qd_lat.npy', 'qd_lon.npy', 'mlt0_hours.npy'):
        if meta.get('hashes', {}).get(name) != digest(base/name):
            raise ValueError(f'Geometry cache hash mismatch: {base/name}')
    qlat = np.load(base/'qd_lat.npy', mmap_mode='r', allow_pickle=False)
    qlon = np.load(base/'qd_lon.npy', mmap_mode='r', allow_pickle=False)
    mlt0 = np.load(base/'mlt0_hours.npy', mmap_mode='r', allow_pickle=False)
    expected_frames = (366 if calendar.isleap(year) else 365) * fpd
    if qlat.ndim != 2 or qlon.shape != qlat.shape or mlt0.shape != (expected_frames,):
        raise ValueError('Invalid geometry cache dimensions or annual time coverage')
    if not all(np.isfinite(a).all() for a in (qlat, qlon, mlt0)):
        raise ValueError('Nonfinite geometry cache values')
    return qlat, qlon, mlt0, meta


def configure_geometry(grid_npz, geometry_cache=None, years=(), frames_per_day=96,
                       use_qd=False):
    """Validate the selected feature assets and return checkpoint provenance."""
    grid_path = str(Path(grid_npz).resolve())
    lat, lon, grid_sha = _grid(grid_path)
    if frames_per_day <= 0 or 86400 % frames_per_day:
        raise ValueError('frames_per_day must divide 86400 seconds')
    provenance = {
        'feature_version': FEATURE_VERSION,
        'grid_sha256': grid_sha,
        'grid_shape': [len(lat), len(lon)],
        'frames_per_day': frames_per_day,
        'cache_by_year': {},
    }
    if use_qd:
        if not geometry_cache:
            raise ValueError('geometry_cache is required for QD/MLT inputs')
        cache_root = str(Path(geometry_cache).resolve())
        for year in years:
            qlat, _, _, meta = _year_cache(cache_root, int(year), grid_sha, frames_per_day)
            if qlat.shape != (len(lat), len(lon)):
                raise ValueError(f'Geometry grid shape mismatch in {year}')
            provenance['cache_by_year'][str(year)] = meta
        os.environ['JPLD_GEOMETRY_CACHE'] = cache_root
    os.environ['JPLD_GRID_NPZ'] = grid_path
    return provenance


def compute_qd_sequence(year,start_frame,num_frames,height=180,width=360,frames_per_day=96):
    _,_,sha=coordinates(height,width)
    root=os.environ.get('JPLD_GEOMETRY_CACHE')
    if not root:raise ValueError('Set JPLD_GEOMETRY_CACHE; run prepare_geometry.py first')
    qlat,qlon,mlt0,meta=_year_cache(root,year,sha,frames_per_day)
    if qlat.shape!=(height,width) or qlon.shape!=qlat.shape:raise ValueError('QD cache shape mismatch')
    if num_frames<1 or start_frame<0 or start_frame+num_frames>len(mlt0):raise ValueError('QD time outside cached year')
    # ApexPy MLT is affine in magnetic longitude at fixed UTC. mlt0 was calculated
    # by Apex.mlon2mlt(0,UTC), including subsolar point conversion, not UTC phase.
    mlt=(qlon[None,:,:]/15+mlt0[start_frame:start_frame+num_frames,None,None])%24
    phase=2*np.pi*mlt/24
    slat=np.broadcast_to(np.sin(np.deg2rad(qlat)),(num_frames,height,width))
    return np.stack([slat,np.sin(phase),np.cos(phase)],axis=1).astype('f4')
