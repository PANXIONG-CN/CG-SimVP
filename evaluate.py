"""Evaluate the packaged backbone variants with UTC-aligned, pooled metrics.
Preflight checks the configured data, normalization and geometry assets.
"""
import argparse
import json
import sys
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scenario_core import ScenarioEvaluator, load_grid, timeline, digest


def resolve_config(path):
    path=Path(path).resolve()
    c=json.loads(path.read_text(encoding='utf-8'))
    c.setdefault('project_dir', str(Path(__file__).resolve().parent))
    c.setdefault('data_root', 'data')
    c.setdefault('grid_npz', str(Path(c['data_root'])/'grid.npz'))
    c.setdefault('output_dir', 'results/evaluation')
    for k in ['project_dir','data_root','checkpoint','training_metrics','grid_npz','timestamps',
              'climatology_path','activity_csv','events_csv','output_dir','grid_provenance','geometry_cache','export_predictions']:
        if c.get(k):
            p=Path(c[k]); c[k]=str(p if p.is_absolute() else (path.parent/p).resolve())
    required=['project_dir','data_root','grid_npz','year','frames_per_day','sample_stride','output_dir']
    for k in required:
        if k not in c:
            raise ValueError(f'Missing config key {k}')
    if c['sample_stride'] <= 0:
        raise ValueError('sample_stride must be positive')
    return c


def prepare(c):
    root=Path(c['data_root'])
    paths=sorted((root/f"{c['year']}_npy").glob('*.npy'))
    if not paths:
        raise FileNotFoundError(f"No annual NPY files under {root}")
    lat,lon,qd=load_grid(c['grid_npz'])
    if c.get('eia_coordinate')=='qd' and qd is None and c.get('geometry_cache'):
        cache=Path(c['geometry_cache'])/str(c['year'])
        meta=json.loads((cache/'metadata.json').read_text())
        if meta['grid_sha256']!=digest(c['grid_npz']) or meta['frames_per_day']!=c['frames_per_day']:
            raise ValueError('Evaluation QD cache does not match verified grid/cadence')
        qd=np.load(cache/'qd_lat.npy',allow_pickle=False)
        if qd.shape!=(len(lat),len(lon)) or not np.isfinite(qd).all():raise ValueError('Invalid evaluation QD grid')
        if digest(cache/'qd_lat.npy')!=meta['hashes']['qd_lat.npy']:raise ValueError('Evaluation QD hash mismatch')
    if not c.get('grid_provenance') or not Path(c['grid_provenance']).is_file():
        raise ValueError('grid_provenance must point to a text/JSON record of the actual TEC grid and how it was verified')
    records=[]; total=0
    for p in paths:
        a=np.load(p,mmap_mode='r',allow_pickle=False)
        if a.ndim!=3 or a.shape[1:]!=(len(lat),len(lon)) or a.shape[0]==0:
            raise ValueError(f'{p}: expected nonempty [frames,{len(lat)},{len(lon)}], got {a.shape}')
        if a.dtype.kind not in 'fiu':
            raise ValueError('TEC array must be numeric')
        records.append(dict(path=str(p),shape=list(a.shape),frame_offset=total,size=p.stat().st_size))
        total+=len(a)
    ts=timeline(c,total)
    stats=root/'global_max.txt'
    if not stats.exists():
        raise FileNotFoundError('global_max.txt is required; refusing to recompute a different normalization at evaluation time')
    mx=float(stats.read_text().strip())
    if not np.isfinite(mx) or mx<=0:
        raise ValueError('Invalid cached global_max')
    norm_warning=None
    if (root/'norm_stats.npy').exists():
        z=np.load(root/'norm_stats.npy',allow_pickle=False).reshape(-1)
        if z.shape!=(2,) or not np.allclose(z,[0.,mx],rtol=1e-6,atol=1e-6):
            norm_warning=f'norm_stats.npy={z.tolist()} disagrees with loader min/max=[0,{mx}]. Original evaluation metrics must be recomputed.'
            print('WARNING:',norm_warning)
    training = None
    model_cfg={}
    if c.get('training_metrics'):
        training=json.loads(Path(c['training_metrics']).read_text(encoding='utf-8'))
        model_cfg=training['config']
        normalization = training.get('normalization')
        if normalization and not np.allclose(
            [normalization['min'], normalization['max']], [0., mx], rtol=1e-6, atol=1e-6
        ):
            raise ValueError('Training/evaluation normalization mismatch')
    else:
        model_cfg=c.get('model',{})
    for key in ['time_step','predict_step','hidden_dim','n_layers','attn_name',
                'use_climatology','use_aux_sza','use_aux_qd','frames_per_day']:
        if key not in model_cfg:
            raise ValueError(f'Model config requires {key}; use train_metrics_SimVP_SUFFIX.json instead of guessing')
    if int(model_cfg['frames_per_day']) != int(c['frames_per_day']):
        raise ValueError('Training/evaluation frames_per_day mismatch')
    geometry = None
    if model_cfg['use_aux_sza'] or model_cfg['use_aux_qd']:
        from solar_zenith import configure_geometry
        geometry = configure_geometry(
            c['grid_npz'], c.get('geometry_cache'), years=[c['year']],
            frames_per_day=c['frames_per_day'], use_qd=bool(model_cfg['use_aux_qd']),
        )
        if c.get('require_corrected_geometry', True):
            recorded = (training or {}).get('geometry') or {}
            for key in ('feature_version', 'grid_sha256'):
                if recorded.get(key) != geometry[key]:
                    raise ValueError(
                        'Training metadata must record matching geometry version/grid. '
                        'Use the metadata associated with the selected checkpoint.'
                    )
            if recorded.get('frames_per_day', c['frames_per_day']) != c['frames_per_day']:
                raise ValueError('Training/evaluation geometry cadence mismatch')
            if model_cfg['use_aux_qd']:
                test_height = geometry['cache_by_year'][str(c['year'])]['height_km']
                train_heights = {
                    meta['height_km'] for meta in recorded.get('cache_by_year', {}).values()
                }
                if train_heights != {test_height}:
                    raise ValueError('QD height differs from recorded training geometry')
    tin,tout=int(model_cfg['time_step']),int(model_cfg['predict_step'])
    if len(lat)%2 or len(lon)%2:
        raise ValueError('Provided SimVP uses stride-2 encoding/decoding and requires even spatial dimensions')
    if min(tin,tout)<=0 or c['sample_stride']>tout:
        raise ValueError('Positive horizons and sample_stride<=predict_step required to prevent target gaps')
    if total <= tin+tout:
        raise ValueError('Dataset too short (also triggers original loader exact-window edge bug)')
    if c.get('eia_every_n_frames',4)<1 or c.get('eia_lon_stride',4)<1:
        raise ValueError('EIA sampling strides must be positive')
    if c.get('use_climatology') is not None:
        raise ValueError('Set climatology feature in training metadata/model, not two competing locations')
    clim_path=c.get('climatology_path') or str(root/'climatology.npy')
    if model_cfg['use_climatology'] or c.get('predictor')=='climatology_unscaled':
        a=np.load(clim_path,mmap_mode='r',allow_pickle=False)
        if a.shape!=(12,c['frames_per_day'],len(lat),len(lon)):
            raise ValueError('climatology.npy shape disagrees with dataset')
        if c.get('climatology_train_years') != [2012,2014,2016,2018,2020]:
            raise ValueError('Record climatology_train_years=[2012,2014,2016,2018,2020] after checking generation provenance')
    n=(total-tin-tout)//c['sample_stride']+1
    limit=c.get('max_samples')
    if limit is not None:
        if int(limit)<=0: raise ValueError('max_samples must be positive')
        n=min(n,int(limit))
    manifest=dict(config=c,model_config=model_cfg,source_files=records,total_frames=total,
        normalization=dict(min=0.,max=mx,source='dataloader_clim_aux actual normalization'),
        geometry=geometry,
        norm_stats_warning=norm_warning,grid_sha256=digest(c['grid_npz']),
        grid_provenance_sha256=digest(c['grid_provenance']),
        chronology_sha256=__import__('hashlib').sha256(
            np.asarray(ts, dtype='datetime64[ns]').astype(np.int64).tobytes()
        ).hexdigest(),
        samples=n,first_target=str(ts[tin]),
        last_target=str(ts[(n-1)*c['sample_stride']+tin+tout-1]),
        lead_definition='(h+1)*cadence after final observed frame; target_index=sample*stride+T_in+h',
        sample_weighting='forecast-target-lead pairs; overlapping targets occur at DIFFERENT leads',
        eia_coordinate=c.get('eia_coordinate','disabled'),
        feature_policy='Inference must match recorded training geometry; scene solar geometry uses independent NOAA calculation',
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__))
    for key in ['activity_csv','events_csv','climatology_path','training_metrics']:
        if c.get(key): manifest[key+'_sha256']=digest(c[key])
    if model_cfg['use_climatology'] or c.get('predictor')=='climatology_unscaled':
        manifest['climatology_path_sha256']=digest(clim_path)
    if qd is not None:
        manifest['evaluation_qd_sha256']=__import__('hashlib').sha256(np.ascontiguousarray(qd,dtype='<f8').tobytes()).hexdigest()
    # Hash local authoring code and original inference dependencies for reproducibility.
    manifest['code_sha256']={}
    for p in [Path(__file__),Path(__file__).with_name('scenario_core.py')]+[
        Path(c['project_dir'])/x for x in ['dataloader_clim_aux.py','simvp.py','attentions.py','solar_zenith.py']]:
        if p.exists(): manifest['code_sha256'][str(p)]=digest(p)
    return lat,lon,qd,ts,model_cfg,mx,clim_path,manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    ap.add_argument('--preflight',action='store_true')
    args=ap.parse_args()
    c=resolve_config(args.config)
    lat,lon,qd,ts,mc,mx,cp,manifest=prepare(c)
    # Validate activity and EIA configuration during preflight, too.
    ev=ScenarioEvaluator(c,lat,lon,qd)
    out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    print(json.dumps({k:manifest[k] for k in ['samples','first_target','last_target','normalization']},indent=2))
    if args.preflight:
        (out/'preflight.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        print('Preflight passed; file shape/count checks do not prove correct original chronology or grid.')
        return
    if (out/'manifest.json').exists():
        raise FileExistsError('Completed output already exists; use a new output_dir to preserve results')
    sys.path.insert(0,c['project_dir'])
    import torch
    from torch.utils.data import DataLoader
    from dataloader_clim_aux import TEC_Dataset
    from simvp import SimVP
    device=torch.device(c.get('device','cuda:0' if torch.cuda.is_available() else 'cpu'))
    ds=TEC_Dataset(c['data_root'],time_step=mc['time_step'],predict_step=mc['predict_step'],
        test=True,test_year=c['year'],sample_stride=c['sample_stride'],max_samples=c.get('max_samples'),
        climatology_path=cp if mc['use_climatology'] else None,frames_per_day=c['frames_per_day'],
        use_aux_sza=bool(mc['use_aux_sza']),use_aux_qd=bool(mc['use_aux_qd']))
    if len(ds)!=manifest['samples'] or ds.global_min!=0 or not np.isclose(ds.global_max,mx):
        raise ValueError('Actual loader differs from audited metadata')
    predictor=c.get('predictor','simvp')
    if predictor not in ('simvp','persistence','climatology_scaled','climatology_unscaled'):
        raise ValueError('predictor must be simvp/persistence/climatology_scaled/climatology_unscaled')
    if predictor=='climatology_scaled' and not mc['use_climatology']:
        raise ValueError('Scaled climatology requires the original residual loader and background')
    if predictor=='climatology_unscaled':
        unclim=np.load(cp,mmap_mode='r',allow_pickle=False)
    if predictor=='simvp':
        net=SimVP(input_dim=1+int(mc['use_aux_sza'])+3*int(mc['use_aux_qd']),
            hidden_dim=mc['hidden_dim'],output_dim=1,time_step=mc['time_step'],predict_step=mc['predict_step'],
            n_layers=mc['n_layers'],attn_name=mc['attn_name']).to(device)
        ckpt=Path(c['checkpoint'])
        state=torch.load(ckpt,map_location='cpu',weights_only=True)
        if 'state_dict' in state: state=state['state_dict']
        state={(k[7:] if k.startswith('module.') else k):v for k,v in state.items()}
        net.load_state_dict(state,strict=True);net.eval()
        manifest['checkpoint_sha256']=digest(ckpt)
    manifest['versions']['torch']=torch.__version__
    loader=DataLoader(ds,batch_size=int(c.get('batch_size',1)),shuffle=False,
        num_workers=int(c.get('num_workers',0)),pin_memory=device.type=='cuda',drop_last=False)
    cursor=0
    export=None
    if c.get('export_predictions'):
        ep=Path(c['export_predictions']);ep.parent.mkdir(parents=True,exist_ok=True)
        if ep.exists():raise FileExistsError(ep)
        export=np.lib.format.open_memmap(ep,mode='w+',dtype='f4',
            shape=(len(ds),mc['predict_step'],len(lat),len(lon)))
    with torch.inference_mode():
        for batch in loader:
            x,y=batch[0],batch[1]
            p=net(x.to(device)) if predictor=='simvp' else torch.zeros_like(y)
            if p.shape!=y.shape or p.ndim!=5 or p.shape[2]!=1:
                raise ValueError(f'Expected matching [B,T,1,H,W]; prediction {p.shape}, truth {y.shape}')
            p=np.asarray(p.detach().cpu().numpy()[:,:,0], dtype=np.float32)
            y=np.asarray(y.numpy()[:,:,0], dtype=np.float32)
            if mc['use_climatology']:
                background=np.asarray(batch[2].numpy()[:,:,0], dtype=np.float32)
                p=p+background; y=y+background
            # Use actual loader normalization; never norm_stats.npy and never clip residuals.
            p=p*mx;y=y*mx
            for b in range(len(p)):
                idx=cursor+b
                sf=idx*c['sample_stride']
                # Canonical targets ensure exact shared support across residual/direct models.
                raw=np.asarray(
                    ds._read_frames(c['year'],sf+mc['time_step'],mc['predict_step'])*mx,
                    dtype=np.float32,
                )
                if not np.allclose(
                    np.asarray(y[b], dtype=np.float32),
                    raw,
                    rtol=2e-5,
                    atol=2e-4,
                ):
                    raise ValueError('Residual/background inverse reconstruction failed against raw target')
                if predictor=='persistence':
                    last=ds._read_frames(c['year'],sf+mc['time_step']-1,1)[0]*mx
                    p[b]=last[None,:,:]
                elif predictor=='climatology_scaled':
                    p[b]=background[b]*mx
                elif predictor=='climatology_unscaled':
                    for h in range(mc['predict_step']):
                        k=sf+mc['time_step']+h
                        p[b,h]=unclim[
                            pd.Timestamp(str(ts[k])).month-1,
                            k%c['frames_per_day'],
                        ]
                for h in range(mc['predict_step']):
                    k=sf+mc['time_step']+h
                    ev.add(
                        pd.Timestamp(str(ts[k])),
                        (h+1)*1440/c['frames_per_day'],
                        p[b,h],
                        raw[h],
                    )
                if export is not None:export[idx]=p[b]
            cursor+=len(p)
            if cursor % max(1,int(c.get('progress_every_samples',20))) < len(p):
                print(f'{cursor}/{len(ds)} samples; {ev.n_frames} target/lead pairs',flush=True)
    if cursor!=len(ds): raise RuntimeError('Incomplete inference')
    manifest['canonical_target_sha256']=ev.truth_hash.hexdigest()
    if export is not None:
        export.flush()
        ids=np.arange(len(ds))[:,None]*c['sample_stride']+mc['time_step']+np.arange(mc['predict_step'])[None,:]
        tpath=Path(c['export_predictions']).with_suffix('.target_utc.npy')
        np.save(tpath, np.asarray(ts, dtype='datetime64[ns]')[ids])
        manifest['export_predictions']=dict(path=c['export_predictions'],target_times=str(tpath),units='TECU')
    ev.save(out,manifest)
    print(f'Done: {out}')


if __name__=='__main__':
    main()
