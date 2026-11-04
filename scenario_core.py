"""Streaming, UTC-target-aligned TEC evaluation. No torch dependency."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def solar_geometry(time, lat, lon):
    """NOAA fractional-year approximation; geometric SZA, no refraction.
    LMST = UTC + east longitude / 15; LAST additionally includes equation of time.
    """
    t = pd.Timestamp(time)
    hour = t.hour + t.minute / 60 + t.second / 3600
    days = 366 if t.is_leap_year else 365
    g = 2 * np.pi / days * (t.dayofyear - 1 + (hour - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g)
                   - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))
    dec = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g)
           - 0.006758*np.cos(2*g) + 0.000907*np.sin(2*g)
           - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
    lmst = (hour + np.asarray(lon)/15) % 24
    last = (lmst + eq/60) % 24
    ha = np.deg2rad(15*(last - 12))
    la = np.deg2rad(np.asarray(lat))[:, None]
    c = np.sin(la)*np.sin(dec) + np.cos(la)*np.cos(dec)*np.cos(ha)[None, :]
    return np.rad2deg(np.arccos(np.clip(c, -1, 1))), lmst, last


def fixed_dipole_lat(lat, lon):
    """Legacy training proxy ONLY; deliberately not named quasi-dipole."""
    la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    pl, po = np.deg2rad(80.65), np.deg2rad(-72.68)
    return np.rad2deg(np.arcsin(np.clip(np.sin(la)*np.sin(pl)
        + np.cos(la)*np.cos(pl)*np.cos(lo-po), -1, 1)))


def load_grid(path):
    with np.load(path, allow_pickle=False) as z:
        lat, lon = z['lat'].astype(float), z['lon'].astype(float)
        qd = z['qd_lat'].astype(float) if 'qd_lat' in z else None
    if lat.ndim != 1 or lon.ndim != 1 or min(len(lat), len(lon)) < 3:
        raise ValueError('grid.npz requires one-dimensional lat/lon in ORIGINAL array order')
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or np.max(abs(lat)) > 90:
        raise ValueError('Invalid latitude/longitude')
    if not (np.all(np.diff(lat)>0) or np.all(np.diff(lat)<0)):
        raise ValueError('Latitude must be strictly monotonic')
    # Require a global, uniformly spaced cyclic longitude grid, without a repeated seam.
    circular = np.diff(np.r_[np.sort(lon % 360), np.sort(lon % 360)[0]+360])
    if not np.allclose(circular, 360/len(lon), atol=1e-5):
        raise ValueError('Longitude must cover globe uniformly without duplicate seam')
    if qd is not None and (qd.shape != (len(lat), len(lon)) or not np.isfinite(qd).all()):
        raise ValueError('Optional qd_lat must be a finite [H,W] grid in the same order')
    return lat, lon, qd


def timeline(cfg, total):
    year, fpd = cfg['year'], cfg['frames_per_day']
    if fpd <= 0 or 86400 % fpd:
        raise ValueError('frames_per_day must divide 86400 seconds')
    step_seconds = 86400 // fpd
    expected = np.datetime64(f'{year}-01-01T00:00:00', 'ns') + (
        np.arange(total, dtype='timedelta64[s]') * step_seconds
    )
    if cfg.get('timestamps'):
        a = np.load(cfg['timestamps'], allow_pickle=False)
        if a.dtype.kind not in 'UMS':
            raise ValueError('timestamps.npy must be datetime64 or ISO UTC strings, not numeric epochs')
        if a.dtype.kind == 'M':
            actual = a.astype('datetime64[ns]').astype(np.int64)
        else:
            actual = np.array(
                [pd.Timestamp(value, tz='UTC').value for value in a],
                dtype=np.int64,
            )
        if len(actual) != total or not np.array_equal(
            actual, expected.astype('datetime64[ns]').astype(np.int64)
        ):
            raise ValueError('Actual timestamps disagree with Jan-1 regular indexing used by loader; fix preprocessing, not evaluator')
    else:
        if not cfg.get('confirmed_full_year_regular_utc', False):
            raise ValueError('Provide timestamps.npy or explicitly confirm original full-year UTC chronology')
        days = (pd.Timestamp(f'{year+1}-01-01')-pd.Timestamp(f'{year}-01-01')).days
        if total != days*fpd:
            raise ValueError(f'Expected {days*fpd} full-year frames, found {total}')
    return expected


class Activity:
    """Explicit half-open observation intervals; no interpolation or future fill."""
    def __init__(self, path=None):
        self.df = None
        if path:
            d = pd.read_csv(path)
            for col in ['start_utc', 'end_utc']:
                d[col] = pd.to_datetime(d[col], utc=True)
            d = d.sort_values('start_utc').reset_index(drop=True)
            if len(d) == 0 or (d.end_utc <= d.start_utc).any():
                raise ValueError('Empty or invalid activity intervals')
            if (d.start_utc.iloc[1:].to_numpy() < d.end_utc.iloc[:-1].to_numpy()).any():
                raise ValueError('Activity intervals overlap')
            for key in ['kp', 'dst']:
                d[key] = pd.to_numeric(d[key], errors='raise')
            if ((d.kp.dropna()<0) | (d.kp.dropna()>9)).any() or (d.dst.abs()>5000).any():
                raise ValueError('Check Kp decimal units (not Kp*10), Dst units and missing-value sentinels')
            self.df = d

    def label(self, time):
        if self.df is None:
            return 'unknown'
        time = pd.Timestamp(time)
        if time.tzinfo is None:
            time = time.tz_localize('UTC')
        else:
            time = time.tz_convert('UTC')
        d = self.df
        i = d.start_utc.searchsorted(time, side='right')-1
        if i < 0 or time >= d.end_utc.iloc[i]:
            return 'unknown'
        kp, dst = d.kp.iloc[i], d.dst.iloc[i]
        # These combined thresholds are an analysis protocol, not an official scale.
        if (np.isfinite(kp) and kp >= 5) or (np.isfinite(dst) and dst <= -50):
            return 'storm'
        if np.isfinite(kp) and np.isfinite(dst) and kp <= 2 and abs(dst) <= 20:
            return 'quiet'
        return 'transition' if np.isfinite(kp) and np.isfinite(dst) else 'unknown'


class Events:
    def __init__(self, path=None):
        self.rows = []
        if path:
            d = pd.read_csv(path, dtype={'event_id': str, 'phase': str})
            for r in d.to_dict('records'):
                a = pd.Timestamp(r['start_utc'], tz='UTC')
                b = pd.Timestamp(r['end_utc'], tz='UTC')
                if a.tzinfo is None or b.tzinfo is None or b<=a:
                    raise ValueError('Event phases need explicit timezone and start < end')
                self.rows.append((a, b, r['event_id'], r['phase']))
            for i, (a,b,e,p) in enumerate(self.rows):
                if any(e==ee and a<bb and aa<b for aa,bb,ee,pp in self.rows[:i]):
                    raise ValueError('Phases overlap within an event')

    def labels(self, time):
        time = pd.Timestamp(time)
        if time.tzinfo is None:
            time = time.tz_localize('UTC')
        else:
            time = time.tz_convert('UTC')
        return [(e,p) for a,b,e,p in self.rows if a<=time<b]


class Accumulator:
    """Float64 sufficient statistics: W,SSE,SAE,bias,Y,Y2,MAPE numerator/weight,N."""
    def __init__(self, threshold=1.):
        self.data = defaultdict(lambda: np.zeros(9, dtype=np.float64))
        self.threshold = threshold

    def update(self, key, pred, true, mask=True, weights=1.):
        sel = np.broadcast_to(mask, true.shape)
        if not np.any(sel):
            return
        y, p = true[sel].astype(float), pred[sel].astype(float)
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('Nonfinite TEC: use a declared common valid-data mask before inference; do not silently drop model failures')
        w = np.broadcast_to(weights, true.shape)[sel].astype(float)
        err = p-y
        eligible = abs(y)>self.threshold
        self.data[key] += np.array([w.sum(), np.dot(w,err*err), np.dot(w,abs(err)),
            np.dot(w,err), np.dot(w,y), np.dot(w,y*y),
            np.sum(w[eligible]*abs(err[eligible])/abs(y[eligible])),
            w[eligible].sum(), len(y)])

    def rows(self):
        out = []
        for (year, lead, scene, group, weighting), a in sorted(self.data.items()):
            w,se,ae,b,y,y2,mp,mw,n = a
            sst = y2-y*y/w
            out.append(dict(year=year, lead_min=lead, scene=scene, group=group, weighting=weighting,
                weight=w, sse=se, sae=ae, n_pixels=int(n), rmse=np.sqrt(se/w), mae=ae/w,
                target_sum=y,target_sq_sum=y2,bias_sum=b,mape_sum=mp,
                bias=b/w, mape=100*mp/mw if mw>0 else np.nan, mape_weight=mw,
                r2=1-se/sst if sst>max(1e-12, abs(y2)*1e-12) else np.nan))
        return out


class ScenarioEvaluator:
    def __init__(self, cfg, lat, lon, qd=None):
        self.cfg, self.lat, self.lon = cfg, lat, lon
        self.acc = Accumulator(cfg.get('mape_threshold_tecu', 1.))
        self.activity, self.events = Activity(cfg.get('activity_csv')), Events(cfg.get('events_csv'))
        self.daily = defaultdict(lambda: np.zeros(3))
        self.maps = {}
        self.eia_rows = []
        self.frame_rows = []
        self.cases = {}
        self.case_times = {pd.Timestamp(t).tz_convert('UTC') for t in cfg.get('case_times_utc',[])}
        self.qd = qd
        if cfg.get('eia_coordinate') == 'fixed_dipole_proxy':
            self.qd = fixed_dipole_lat(lat, lon)
        elif cfg.get('eia_coordinate') == 'qd' and qd is None:
            raise ValueError('Real QD EIA requires independently supplied qd_lat and provenance')
        elif cfg.get('eia_coordinate', 'disabled') not in ('qd', 'disabled'):
            if cfg.get('eia_coordinate') != 'fixed_dipole_proxy':
                raise ValueError('Unknown eia_coordinate')
        self.truth_hash = hashlib.sha256()
        self.n_frames = 0

    def add(self, time, lead, pred, true):
        time = pd.Timestamp(time)
        if time.tzinfo is None:
            time = time.tz_localize('UTC')
        else:
            time = time.tz_convert('UTC')
        if pred.shape != (len(self.lat), len(self.lon)) or true.shape != pred.shape:
            raise ValueError('Expected one TEC map [H,W]')
        if not np.isfinite(true).all() or not np.isfinite(pred).all():
            raise ValueError('Nonfinite map; refusing model-dependent support')
        if (true<0).any():
            raise ValueError('Negative observed TEC: check fill values / preprocessing')
        self.truth_hash.update(str((time.isoformat(),lead)).encode())
        self.truth_hash.update(np.ascontiguousarray(true, dtype='<f4').tobytes())
        year = time.year
        month = time.month
        season = ('DJF' if month in (12,1,2) else 'MAM' if month in (3,4,5)
                  else 'JJA' if month in (6,7,8) else 'SON')
        sza, lmst, last = solar_geometry(time,self.lat,self.lon)
        latitude = self.lat[:, None]
        masks = [('overall','all',True), ('month',f'{month:02}',True), ('season',season,True)]
        bands = [('SH_high',latitude < -60), ('SH_mid',(latitude>=-60)&(latitude < -30)),
                 ('low',(latitude>=-30)&(latitude<30)), ('NH_mid',(latitude>=30)&(latitude<60)),
                 ('NH_high',latitude>=60)]
        for band, mask in bands:
            masks += [('latitude',band,mask), ('season_latitude',f'{season}/{band}',mask)]
        masks += [('illumination','day',sza<80), ('illumination','terminator',(sza>=80)&(sza<=100)),
                  ('illumination','night',sza>100), ('activity',self.activity.label(time),True)]
        if self.qd is not None and self.cfg.get('eia_coordinate') != 'disabled':
            masks += [('magnetic_latitude','low_abs_le_30',abs(self.qd)<=30),
                      ('magnetic_latitude','south_crest_search',(self.qd>=-25)&(self.qd<=-5)),
                      ('magnetic_latitude','north_crest_search',(self.qd>=5)&(self.qd<=25))]
        for h in range(0,24,3):
            masks.append(('lmst',f'{h:02}-{h+3:02}',((lmst>=h)&(lmst<h+3))[None,:]))
        for event,phase in self.events.labels(time):
            masks.append(('event_phase',f'{event}/{phase}',True))
        for weighting, weight in [('pixel',1.), ('coslat',np.clip(np.cos(np.deg2rad(latitude)),0,None))]:
            for scene, group, mask in masks:
                self.acc.update((year,lead,scene,group,weighting),pred,true,mask,weight)
        err = pred.astype(float)-true
        self.frame_rows.append(dict(target_utc=time.isoformat(),lead_min=lead,
            rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),
            activity=self.activity.label(time)))
        for weighting,w in [('pixel',np.ones_like(err)),
                             ('coslat',np.broadcast_to(np.clip(np.cos(np.deg2rad(latitude)),0,None),err.shape))]:
            self.daily[(year,str(time.date()),lead,weighting)] += [np.sum(w*err**2),w.sum(),err.size]
        if lead not in self.maps:
            self.maps[lead] = [np.zeros_like(err),np.zeros_like(err),0]
        m = self.maps[lead]
        m[0] += err**2
        m[1] += err
        m[2] += 1
        if self.qd is not None and self.cfg.get('eia_coordinate') != 'disabled':
            every = int(self.cfg.get('eia_every_n_frames',4))
            # Selection based on actual target UTC, independently of batch traversal.
            tick = int(time.timestamp()/(86400/self.cfg['frames_per_day']))
            if tick % every == 0:
                self.eia(time,lead,pred,true,lmst,sza)
        self.n_frames += 1
        if time in self.case_times:
            stem=f"{time.strftime('%Y%m%dT%H%M%SZ')}_{lead:g}min"
            self.cases[stem]=(pred.astype('f4').copy(),true.astype('f4').copy())

    def eia(self,time,lead,pred,true,lmst,sza):
        """QD/proxy meridional profiles at fixed geographic longitude, NOT magnetic meridians.
        Report morphology only where truth shows interior twin crests above equatorial trough.
        Failed predicted twin crests remain in detection denominator.
        """
        q = np.arange(-30.,31.)
        def features(y):
            trough = np.min(y[(q>=-5)&(q<=5)])
            peaks=[]
            for lo,hi in [(-25,-5),(5,25)]:
                idx=np.flatnonzero((q>=lo)&(q<=hi))
                j=idx[np.argmax(y[idx])]
                good=j not in (idx[0],idx[-1]) and y[j]-trough >= self.cfg.get('eia_prominence_tecu',1.)
                peaks.append((q[j],y[j],good))
            return peaks, trough
        for j in range(0,len(self.lon),int(self.cfg.get('eia_lon_stride',4))):
            if not 12<=lmst[j]<18:
                continue
            domain=np.flatnonzero(abs(self.lat)<=50)
            coord=self.qd[domain,j]
            order=np.argsort(coord)
            # Reject a non-monotonic magnetic-latitude path and incomplete coverage.
            if not (np.all(np.diff(coord)>0) or np.all(np.diff(coord)<0)):
                continue
            x=coord[order]
            if x[0]>-30 or x[-1]<30 or np.max(np.diff(x))>3:
                continue
            yt=np.interp(q,x,true[domain[order],j]); yp=np.interp(q,x,pred[domain[order],j])
            ft,tt=features(yt); fp,tp=features(yp)
            truth_valid=all(v[2] for v in ft)
            pred_valid=all(v[2] for v in fp)
            row=dict(target_utc=time.isoformat(),lead_min=lead,lon_east=self.lon[j],lmst=lmst[j],
                coordinate=self.cfg['eia_coordinate'],truth_twin=int(truth_valid),pred_twin=int(pred_valid))
            # Conditional errors must be interpreted together with detection failures.
            if truth_valid:
                for hemi,k in [('south',0),('north',1)]:
                    row[f'{hemi}_true_lat']=ft[k][0]
                    row[f'{hemi}_true_amp']=ft[k][1]
                    row[f'{hemi}_position_error_deg']=fp[k][0]-ft[k][0] if pred_valid else np.nan
                    row[f'{hemi}_amplitude_error_tecu']=fp[k][1]-ft[k][1] if pred_valid else np.nan
                row['separation_error_deg']=(fp[1][0]-fp[0][0])-(ft[1][0]-ft[0][0]) if pred_valid else np.nan
                row['trough_error_tecu']=tp-tt
            self.eia_rows.append(row)

    def save(self,out,manifest):
        out=Path(out); out.mkdir(parents=True,exist_ok=True)
        if self.activity.df is not None:
            self.activity.df.to_csv(out/'activity_intervals.csv',index=False)
        if self.events.rows:
            pd.DataFrame([dict(start_utc=a.isoformat(),end_utc=b.isoformat(),event_id=e,phase=p)
                for a,b,e,p in self.events.rows]).to_csv(out/'event_phases.csv',index=False)
        for stem,(pred,true) in self.cases.items():
            np.savez_compressed(out/f'case_{stem}.npz',pred=pred,true=true,lat=self.lat,lon=self.lon)
        manifest['saved_cases']=list(self.cases)
        manifest['requested_case_times_not_in_support']=[t.isoformat() for t in sorted(self.case_times)
            if not any(k.startswith(t.strftime('%Y%m%dT%H%M%SZ')) for k in self.cases)]
        pd.DataFrame(self.acc.rows()).to_csv(out/'metrics.csv',index=False)
        pooled=Accumulator(self.acc.threshold)
        for (year,lead,scene,group,weighting),a in self.acc.data.items():
            pooled.data[(year,-1,scene,group,weighting)] += a
        pd.DataFrame(pooled.rows()).to_csv(out/'pooled_metrics.csv',index=False)
        daily=[dict(year=y,date=d,lead_min=h,weighting=w,sse=a[0],weight=a[1],n_pixels=int(a[2]))
               for (y,d,h,w),a in sorted(self.daily.items())]
        pd.DataFrame(daily).to_csv(out/'daily.csv',index=False)
        pd.DataFrame(self.frame_rows).to_csv(out/'frame_metrics.csv',index=False)
        if self.eia_rows:
            e=pd.DataFrame(self.eia_rows); e.to_csv(out/'eia_profiles.csv',index=False)
            summary=[]
            for h,g in e.groupby('lead_min'):
                valid=g.truth_twin==1
                tp=int(((g.pred_twin==1)&valid).sum())
                r=dict(lead_min=h,n_profiles=len(g),truth_twin=int(valid.sum()),
                       recall=tp/valid.sum() if valid.any() else np.nan,
                       false_positive_rate=float(g.loc[~valid,'pred_twin'].mean()),matched_twins=tp)
                for k in g.columns:
                    if '_error_' in k:
                        r[k+'_mae']=g.loc[valid,k].abs().mean()
                        r[k+'_bias']=g.loc[valid,k].mean()
                summary.append(r)
            pd.DataFrame(summary).to_csv(out/'eia_summary.csv',index=False)
        fields=dict(lat=self.lat,lon=self.lon)
        for h,(se,b,n) in self.maps.items():
            fields[f'rmse_{h:g}']=np.sqrt(se/n)
            fields[f'bias_{h:g}']=b/n
        np.savez_compressed(out/'spatial.npz',**fields)
        manifest.update(target_sha256=self.truth_hash.hexdigest(),evaluated_target_lead_pairs=self.n_frames)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
