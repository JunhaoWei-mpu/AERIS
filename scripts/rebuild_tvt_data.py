#!/usr/bin/env python3
"""Rebuild existing ALFA/RflyMAD telemetry without legacy preprocessing leaks.

Creates a NEW dataset namespace; never overwrites historical tensors. Ground
truth is metadata, never a feature. Raw injection flags and pre-injection
training eligibility are kept separately for periodic RflyMAD faults.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.tvt_raw import causal_hold, read_topic, sha256, alfa_fault_annotation, rfly_fault_flags


# All substantive legacy ALFA channels remain, with truthful roles. The old
# unassigned SERVO1 is removed; actual rudder/left/right flaperon outputs and
# the four recorded navigation references replace guessed actuator semantics.
ALFA_CHANNELS = [
    *[(f'pilot/{name}', 'mavros-rc-in', f'field.channels{i}') for i,name in enumerate(('roll','pitch','throttle','yaw'))],
    *[(f'actuator/{name}', 'mavros-rc-out', f'field.channels{i}') for i,name in
      ((1,'elevator_pwm'),(2,'throttle_pwm'),(3,'rudder_pwm'),(4,'flaperon_right_pwm'),(5,'flaperon_left_pwm'))],
    *[(f'reference/{name}', f'mavros-nav_info-{name}', 'field.commanded') for name in ('roll','pitch','yaw','airspeed')],
    ('airspeed','mavros-nav_info-airspeed','field.measured'),
    ('pressure/atm','mavros-imu-atm_pressure','field.fluid_pressure'),
    *[(f'imu/gyro_{axis}','mavros-imu-data',f'field.angular_velocity.{axis}') for axis in 'xyz'],
    *[(f'state/{name}',f'mavros-nav_info-{name}','field.measured') for name in ('roll','pitch','yaw')],
    ('state/alt','mavros-global_position-rel_alt','field.data'),
]
RFLY_CHANNELS=['cmd_roll','cmd_pitch','cmd_yaw','cmd_thrust','motor_1','motor_2','motor_3','motor_4',
                'gyro_x','gyro_y','gyro_z','baro_pres','att_roll','att_pitch','att_yaw','alt_z']


def dump(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def source_record(p):return {'path':str(p.resolve()),'sha256':sha256(p)}


def build_alfa(folder, max_age):
    ann=alfa_fault_annotation(folder)
    if not ann['eligible']:raise ValueError(ann['reason'])
    cache={}; sources=[source_record(p) for p in ann['sources']]
    for _,topic,_ in ALFA_CHANNELS:
        if topic in cache:continue
        matches=list(folder.glob(f'*-{topic}.csv'))
        if len(matches)!=1:raise ValueError(f'missing/ambiguous topic {topic}')
        cache[topic]=read_topic(matches[0],'%time'); sources.append(source_record(matches[0]))
    # Use local integer-nanosecond origin before float conversion.
    origin=min(int(d['%time'].iloc[0]) for d in cache.values())
    starts=[(int(d['%time'].iloc[0])-origin)/1e9 for d in cache.values()]
    ends=[(int(d['%time'].iloc[-1])-origin)/1e9 for d in cache.values()]
    grid=np.arange(max(starts),min(ends),1/50.)
    if len(grid)<100:raise ValueError('insufficient common topic coverage')
    values=[];ages=[]
    for name,topic,col in ALFA_CHANNELS:
        d=cache[topic]
        if col not in d:raise ValueError(f'missing channel {col}')
        t=(d['%time'].to_numpy(dtype=np.int64)-origin)/1e9
        x,age=causal_hold(t,pd.to_numeric(d[col],errors='coerce'),grid,max_age=max_age)
        values.append(x);ages.append(age)
    x=np.stack(values,axis=1)
    onset=(ann['onset_ns']-origin)/1e9 if ann['onset_ns'] is not None else None
    label=(grid>=onset).astype(np.int8) if onset is not None else np.zeros(len(grid),np.int8)
    families=ann['families'];ft='+'.join(families) if families else 'nominal'
    rec={'flight_id':folder.name,'fault_type':ft,'fault_families':families,'raw_origin_ns':origin,
         'start_ns':origin+round(grid[0]*1e9),'end_ns':origin+round(grid[-1]*1e9),
         'label_source':'first positive raw failure-status message; published processed sequence',
         'onset_relative_s':onset-grid[0] if onset is not None else None,'sources':sources,
         'source_name_group':re.match(r'^(carbonZ_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})',folder.name).group(1)}
    arrays=dict(x=x,t=(grid-grid[0]).astype(np.float64),label=label,
                nominal_eligible=label==0,channel_age_s=np.stack(ages,axis=1).astype(np.float32),
                sample_rate=np.array(50.),channel_names=np.array([c[0] for c in ALFA_CHANNELS]),
                fault_type=np.array(ft),fault_families=np.array(families))
    return arrays,rec


def build_rfly(folder,max_age):
    logs=[p for p in folder.glob('log_*') if p.is_dir()]
    if len(logs)!=1:raise ValueError('missing/ambiguous PX4 log')
    log=logs[0]; stem=log.name
    topics=('actuator_controls_0_0','actuator_outputs_0','sensor_combined_0','sensor_baro_0',
            'vehicle_attitude_0','vehicle_local_position_0')
    cache={};sources=[]
    for name in topics:
        path=log/f'{stem}_{name}.csv'; cache[name]=read_topic(path,'timestamp');sources.append(source_record(path))
    flags,active,fp=rfly_fault_flags(log);sources.append(source_record(fp))
    top,regime,case=folder.parts[-3:]
    ft='nominal' if top=='Real-No_Fault' else ('motor' if top=='Real-Motor' else 'sensor_'+regime.split('-',1)[1])
    if ft!='nominal' and not active.any():raise ValueError('fault case without verified injection flag')
    if ft=='nominal' and active.any():raise ValueError('nominal case with fault injection flag')
    origin=min(int(d.timestamp.iloc[0]) for d in cache.values())
    start=max(int(d.timestamp.iloc[0]) for d in cache.values())
    end=min(int(d.timestamp.iloc[-1]) for d in cache.values())
    # Include only timestamps with injection-topic coverage; do not infer labels
    # outside recorded flag coverage. Ground/flight mode is stored for audit.
    start=max(start,int(flags.timestamp.iloc[0]));end=min(end,int(flags.timestamp.iloc[-1]))
    grid_us=np.arange(start,end,50000,dtype=np.int64);grid=(grid_us-origin)/1e6
    if len(grid)<100:raise ValueError('insufficient common topic coverage')
    def sample(topic,col):
        d=cache[topic]
        return causal_hold((d.timestamp.to_numpy(np.int64)-origin)/1e6,pd.to_numeric(d[col],errors='coerce'),grid,max_age=max_age)
    vals=[];ages=[]
    for topic,cols in [('actuator_controls_0_0',[f'control[{i}]' for i in range(4)]),
                       ('actuator_outputs_0',[f'output[{i}]' for i in range(4)]),
                       ('sensor_combined_0',[f'gyro_rad[{i}]' for i in range(3)]),
                       ('sensor_baro_0',['pressure'])]:
        for col in cols:
            x,age=sample(topic,col);vals.append(x);ages.append(age)
    d=cache['vehicle_attitude_0'];q=[pd.to_numeric(d[f'q[{i}]'],errors='coerce').to_numpy() for i in range(4)]
    q0,q1,q2,q3=q
    euler=[np.arctan2(2*(q0*q1+q2*q3),1-2*(q1*q1+q2*q2)),
           np.arcsin(np.clip(2*(q0*q2-q3*q1),-1,1)),
           np.arctan2(2*(q0*q3+q1*q2),1-2*(q2*q2+q3*q3))]
    for v in euler:
        x,age=causal_hold((d.timestamp.to_numpy(np.int64)-origin)/1e6,v,grid,max_age=max_age);vals.append(x);ages.append(age)
    x,age=sample('vehicle_local_position_0','z');vals.append(-x);ages.append(age)
    flag,flag_age=causal_hold((flags.timestamp.to_numpy(np.int64)-origin)/1e6,active.astype(float),grid,max_age=max_age)
    label=np.where(np.isfinite(flag),flag,-1).astype(np.int8)
    onset=int(flags.loc[active,'timestamp'].iloc[0]) if active.any() else None
    # A fault-off pulse after first injection is NOT admissible nominal training.
    eligible=(label==0)&((grid_us<onset) if onset is not None else True)
    flight_id='__'.join((top,regime,case));families=[] if ft=='nominal' else [ft]
    rec={'flight_id':flight_id,'fault_type':ft,'fault_families':families,'top':top,'regime':regime,'case':case,
         'raw_origin_us':origin,'onset_relative_s':(onset-start)/1e6 if onset is not None else None,
         'label_source':'native PX4 rfly_ctrl_lxl ID; zero-order hold; default ID 0/1500',
         'nominal_training_policy':'strictly before first injection; excludes periodic off phases',
         'source_name_group':'__'.join((top,regime,re.sub(r'_\d+$','',case))),
         'start_ns':int(grid_us[0])*1000,'end_ns':int(grid_us[-1])*1000,'sources':sources}
    return dict(x=np.stack(vals,axis=1),t=(grid_us-grid_us[0])/1e6,label=label,nominal_eligible=eligible,
                channel_age_s=np.stack(ages,axis=1).astype(np.float32),sample_rate=np.array(20.),
                channel_names=np.array(RFLY_CHANNELS),fault_type=np.array(ft),fault_families=np.array(families)),rec


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['alfa','rflymad'],required=True)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-age',type=float,default=1.0)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):raise SystemExit('refusing to overwrite existing dataset')
    a.out.mkdir(parents=True,exist_ok=True)
    folders=sorted(x for x in a.raw.iterdir() if x.is_dir()) if a.dataset=='alfa' else sorted(
        x for top in ('Real-No_Fault','Real-Motor','Real-Sensors') for x in (a.raw/top).glob('*/*') if x.is_dir())
    records=[];excluded=[];builder=build_alfa if a.dataset=='alfa' else build_rfly
    for i,folder in enumerate(folders):
        try:arrays,rec=builder(folder,a.max_age)
        except (ValueError,FileNotFoundError,KeyError) as e:
            excluded.append({'source':str(folder),'reason':str(e)});continue
        dest=a.out/(rec['flight_id']+'.npz');np.savez_compressed(dest,**arrays)
        rec.update({'npz_sha256':sha256(dest),'n_steps':len(arrays['t']),'n_channels':arrays['x'].shape[1],
                    'finite_fraction':float(np.isfinite(arrays['x']).mean()),'unknown_label_samples':int((arrays['label']<0).sum()),
                    'nominal_eligible_samples':int(arrays['nominal_eligible'].sum())})
        records.append(rec)
        if (i+1)%50==0:print(f'{i+1}/{len(folders)} raw cases processed',flush=True)
    dump(a.out/'meta.json',{'schema':'aeris-tvt-raw-v1','dataset':a.dataset,'rate':50 if a.dataset=='alfa' else 20,
                          'channels':[c[0] for c in ALFA_CHANNELS] if a.dataset=='alfa' else RFLY_CHANNELS,
                          'normalization':'none; fit train-nominal only downstream','sampling':'causal last received sample',
                          'max_age_s':a.max_age,'script_sha256':sha256(Path(__file__)),
                          'helper_sha256':sha256(Path('src/data/tvt_raw.py')),'flights':records,'excluded':excluded,
                          'performance_evidence_approved':False})
    print(json.dumps({'dataset':a.dataset,'written':len(records),'excluded':excluded},indent=2))


if __name__=='__main__':main()
