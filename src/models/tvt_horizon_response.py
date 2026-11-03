"""Measure AY response over the already fixed physical increment horizon."""
import numpy as np
import torch
from src.models.tvt_matched_chain import summarize_chain


@torch.no_grad()
def summarize_horizon_response(x,inputs,constant):
    result=summarize_chain(x,inputs,constant)
    x=x.clone();x[:,:,constant]=0;s=inputs.semantics;h=inputs.horizon
    assert x.shape[1]>=2*h
    diff=x[:,h:]-x[:,:-h];parts=[]
    for j in s.response:
        d=diff[:,:,j]
        if j in s.angles:
            raw=d*inputs.std[j]
            d=(torch.remainder(raw+s.angle_period/2,s.angle_period)-s.angle_period/2)/inputs.std[j]
        parts.append(d)
    response=torch.stack(parts,-1).double().cpu().numpy()
    active=[i for i,j in enumerate(s.response) if j not in constant]
    response=response[:,:,active]
    cumulative=np.concatenate([np.zeros_like(response[:,:1]),np.cumsum(response,axis=1)],axis=1)
    response=(cumulative[:,h:]-cumulative[:,:-h])/h
    result['ay_M']=response.mean(1);result['ay_V']=response.var(1)
    return result
