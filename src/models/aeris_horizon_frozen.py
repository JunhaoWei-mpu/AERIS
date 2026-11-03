"""Inference-only interface for the frozen empirical-reference AERIS mechanism.

This wraps existing equations and weights. It does not fit a reference, choose
a checkpoint, alter missing-data policy, or implement an alarm threshold.
"""
from pathlib import Path
import numpy as np
import torch
from src.models.tvt_control_chain import ChainInputs,SEMANTICS
from src.models.tvt_horizon_response import summarize_horizon_response
from src.models.tvt_response_median import ay_energy
from src.models.tvt_online import make_window_independent
from src.models.tvt_stable_window_inference import StableWindowInference


class FrozenHorizonAERIS:
    def __init__(self, bundle, device='cpu'):
        if bundle['format'] != 'aeris-horizon35-v1':
            raise ValueError('unsupported frozen bundle')
        self.platform=bundle['platform'];self.device=torch.device(device)
        self.normalization=bundle['normalization'];self.constant=bundle['constant']
        self.mean=np.asarray(self.normalization['mean'],np.float32)
        self.std=np.asarray(self.normalization['std'],np.float32)
        self.reference=bundle['reference'];self.calibration=bundle['calibration']
        self.base=StableWindowInference(make_window_independent(bundle['base'].to(self.device).eval()))
        self.inputs=ChainInputs(self.platform,self.normalization,.4)
        self.base_indices=list(SEMANTICS[self.platform].base)
        self.active=[i for i,j in enumerate(self.base_indices) if j not in self.constant]

    @classmethod
    def load(cls,path,device='cpu'):
        return cls(torch.load(Path(path),map_location='cpu',weights_only=False),device)

    @torch.no_grad()
    def score_windows(self,windows,source_groups,batch_size=128):
        """Score finite, time-ordered raw W=100 windows ending at the decision.

        The caller supplies physical channel ordering, causal resampling and
        continuity checks. Source groups identify original recordings, so a
        known source is excluded from its own nominal reference. This function
        never consumes fault labels. Availability and any 199-sample benchmark
        support mask are managed by the data adapter, not silently imputed here.
        """
        raw=np.asarray(windows,dtype=np.float32);groups=np.asarray(source_groups,dtype=str)
        if raw.ndim!=3 or raw.shape[1:]!=(100,len(self.mean)) or groups.shape!=(len(raw),):
            raise ValueError('expected aligned raw windows (N,100,D) and N source groups')
        if len(raw)==0 or batch_size<1 or not np.isfinite(raw).all():
            raise ValueError('finite nonempty complete windows and positive batch size required')
        base=[];summaries=[]
        for start in range(0,len(raw),batch_size):
            x=torch.from_numpy(((raw[start:start+batch_size]-self.mean)/self.std).astype(np.float32))
            summaries.append(summarize_horizon_response(x,self.inputs,self.constant))
            x=x.to(self.device);x[:,:,self.constant]=0;y=x[:,:,self.base_indices]
            base.append((self.base(y)['recon']-y).square()[:,:,self.active].mean((1,2)).cpu().numpy())
        q={k:np.concatenate([s[k] for s in summaries]) for k in summaries[0]};q['groups']=groups
        d=self.reference.evidence(q,mode='context')
        _,d['ay_L'],_=ay_energy(self.reference,q,'context')
        base=np.concatenate(base);c=self.calibration
        evidence={}
        for edge in ('ca','ay'):
            location=c.location[edge].transform(d[edge+'_L'])
            dispersion=c.dispersion[edge].transform(c.dispersion_value(edge,d[edge+'_R']))
            evidence[edge]=c.branch[edge].transform(np.maximum(location,dispersion))
        base_evidence=c.base.transform(base)
        before_final=np.maximum.reduce([base_evidence,evidence['ca'],evidence['ay']])
        return dict(score=c.final['full'].transform(before_final),base_raw=base,
                    base_evidence=base_evidence,ca_evidence=evidence['ca'],ay_evidence=evidence['ay'])
