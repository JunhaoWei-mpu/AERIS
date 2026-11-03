"""Batch independent evaluation equivalent to each window's singleton FFT.

Only period selection is separated. Windows with the same independently
selected periods share convolution calls; no other window selects their periods.
This is an inference adapter for the existing weights, not a new architecture.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .halo import MultiPeriodBlock
from .modules.spectral import SpectralBranch


def period_groups(x,k,unique):
    amplitude=torch.fft.rfft(x.detach().float(),dim=1).abs().mean(dim=2)
    amplitude[:,0]=0.
    bins=amplitude.topk(min(k,amplitude.shape[1]-1),dim=1).indices.cpu().tolist()
    groups={};length=x.shape[1]
    for i,selected in enumerate(bins):
        periods=[]
        for f in selected:
            if f<=0:continue
            period=min(length,max(2,length//f))
            if not unique or period not in periods:periods.append(period)
        if not periods:periods=[max(2,length//(2 if unique else 4))]
        groups.setdefault(tuple(periods),[]).append(i)
    return groups


class IndependentMultiPeriod(nn.Module):
    def __init__(self,source):super().__init__();self.source=source

    def forward(self,x):
        if self.training:raise RuntimeError('online period adapter is evaluation-only')
        output=None;s=self.source
        for periods,indices in period_groups(x,s.top_k,False).items():
            chunk=x[indices];b,t,c=chunk.shape;features=[]
            for period in periods:
                n=t//period
                grid=chunk[:,:n*period].reshape(b,n,period,c).permute(0,3,2,1).contiguous().reshape(b*c,1,period,n)
                f=torch.stack([branch(grid) for branch in s.branches],0).mean(0).mean((2,3))
                features.append(f.reshape(b,c,s.inner_dim))
            value=s.out_proj(torch.stack(features,0).mean(0))
            if output is None:output=value.new_empty((len(x),*value.shape[1:]))
            output[indices]=value
        return output


class IndependentSpectral(nn.Module):
    def __init__(self,source):super().__init__();self.source=source

    def forward(self,x):
        if self.training:raise RuntimeError('online period adapter is evaluation-only')
        output=None;s=self.source
        for periods,indices in period_groups(x,s.topk,True).items():
            chunk=x[indices];outs=[];weights=[]
            for period in periods:
                grid,pad=s._fold(chunk,period);outs.append(s._unfold(s.conv(grid),pad))
                weights.append(grid.square().mean((2,3)).mean(1,keepdim=True))
            stacked=torch.stack(outs,0)
            weight=F.softmax(torch.stack(weights,0),dim=0).unsqueeze(-1).unsqueeze(-1)
            merged=(stacked*weight.squeeze(-1)).sum(0)
            value=s.proj(merged.permute(0,2,1).contiguous())
            if output is None:output=value.new_empty((len(x),*value.shape[1:]))
            output[indices]=value
        return output


def make_window_independent(model):
    """Modify only FFT evaluation, retaining every original parameter tensor."""
    if model.training:raise ValueError('put the backbone in eval mode first')
    def replace(parent):
        for name,child in list(parent.named_children()):
            if isinstance(child,MultiPeriodBlock):setattr(parent,name,IndependentMultiPeriod(child))
            elif isinstance(child,SpectralBranch):setattr(parent,name,IndependentSpectral(child))
            else:replace(child)
    replace(model)
    return model.eval()
