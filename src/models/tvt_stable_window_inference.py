"""Re-evaluate FFT-boundary near ties as singleton windows, without score choice."""
import torch
from torch import nn
from src.models.tvt_online import IndependentMultiPeriod,IndependentSpectral


class StableWindowInference(nn.Module):
    """Numerical adapter, not a trainable module or a detection threshold.

    FP32 rounding can swap the kth and (k+1)th FFT bins near equality. A fixed
    machine-precision guard triggers singleton execution of the whole window.
    It neither picks a period from detection results nor averages models.
    """
    def __init__(self,model):
        super().__init__();self.model=model;self.collect=False;self.ambiguous=None
        self.fallback_count=0;self.window_count=0
        for module in self.model.modules():
            if isinstance(module,(IndependentMultiPeriod,IndependentSpectral)):
                module.register_forward_pre_hook(self._inspect)
        self.eval()

    def _inspect(self,module,args):
        if not self.collect:return
        x=args[0];assert len(x)==len(self.ambiguous),'unexpected grouped spectral input'
        k=module.source.top_k if isinstance(module,IndependentMultiPeriod) else module.source.topk
        amp=torch.fft.rfft(x.detach().float(),dim=1).abs().mean(2);amp[:,0]=0
        count=min(k,amp.shape[1]-1)
        if count+1>amp.shape[1]:return
        values=amp.topk(count+1,dim=1).values
        tolerance=32*torch.finfo(torch.float32).eps*values[:,0].clamp_min(torch.finfo(torch.float32).tiny)
        self.ambiguous|=(values[:,count-1]-values[:,count])<=tolerance

    @torch.no_grad()
    def forward(self,x):
        if self.training:raise RuntimeError('numerical inference adapter is evaluation-only')
        self.window_count+=len(x)
        self.ambiguous=torch.zeros(len(x),dtype=torch.bool,device=x.device);self.collect=len(x)>1
        try:out=self.model(x)
        finally:self.collect=False
        ids=torch.flatnonzero(self.ambiguous).tolist() if hasattr(torch,'flatnonzero') else self.ambiguous.nonzero(as_tuple=True)[0].tolist()
        self.fallback_count+=len(ids)
        if ids:
            out=dict(out);out['recon']=out['recon'].clone()
            for i in ids:out['recon'][i]=self.model(x[i:i+1])['recon'][0]
        return out
