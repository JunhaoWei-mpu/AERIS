"""Nominal-only control-chain evidence. No test-label or fitting-on-score API.

PWM is a logged output of the actuator command pathway, not measured physical
surface position, rotor speed or thrust. See tvt/CONTROL_CHAIN_PROTOCOL.md.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from sklearn.covariance import LedoitWolf
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class ChainSemantics:
    platform: str
    intent: tuple
    actuation: tuple
    response: tuple
    angles: tuple
    angle_period: float
    sample_rate: float

    @property
    def base(self):
        return self.actuation + self.response


SEMANTICS = {
    'alfa': ChainSemantics('alfa', (0,1,2,3,9,10,11,12), (4,5,6,7,8),
                           tuple(range(13,22)), (18,19,20), 360., 50.),
    'rflymad': ChainSemantics('rflymad', tuple(range(4)), tuple(range(4,8)),
                              tuple(range(8,16)), (12,13,14), 2*np.pi, 20.),
}


class ChainInputs(nn.Module):
    """Causal predictors observe drivers through t-1, states through t-h.

    No current target enters either predictor or its qualification context.
    State history encodes circular coordinates as sin/cos; AY predicts the
    shortest signed angular increment, in training-normalized channel units.
    """
    def __init__(self, platform, normalization, horizon_seconds=.4):
        super().__init__()
        self.semantics = SEMANTICS[platform]
        self.horizon = int(round(horizon_seconds*self.semantics.sample_rate))
        self.register_buffer('mean', torch.tensor(normalization['mean'], dtype=torch.float32))
        self.register_buffer('std', torch.tensor(normalization['std'], dtype=torch.float32))

    def state(self, x):
        parts=[]
        for j in self.semantics.response:
            z=x[...,j:j+1]
            if j in self.semantics.angles:
                angle=(z*self.std[j]+self.mean[j])*(2*np.pi/self.semantics.angle_period)
                parts.extend((angle.sin(), angle.cos()))
            else:
                parts.append(z)
        return torch.cat(parts, -1)

    def forward(self, x, edge):
        if edge not in ('ca','ay') or x.shape[1] <= self.horizon+2:
            raise ValueError('invalid edge or insufficient history')
        s=self.semantics
        ids=s.intent if edge=='ca' else s.actuation
        driver=x[:,:-1,list(ids)]
        history=self.state(x[:,:-self.horizon])
        if edge=='ca':
            target=x[:,-1,list(s.actuation)]
        else:
            delta=x[:,-1]-x[:,-self.horizon-1]
            target=[]
            for j in s.response:
                d=delta[:,j]
                if j in s.angles:
                    raw=d*self.std[j]
                    d=(torch.remainder(raw+s.angle_period/2,s.angle_period)-s.angle_period/2)/self.std[j]
                target.append(d)
            target=torch.stack(target,-1)
        # Qualification context is a compact INPUT-only summary. It is not
        # reconstruction residual, future realized response, or anomaly score.
        context=torch.cat((driver[:,-1], driver.mean(1), driver.std(1,unbiased=False),
                           history[:,-1], history.mean(1), history.std(1,unbiased=False)),1)
        innovation=driver[:,1:]-driver[:,:-1]
        gram=innovation.transpose(1,2)@innovation/max(1,innovation.shape[1])
        eye=torch.eye(gram.shape[-1],device=x.device,dtype=x.dtype)
        excitation=torch.linalg.slogdet(eye+gram.float()).logabsdet/gram.shape[-1]
        return driver,history,target,context,excitation


class EdgePredictor(nn.Module):
    def __init__(self, driver_dim, history_dim, output_dim, hidden=64, command_zero=False):
        super().__init__()
        self.command_zero=command_zero
        self.driver=nn.GRU(driver_dim, hidden, batch_first=True)
        self.history=nn.GRU(history_dim, hidden, batch_first=True)
        self.head=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,2*output_dim))
        self.register_buffer('target_mean',torch.zeros(output_dim))
        self.register_buffer('target_std',torch.ones(output_dim))

    def forward(self, driver, history):
        if self.command_zero:
            driver=torch.zeros_like(driver)
        _,a=self.driver(driver); _,h=self.history(history)
        mu,log_scale=self.head(torch.cat((a[-1],h[-1]),-1)).chunk(2,-1)
        return mu,log_scale.clamp(-4.,3.)

    def loss(self, driver, history, target):
        mu,ls=self(driver,history)
        target=(target-self.target_mean)/self.target_std
        return (.5*((target-mu)*torch.exp(-ls)).square()+ls).mean()

    def evidence(self, driver, history, target):
        mu,ls=self(driver,history)
        target=(target-self.target_mean)/self.target_std
        innovation=(((target-mu)*torch.exp(-ls)).square().mean(-1)).sqrt()
        # Dimensionless scale after train-only target standardization.
        # This value has NO realized-target dependence.
        uncertainty=ls.exp().square().mean(-1).sqrt()
        return innovation,uncertainty


def _finite(*arrays):
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('nonfinite calibration or score input')


def _decreasing(x, low, high):
    return np.clip(1.-(x-low)/max(float(high-low),1e-8),0.,1.)


class Qualification:
    def fit(self, train_context, calibration_context, excitation, uncertainty):
        _finite(train_context,calibration_context,excitation,uncertainty)
        self.mean=np.mean(train_context,axis=0)
        self.std=np.maximum(np.std(train_context,axis=0),1e-5)
        self.support=LedoitWolf().fit((train_context-self.mean)/self.std)
        distance=self.distance(calibration_context)
        self.support_limits=np.quantile(distance,[.95,.999])
        self.exc_limits=np.quantile(excitation,[.1,.5])
        self.unc_limits=np.quantile(uncertainty,[.5,.95])
        return self

    def distance(self, context):
        return self.support.mahalanobis((context-self.mean)/self.std)

    def factors(self, context, excitation, uncertainty):
        _finite(context,excitation,uncertainty)
        low,high=self.exc_limits
        exc=np.clip((excitation-low)/max(float(high-low),1e-8),0.,1.)
        support=_decreasing(self.distance(context),*self.support_limits)
        unc=_decreasing(uncertainty,*self.unc_limits)
        return np.stack((exc,support,unc),1)

    def score(self, context, excitation, uncertainty):
        return self.factors(context,excitation,uncertainty).prod(1)


class ConditionalResidual:
    """Two low-complexity monotone regressions on nominal calibration only."""
    def fit(self, base, residual):
        _finite(base,residual)
        # Bin before isotonic fitting: avoid an unrestricted step at each of
        # many highly overlapping windows. Equal-count bins, at least 80/bin.
        self.n_bins=min(16,max(1,len(base)//80))
        groups=np.array_split(np.argsort(base,kind='stable'),self.n_bins)
        x=np.array([np.mean(base[g]) for g in groups])
        y=np.array([np.mean(residual[g]) for g in groups])
        weights=np.array([len(g) for g in groups])
        self.location=IsotonicRegression(out_of_bounds='clip').fit(x,y,sample_weight=weights)
        deviation=np.abs(residual-self.location.predict(base))
        self.floor=max(float(np.median(deviation))*.1,1e-6)
        scale=np.array([np.mean(deviation[g]) for g in groups])
        self.scale=IsotonicRegression(out_of_bounds='clip').fit(x,scale,sample_weight=weights)
        return self

    def transform(self, base, residual):
        _finite(base,residual)
        return (residual-self.location.predict(base))/np.maximum(self.scale.predict(base),self.floor)


class EmpiricalTail:
    """Conservative >= survival with add-one smoothing; no tail extrapolation.

    Finite-tail ties and serial dependence are reported, not hidden behind a
    conformal-coverage or calibrated posterior-probability claim.
    """
    def fit(self, values):
        _finite(values)
        self.values=np.sort(np.asarray(values,dtype=np.float64))
        if len(self.values)<20:raise ValueError('insufficient nominal tail reference')
        return self

    @property
    def cap(self):
        return float(np.log(len(self.values)+1.))

    def transform(self, values):
        _finite(values)
        left=np.searchsorted(self.values,values,side='left')
        survival=(1.+len(self.values)-left)/(len(self.values)+1.)
        return -np.log(survival+1e-12)


def fuse(base, edge_evidence, qualification, rule):
    base=np.asarray(base); evidence=np.asarray(edge_evidence); q=np.asarray(qualification)
    if evidence.shape!=q.shape or evidence.shape!=(len(base),2):raise ValueError('two edges required')
    _finite(base,evidence,q)
    if np.any((q<0)|(q>1)):raise ValueError('qualification outside [0,1]')
    if rule=='max':return np.maximum(base,(q*evidence).max(1))
    if rule=='positive_excess':return base+(q*np.maximum(evidence-base[:,None],0.)).sum(1)
    raise ValueError('only the two registered fusion rules are allowed')


class ChainCalibrator:
    def fit(self, regression, tail, train_context):
        self.base=EmpiricalTail().fit(tail['base'])
        self.residual={};self.tails={};self.qualification={}
        for edge in ('ca','ay'):
            r=regression[edge]
            self.qualification[edge]=Qualification().fit(train_context[edge],r['context'],r['excitation'],r['uncertainty'])
            # Command-zero gets its own residual and nominal-tail calibration,
            # but exactly the FULL qualifiers, so zero excitation cannot make
            # the ablation collapse to Base by construction.
            for prefix in ('','zero_'):
                key=prefix+edge
                self.residual[key]=ConditionalResidual().fit(regression['base'],regression[key]['residual'])
                orth=self.residual[key].transform(tail['base'],tail[key]['residual'])
                self.tails[key]=EmpiricalTail().fit(orth)
        return self

    def evidence(self, raw):
        base=self.base.transform(raw['base'])
        q=np.stack([self.qualification[e].score(raw[e]['context'],raw[e]['excitation'],raw[e]['uncertainty']) for e in ('ca','ay')],1)
        tails={}
        for prefix in ('','zero_'):
            tails[prefix]=np.stack([self.tails[prefix+e].transform(self.residual[prefix+e].transform(raw['base'],raw[prefix+e]['residual'])) for e in ('ca','ay')],1)
        return base,tails,q

    def score(self, raw, rule):
        base,e,q=self.evidence(raw)
        return {'base':base, 'predictor_free':base.copy(),
                'command_zero':fuse(base,e['zero_'],q,rule),
                'ca_only':fuse(base,e[''],q*np.array([1.,0.]),rule),
                'ay_only':fuse(base,e[''],q*np.array([0.,1.]),rule),
                'full':fuse(base,e[''],q,rule),
                'without_qualification':fuse(base,e[''],np.ones_like(q),rule)}
