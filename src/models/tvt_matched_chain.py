"""Source-excluding empirical response distributions on two recorded edges."""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from src.models.tvt_exponential_tail import ExponentialTail


@torch.no_grad()
def summarize_chain(x, inputs, constant):
    x=x.clone();x[:,:,constant]=0
    s=inputs.semantics;h=inputs.horizon
    initial=inputs.state(x[:,:1]).double().cpu().numpy()[:,0]
    result={'history':initial}
    for edge,indices in (('ca',s.intent),('ay',s.actuation)):
        driver=x[:,:-1,list(indices)].double().cpu().numpy()
        result[edge+'_context']=np.concatenate([driver.mean(1),driver.std(1),driver[:,-1],initial],1)
        if edge=='ca':
            response=x[:,:,list(s.actuation)].double().cpu().numpy()
            active=[i for i,j in enumerate(s.actuation) if j not in constant]
        else:
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
        result[edge+'_M']=response.mean(1);result[edge+'_V']=response.var(1)
    return result


class MatchedChainReference:
    k=64
    epsilon=1e-8

    def fit(self,library):
        self.library=library
        keys=('history','ca_context','ay_context')
        self.mean={k:library[k].mean(0) for k in keys}
        self.scale={k:np.maximum(library[k].std(0),1e-6) for k in keys}
        self.coordinates={k:(library[k]-self.mean[k])/self.scale[k] for k in keys}
        if any(np.sum(library['groups']!=g)<self.k for g in np.unique(library['groups'])):
            raise ValueError('fewer than fixed 64 eligible other-source windows')
        return self

    def evidence(self,query,mode='context',coverage=False):
        if mode not in ('context','history','pooled'):raise ValueError(mode)
        result={}
        for edge in ('ca','ay'):
            key=edge+'_context' if mode=='context' else 'history'
            coord=(query[key]-self.mean[key])/self.scale[key]
            energy=[];ratios=[];cover=[]
            pooled={}
            for start in range(0,len(coord),128):
                stop=min(start+128,len(coord));groups=query['groups'][start:stop]
                if mode=='pooled':
                    center=[];scale2=[];typical=[];lower=[];upper=[]
                    for group in groups:
                        if str(group) not in pooled:
                            take=self.library['groups']!=group
                            means=self.library[edge+'_M'][take];variances=self.library[edge+'_V'][take]
                            mu=np.median(means,0);tv=np.median(variances,0)
                            sc=np.maximum(tv+(1.4826*np.median(abs(means-mu),0))**2,self.epsilon)
                            lo,hi=np.quantile(means,[.025,.975],axis=0)
                            pooled[str(group)]=(mu,sc,tv,lo,hi)
                        mu,sc,tv,lo,hi=pooled[str(group)]
                        center.append(mu);scale2.append(sc);typical.append(tv);lower.append(lo);upper.append(hi)
                    center,scale2,typical,lower,upper=map(np.asarray,(center,scale2,typical,lower,upper))
                else:
                    distance=cdist(coord[start:stop],self.coordinates[key],'sqeuclidean')
                    distance[groups[:,None]==self.library['groups'][None,:]]=np.inf
                    ids=np.argpartition(distance,self.k-1,axis=1)[:,:self.k]
                    assert np.isfinite(np.take_along_axis(distance,ids,1)).all()
                    means=self.library[edge+'_M'][ids];variances=self.library[edge+'_V'][ids]
                    center=np.median(means,1);typical=np.median(variances,1)
                    scale2=np.maximum(typical+(1.4826*np.median(abs(means-center[:,None]),1))**2,self.epsilon)
                    if coverage:lower,upper=np.quantile(means,[.025,.975],axis=1)
                m=query[edge+'_M'][start:stop];v=query[edge+'_V'][start:stop]
                energy.append(((v+(m-center)**2)/scale2).mean(1))
                ratios.append(np.log((v+self.epsilon)/(typical+self.epsilon)))
                if coverage:cover.append(((m>=lower)&(m<=upper)).mean(1))
            result[edge+'_L']=np.concatenate(energy);result[edge+'_R']=np.concatenate(ratios)
            if coverage:result[edge+'_coverage']=np.concatenate(cover)
        return result


class MatchedChainCalibration:
    keys=('ca','ay','zero_ca','zero_ay','free_ca','free_ay')

    def primitive(self,d):
        return {e:np.stack([self.location[e].transform(d[e+'_L']),
            self.dispersion[e].transform(abs(d[e+'_R']-self.center[e]).mean(1))],1) for e in self.keys}

    def raw(self,d):
        p=self.primitive(d);b={e:self.branch[e].transform(v.max(1)) for e,v in p.items()}
        base=self.base.transform(d['base'])
        return dict(full=np.maximum.reduce([base,b['ca'],b['ay']]),
            command_zero=np.maximum.reduce([base,b['zero_ca'],b['zero_ay']]),
            predictor_free=np.maximum.reduce([base,b['free_ca'],b['free_ay']]),
            without_dispersion=np.maximum.reduce([base,p['ca'][:,0],p['ay'][:,0]]))

    def fit(self,regression,tail,branch,final):
        self.center={e:np.median(regression[e+'_R'],0) for e in self.keys}
        self.base=ExponentialTail().fit(tail['base'])
        self.location={e:ExponentialTail().fit(tail[e+'_L']) for e in self.keys}
        self.dispersion={e:ExponentialTail().fit(abs(tail[e+'_R']-self.center[e]).mean(1)) for e in self.keys}
        self.branch={e:ExponentialTail().fit(v.max(1)) for e,v in self.primitive(branch).items()}
        self.final={k:ExponentialTail().fit(v) for k,v in self.raw(final).items()}
        return self

    def score(self,d):return {k:self.final[k].transform(v) for k,v in self.raw(d).items()}
