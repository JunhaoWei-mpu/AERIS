"""One AY aggregation repair; unchanged matched neighbors and channel moments."""
import numpy as np
from scipy.spatial.distance import cdist
from src.models.tvt_matched_chain import MatchedChainCalibration
from src.models.tvt_exponential_tail import ExponentialTail


def ay_energy(reference,query,mode):
    assert mode in ('context','history')
    key='ay_context' if mode=='context' else 'history'
    coord=(query[key]-reference.mean[key])/reference.scale[key]
    means=[];medians=[];ratios=[]
    for start in range(0,len(coord),128):
        stop=min(start+128,len(coord));groups=query['groups'][start:stop]
        distance=cdist(coord[start:stop],reference.coordinates[key],'sqeuclidean')
        distance[groups[:,None]==reference.library['groups'][None,:]]=np.inf
        ids=np.argpartition(distance,reference.k-1,axis=1)[:,:reference.k]
        assert np.isfinite(np.take_along_axis(distance,ids,1)).all()
        mref=reference.library['ay_M'][ids];vref=reference.library['ay_V'][ids]
        center=np.median(mref,1);typical=np.median(vref,1)
        scale=np.maximum(typical+(1.4826*np.median(abs(mref-center[:,None]),1))**2,reference.epsilon)
        m=query['ay_M'][start:stop];v=query['ay_V'][start:stop]
        contribution=(v+(m-center)**2)/scale
        means.append(contribution.mean(1));medians.append(np.median(contribution,axis=1))
        ratios.append(np.log((v+reference.epsilon)/(typical+reference.epsilon)))
    return np.concatenate(means),np.concatenate(medians),np.concatenate(ratios)


class MedianResponseCalibration(MatchedChainCalibration):
    keys=('ca','ay','zero_ca','zero_ay')

    def dispersion_value(self,edge,values):
        contribution=abs(values-self.center[edge])
        return np.median(contribution,axis=1) if edge.endswith('ay') else contribution.mean(1)

    def primitive(self,d):
        return {e:np.stack([self.location[e].transform(d[e+'_L']),
            self.dispersion[e].transform(self.dispersion_value(e,d[e+'_R']))],1) for e in self.keys}

    def raw(self,d):
        p=self.primitive(d);b={e:self.branch[e].transform(v.max(1)) for e,v in p.items()}
        base=self.base.transform(d['base'])
        return dict(full=np.maximum.reduce([base,b['ca'],b['ay']]),
            command_zero=np.maximum.reduce([base,b['zero_ca'],b['zero_ay']]),
            without_dispersion=np.maximum.reduce([base,p['ca'][:,0],p['ay'][:,0]]),
            without_ay=np.maximum(base,b['ca']))

    def fit(self,regression,tail,branch,final):
        self.center={e:np.median(regression[e+'_R'],0) for e in self.keys}
        self.base=ExponentialTail().fit(tail['base'])
        self.location={e:ExponentialTail().fit(tail[e+'_L']) for e in self.keys}
        self.dispersion={e:ExponentialTail().fit(self.dispersion_value(e,tail[e+'_R'])) for e in self.keys}
        self.branch={e:ExponentialTail().fit(v.max(1)) for e,v in self.primitive(branch).items()}
        self.final={k:ExponentialTail().fit(v) for k,v in self.raw(final).items()}
        return self
