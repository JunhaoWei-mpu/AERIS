"""Frozen Horizon35 computation plus the required pooled-reference control."""
import numpy as np
from src.models.tvt_response_median import MedianResponseCalibration


def pooled_evidence(reference,query):
    result=reference.evidence(query,mode='pooled')
    median=np.empty(len(query['groups']),dtype=np.float64)
    ratio=np.empty_like(query['ay_V'])
    for group in np.unique(query['groups']):
        take=reference.library['groups']!=group;rows=query['groups']==group
        means=reference.library['ay_M'][take];variances=reference.library['ay_V'][take]
        center=np.median(means,0);typical=np.median(variances,0)
        scale=np.maximum(typical+(1.4826*np.median(abs(means-center),0))**2,reference.epsilon)
        m=query['ay_M'][rows];v=query['ay_V'][rows]
        contribution=(v+(m-center)**2)/scale
        np.testing.assert_allclose(contribution.mean(1),result['ay_L'][rows],rtol=1e-12,atol=1e-12)
        median[rows]=np.median(contribution,axis=1)
        ratio[rows]=np.log((v+reference.epsilon)/(typical+reference.epsilon))
    np.testing.assert_allclose(ratio,result['ay_R'],rtol=1e-12,atol=1e-12)
    result['ay_L']=median
    return {'free_'+k:v for k,v in result.items()}


class HorizonMonitorCalibration(MedianResponseCalibration):
    keys=('ca','ay','zero_ca','zero_ay','free_ca','free_ay')

    def raw(self,d):
        result=super().raw(d);primitive=self.primitive(d)
        ca=self.branch['free_ca'].transform(primitive['free_ca'].max(1))
        ay=self.branch['free_ay'].transform(primitive['free_ay'].max(1))
        result['predictor_free']=np.maximum.reduce([self.base.transform(d['base']),ca,ay])
        return result
