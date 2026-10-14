"""Euclidean projection of centered four-class corrections onto observed support.

Zero is always a vertex. No labels, returns, dates or tunable trading thresholds
enter this numerical operation. A convex-combination witness and optimality gap
are returned for independent verification.
"""
import itertools
import numpy as np
from scipy.spatial import ConvexHull


def centered(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all() or x.shape[-1]!=4:raise ValueError('Finite four-class scores required')
    return x-x.mean(axis=-1,keepdims=True)


def closest_facet(y,vertices):
    """Closest point on a segment or triangle, including all its lower faces."""
    best=None
    for size in range(1,len(vertices)+1):
        for ids in itertools.combinations(range(len(vertices)),size):
            points=vertices[list(ids)];origin=points[0]
            if size==1:weights=np.ones(1)
            else:
                basis=(points[1:]-origin).T
                a=np.linalg.lstsq(basis,y-origin,rcond=None)[0];weights=np.r_[1-a.sum(),a]
                if weights.min() < -1e-12:continue
                weights=np.maximum(weights,0);weights/=weights.sum()
            projected=weights@points;distance=float(np.sum((projected-y)**2))
            if best is None or distance<best[0]:
                full=np.zeros(len(vertices));full[list(ids)]=weights;best=(distance,full)
    return best


def project(raw,observed):
    raw=centered(raw).reshape(4);observed=centered(np.asarray(observed,float).reshape(-1,4))
    points=np.vstack([np.zeros(4),observed]);_,singular,vt=np.linalg.svd(points,full_matrices=False)
    rank=int(np.sum(singular>1e-12*max(1.,singular[0]))) if len(singular) else 0
    if rank>3:raise ValueError('Centered corrections must have rank at most three')
    weights=np.zeros(len(points))
    if rank==0:weights[0]=1.
    else:
        basis=vt[:rank].T;coords=points@basis;y=raw@basis
        if rank==1:
            lo=int(np.argmin(coords[:,0]));hi=int(np.argmax(coords[:,0]));a=float(np.clip((y[0]-coords[lo,0])/(coords[hi,0]-coords[lo,0]),0,1))
            weights[lo]=1-a;weights[hi]+=a
        else:
            hull=ConvexHull(coords)
            inside=np.max(hull.equations[:,:-1]@y+hull.equations[:,-1])<=1e-12*max(1.,np.linalg.norm(y))
            found=False
            if inside:
                middle=coords.mean(axis=0)
                for ids in hull.simplices:
                    a=np.linalg.solve((coords[ids]-middle).T,y-middle);w=np.r_[1-a.sum(),a]
                    if w.min()>=-1e-10:
                        w=np.maximum(w,0);w/=w.sum();weights[:]=w[0]/len(points);weights[ids]+=w[1:];found=True;break
            if not found:
                best=None
                for ids in hull.simplices:
                    dist,w=closest_facet(y,coords[ids])
                    if best is None or dist<best[0]:best=(dist,ids,w)
                _,ids,w=best;weights[ids]=w
    result=weights@points;gap=float(max(0.,np.max((points-result)@(raw-result))))
    tolerance=1e-9*max(1.,np.linalg.norm(raw)*np.max(np.linalg.norm(points,axis=1)))
    if gap>tolerance:raise ValueError(f'Projection optimality failed: {gap} > {tolerance}')
    if np.linalg.norm(result)>np.linalg.norm(raw)+1e-9*max(1.,np.linalg.norm(raw)):
        raise ValueError('Zero-anchored projection increased correction norm')
    ids=np.flatnonzero(weights>0)
    return dict(raw=raw,correction=result,vertex_indices=ids,vertex_weights=weights[ids],rank=rank,
        context_rows=len(observed),distance=float(np.linalg.norm(raw-result)),optimality_gap=gap,
        numerical_tolerance=tolerance,zero_anchor_weight=float(weights[0]))
