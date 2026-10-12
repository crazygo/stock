"""Independent train-only interaction reconstruction and model normal equations."""
import numpy as np
from joint_learning import columns,pairs,cross_name
from audit_macro_models import weighted_equations as original_equations
from verify_reliability import close
from verify_continuation import near


def reference_expansion(panel,ids,model,info,clock,kind,w):
    out=panel.copy();n=len(ids)
    if kind=='additive' or n<120:
        assert 'basis' not in info
        return out,0
    cols=columns(clock,'additive');X=panel.loc[ids,cols].to_numpy(float)
    center=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1
    basis=model['basis'];assert list(map(tuple,basis['pairs']))==pairs(kind) and list(map(tuple,info['basis']['pairs']))==pairs(kind)
    for j,c in enumerate(cols):
        near(basis['center'][c],center[j]);near(basis['scale'][c],scale[j]);near(info['basis']['center'][c],center[j]);near(info['basis']['scale'][c],scale[j])
    # Independent row/mapping implementation; no production expanded() call.
    matrix=panel[cols].to_numpy(float);Z=np.divide(matrix-center,scale);pos={c:j for j,c in enumerate(cols)}
    for m,s in pairs(kind):out[cross_name(m,s)]=np.multiply(Z[:,pos[m]],Z[:,pos[s]])
    return out,len(cols)*4+len(pairs(kind))*len(panel)


def weighted_equations(panel,ids,at,model,info,cols,w,kind):
    expanded,n=reference_expansion(panel,ids,model,info,panel.loc[at,'clock'],kind,w)
    values,k=original_equations(expanded,ids,at,model,info,cols,w)
    return values,k,n
