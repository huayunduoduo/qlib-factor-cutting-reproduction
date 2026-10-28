"""Ordinary GRU and dual-branch difference GRU; torch is an optional dependency."""
import numpy as np
import torch
from torch import nn

def masks(x):
    # Explicit tie rule: High >= median; Low < median. Keep the full 20-step axis.
    # Use quantile for the arithmetic median of an even-length sequence.
    high=x[:,:,0]>=torch.quantile(x[:,:,0],.5,dim=1,keepdim=True)
    return high.unsqueeze(-1), (~high).unsqueeze(-1)

class SequenceModel(nn.Module):
    def __init__(self,dual=False,hidden=64,layers=2):
        super().__init__();self.dual=dual
        self.high=nn.GRU(2,hidden,layers,batch_first=True)
        if dual:self.low=nn.GRU(2,hidden,layers,batch_first=True)
        self.output=nn.Linear(hidden,1)
    def forward(self,x,raw=None):
        if self.dual:
            h,l=masks(x if raw is None else raw)
            a,_=self.high(x*h);b,_=self.low(x*l);z=a[:,-1]-b[:,-1]
        else:z=self.high(x)[0][:,-1]
        return self.output(z).squeeze(-1)

def ic_loss(score,y):
    score=score-score.mean();y=y-y.mean()
    return -(score*y).sum()/(score.square().sum()*y.square().sum()).sqrt().clamp_min(1e-8)

def fit_predict(train,valid,test,epochs=100,seed=2026,dual=False,patience=20,checkpoint=None):
    """Lists of (date, index, x[N,20,2], y[N]); whole-date batches for IC loss."""
    torch.manual_seed(seed);torch.set_num_threads(2)
    np.random.seed(seed)
    model=SequenceModel(dual)
    allx=np.concatenate([z[2].reshape(-1,2) for z in train])
    med=np.median(allx,axis=0);scale=np.median(np.abs(allx-med),axis=0)*1.4826
    scale=np.maximum(scale,1e-6)
    def tx(x):return torch.tensor(np.clip((x-med)/scale,-5,5),dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=.001);best=np.inf;wait=0;state=None;history=[]
    rng=np.random.default_rng(seed)
    from pathlib import Path
    checkpoint=Path(checkpoint) if checkpoint else None
    first_epoch=0
    if checkpoint and checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['seed']==seed and saved['dual']==dual
        np.testing.assert_array_equal(saved['median'],med)
        np.testing.assert_array_equal(saved['scale'],scale)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        best=saved['best'];wait=saved['wait'];state=saved['best_state'];history=saved['history']
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'])
        first_epoch=saved['epoch']+1
        print('RESUME next epoch',first_epoch+1,flush=True)
    for epoch in range(first_epoch,epochs):
        if wait>=patience:break
        model.train();train_loss=[]
        for k in rng.permutation(len(train)):
            _,_,x,y=train[k]
            opt.zero_grad();loss=ic_loss(model(tx(x),torch.tensor(x,dtype=torch.float32)),torch.tensor(y,dtype=torch.float32))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();train_loss.append(float(loss.detach()))
        model.eval();values=[]
        with torch.no_grad():
            for _,_,x,y in valid:
                values.append(float(ic_loss(model(tx(x),torch.tensor(x,dtype=torch.float32)),torch.tensor(y,dtype=torch.float32))))
        val=float(np.mean(values));print('EPOCH',epoch+1,'valid_negative_ic',val,flush=True);history.append({'epoch':epoch+1,'train_negative_ic':float(np.mean(train_loss)),'valid_negative_ic':val})
        if val<best:
            best=val;state={k:v.detach().clone() for k,v in model.state_dict().items()};wait=0
        else:wait+=1
        if checkpoint:
            temporary=checkpoint.with_suffix('.pending')
            torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'best':best,'wait':wait,'best_state':state,'history':history,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'median':med,'scale':scale,'seed':seed,'dual':dual},temporary)
            temporary.replace(checkpoint)
        if wait>=patience:break
    model.load_state_dict(state);model.eval();result=[]
    with torch.no_grad():
        for date,idx,x,y in test:
            result.append((idx,model(tx(x),torch.tensor(x,dtype=torch.float32)).numpy()))
    return result,{'state_dict':state,'median':med,'scale':scale,'history':history,'seed':seed,'dual':dual}
