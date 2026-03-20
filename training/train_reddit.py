# train_reddit.py (PRODUCTION)

import os, torch, numpy as np, logging
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import Reddit2
from torch_geometric.utils import to_undirected, subgraph, train_test_split_edges
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)

H=512; E=128; L=4; NEG=5; EPOCHS=200; WD=1e-4

class GNN(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.convs=nn.ModuleList([GCNConv(dim,H)] +
            [GCNConv(H,H) for _ in range(L-2)] +
            [GCNConv(H,E)])
        self.cls=nn.Linear(E,4)
    def encode(self,x,e):
        for c in self.convs: x=F.relu(c(x,e))
        return x
    def forward(self,x,e,p,n):
        z=self.encode(x,e)
        pred=lambda ed: torch.sigmoid((z[ed[0]]*z[ed[1]]).sum(1))
        return z,self.cls(z),torch.cat([pred(p),pred(n)])

def load():
    d=Reddit2(root="/tmp/reddit")[0]
    idx=torch.randperm(d.num_nodes)[:30000]
    ei,_=subgraph(idx,d.edge_index,relabel_nodes=True)
    return train_test_split_edges(
        d.__class__(x=(d.x[idx]-d.x.mean(0))/d.x.std(0),
                    edge_index=to_undirected(ei),
                    y=(d.y[idx]%4),num_nodes=len(idx)))

def train():
    d=load(); m=GNN(d.x.size(1)).cuda()
    opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=WD)
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad()
        x,e=d.x.cuda(),d.train_pos_edge_index.cuda()
        pos=e; neg=torch.randint(0,d.num_nodes,(2,pos.size(1)*NEG)).cuda()
        z,logits,lp=m(x,e,pos,neg)
        loss=F.cross_entropy(logits,d.y.cuda())+F.binary_cross_entropy(lp,
            torch.cat([torch.ones(pos.size(1)),torch.zeros(neg.size(1))]).cuda())
        loss.backward(); opt.step()

    torch.save(m.state_dict(),"weights/model_weights_reddit.pth")
    np.save("weights/embeddings_reddit.npy",
            m.encode(x,e).cpu().detach().numpy())
    print("Training successfully completed!")
if __name__=="__main__": train()