# train_twitter.py (PRODUCTION)

import os, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import SNAPDataset
from torch_geometric.utils import to_undirected, train_test_split_edges
from torch_geometric.nn import GATConv

H=256; E=128; HEADS=4; NEG=5; EPOCHS=250

class GAT(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.g1=GATConv(dim,H,heads=HEADS)
        self.g2=GATConv(H*HEADS,E,heads=1)
        self.cls=nn.Linear(E,4)
    def encode(self,x,e):
        return F.elu(self.g2(F.elu(self.g1(x,e)),e))
    def forward(self,x,e,p,n):
        z=self.encode(x,e)
        pred=lambda ed: torch.sigmoid((z[ed[0]]*z[ed[1]]).sum(1))
        return z,self.cls(z),torch.cat([pred(p),pred(n)])

def load():
    ds=SNAPDataset(root="/tmp/snap",name="ego-Twitter")
    edges=[]; offset=0
    for d in ds:
        edges.append(d.edge_index+offset); offset+=d.num_nodes
    ei=to_undirected(torch.cat(edges,1))
    n=offset
    x=torch.randn(n,128)
    deg=torch.bincount(ei[0],minlength=n).float()
    y=torch.bucketize(deg,deg.quantile(torch.tensor([.25,.5,.75])))
    return train_test_split_edges(
        torch_geometric.data.Data(x=x,edge_index=ei,y=y,num_nodes=n))

def train():
    d=load(); m=GAT(128).cuda()
    opt=torch.optim.Adam(m.parameters(),1e-3)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        x,e=d.x.cuda(),d.train_pos_edge_index.cuda()
        pos=e; neg=torch.randint(0,d.num_nodes,(2,pos.size(1)*NEG)).cuda()
        z,logits,lp=m(x,e,pos,neg)
        loss=F.cross_entropy(logits,d.y.cuda())+F.binary_cross_entropy(lp,
            torch.cat([torch.ones(pos.size(1)),torch.zeros(neg.size(1))]).cuda())
        loss.backward(); opt.step()

    torch.save(m.state_dict(),"weights/model_weights_twitter.pth")
    np.save("weights/embeddings_twitter.npy",
            m.encode(x,e).cpu().detach().numpy())
    print("Training successfully completed!")
if __name__=="__main__": train()