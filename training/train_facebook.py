# train_facebook.py (PRODUCTION)

import os, time, logging, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, train_test_split_edges
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
FEATURE_DIM=128; HIDDEN_DIM=512; EMBEDDING_DIM=128
NUM_CLASSES=4; NUM_LAYERS=5; DROPOUT=0.4
LR=1e-3; WD=1e-4; EPOCHS=300; PATIENCE=40; NEG_RATIO=5
OUTPUT_DIR="weights"

def build_features(edge_index, n):
    deg=torch.bincount(edge_index[0], minlength=n).float().unsqueeze(1)
    return torch.cat([deg, torch.log1p(deg), torch.randn(n, FEATURE_DIM-2)*0.1], dim=1)

class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs=nn.ModuleList([GCNConv(FEATURE_DIM,HIDDEN_DIM)] +
            [GCNConv(HIDDEN_DIM,HIDDEN_DIM) for _ in range(NUM_LAYERS-2)] +
            [GCNConv(HIDDEN_DIM,EMBEDDING_DIM)])
        self.cls=nn.Linear(EMBEDDING_DIM,NUM_CLASSES)

    def encode(self,x,e):
        for c in self.convs:
            x=F.dropout(F.relu(c(x,e)),p=DROPOUT,training=self.training)
        return x

    def forward(self,x,e,pos,neg):
        z=self.encode(x,e)
        logits=self.cls(z)
        pred=lambda ed: torch.sigmoid((z[ed[0]]*z[ed[1]]).sum(1))
        return z,logits,torch.cat([pred(pos),pred(neg)])

def load():
    path="/kaggle/input/facebook-large-page-page-network/musae_facebook_edges.csv"
    if os.path.exists(path):
        import pandas as pd
        df=pd.read_csv(path)
        ei=torch.tensor([df["id_1"],df["id_2"]])
    else:
        logger.warning("Synthetic")
        ei=torch.randint(0,10000,(2,100000))
    ei=to_undirected(ei)
    n=ei.max().item()+1
    return Data(x=build_features(ei,n),edge_index=ei,y=torch.randint(0,NUM_CLASSES,(n,)),num_nodes=n)

def train():
    d=load(); d=train_test_split_edges(d)
    m=GNN().cuda() if torch.cuda.is_available() else GNN()
    opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)

    best=0; os.makedirs(OUTPUT_DIR,exist_ok=True)
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad()
        x,e=d.x.cuda(),d.train_pos_edge_index.cuda()
        pos=e; neg=torch.randint(0,d.num_nodes,(2,pos.size(1)*NEG_RATIO)).cuda()
        z,logits,lp=m(x,e,pos,neg)
        loss=F.cross_entropy(logits,d.y.cuda())+F.binary_cross_entropy(lp,
            torch.cat([torch.ones(pos.size(1)),torch.zeros(neg.size(1))]).cuda())
        loss.backward(); opt.step(); sch.step()

        if ep%10==0:
            m.eval(); z=m.encode(x,e)
            auc=roc_auc_score(np.r_[np.ones(pos.size(1)),np.zeros(pos.size(1))],
                              torch.cat([m(x,e,pos,pos)[2][:pos.size(1)],
                                         m(x,e,pos,pos)[2][pos.size(1):]]).cpu())
            if auc>best:
                best=auc; torch.save(m.state_dict(),f"{OUTPUT_DIR}/best.pth")

    m.load_state_dict(torch.load(f"{OUTPUT_DIR}/best.pth"))
    torch.save(m.state_dict(),f"{OUTPUT_DIR}/model_weights_facebook.pth")
    np.save(f"{OUTPUT_DIR}/embeddings_facebook.npy",
            m.encode(d.x.cuda(),d.train_pos_edge_index.cuda()).cpu().detach().numpy())
    print("Training successfully completed!")
if __name__=="__main__": train()