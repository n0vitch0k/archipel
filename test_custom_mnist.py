import sys, time, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path.cwd()/'archipel'/'src'))
from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.current.courant import Courant
from archipel.current.topk_curriculum import TopKCurriculum, RoutingUsageTracker
from archipel.utils.specialization_matrix import compute_specialization_matrix_with_predictions, specialization_score_precision_weighted

class MNISTEncoder(nn.Module):
    def __init__(self,out_dim=128):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.fc=nn.Linear(32*7*7,out_dim)
    def forward(self,x): return self.fc(self.conv(x).flatten(1))

class MNISTArchipel(ArchipelPhase2):
    def __init__(self):
        super().__init__(num_islands=4,input_dim=128,hidden_dim=64,ocean_dim=128,top_k=1,max_islands=8,min_islands=2,coherence_variance_threshold=0.3)
        self.encoder=nn.Identity()
        self.mnist_enc=MNISTEncoder(out_dim=128)
    def forward(self,x,t=None,targets=None): return super().forward(self.mnist_enc(x), targets=targets if targets is not None else t)
    def kill_island(self,i,distill=True,dl=None): return super().kill_island(i,distill=distill,dataloader=dl if dl else self._dataloader_for_distillation,encoder=self.mnist_enc)

def evaluate(model,loader):
    model.eval()
    c=t=0
    with torch.no_grad():
        for x,y in loader:
            p=model(x)['output'].argmax(1)
            c+=(p==y).sum().item()
            t+=y.numel()
    model.train()
    return c/t

torch.manual_seed(42)
tr=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train=Subset(datasets.MNIST('./data', train=True, download=True, transform=tr), list(range(100*128)))
test=Subset(datasets.MNIST('./data', train=False, download=True, transform=tr), list(range(1024)))
loader=DataLoader(train,batch_size=128,shuffle=True,num_workers=0)
test_ld=DataLoader(test,batch_size=512,shuffle=False,num_workers=0)

model=MNISTArchipel()
model._dataloader_for_distillation=loader
opt=optim.Adam(model.parameters(),lr=1e-3)
curriculum=TopKCurriculum(num_islands=4,k_init=3,k_final=1,warmup_steps=6)
tracker=RoutingUsageTracker(num_islands=4)

logs,_=train_loop_lifecycle(model,loader,opt,Courant(4),epochs=1,device='cpu',log_every=50,top_k_curriculum=curriculum,routing_usage_tracker=tracker)
acc=evaluate(model,test_ld)
mat,preds,targets=compute_specialization_matrix_with_predictions(model,test_ld)
func=specialization_score_precision_weighted(mat,preds,targets)
last=logs[-1]
print('islands',model.num_islands,'births',sum(l.get('event')=='birth' for l in logs),'deaths',sum(l.get('event')=='death' for l in logs),'acc',round(acc,4),'loss',round(last['loss'],4),'func',round(func,4),'spec_coverage',last['spec_coverage'],'spec_islands',last['specialized_island_count'],'entropy',round(last['routing_usage_entropy'],4),'min_usage',round(last['min_usage_ratio'],4))
