"""
Niveau 1.4/1.5 — Etape 2 seulement : Baseline MLP non modulaire vs Archipel (2 époques, validation rapide).

Archipel est entraîné 2 époques (rapide, déjà validé à 95%+ accuracy).
MLP est entraîné 2 époques.
Évaluation sur test set officiel (10k) pour les deux.
Comparaison directe.
"""
import sys, time, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "archipel" / "src"))

from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.current.courant import Courant


# ─── Encodeur ───
class MNISTEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32*7*7,out_dim)
    def forward(self,x): return self.fc(self.conv(x).flatten(1))


# ─── Archipel ───
class MNISTArchipel(ArchipelPhase2):
    def __init__(self):
        super().__init__(num_islands=4,input_dim=128,hidden_dim=64,ocean_dim=128,
                         top_k=2,max_islands=8,min_islands=2,coherence_variance_threshold=0.3)
        self.encoder=nn.Identity(); self.mnist_enc=MNISTEncoder(out_dim=128)
    def forward(self,x,t=None): return super().forward(self.mnist_enc(x),targets=t)
    def kill_island(self,i,distill=True,dl=None):
        dl2=dl if dl else self._dataloader_for_distillation
        return super().kill_island(i,distill=distill,dataloader=dl2,encoder=self.mnist_enc)


# ─── Baseline MLP ───
class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=MNISTEncoder(out_dim=128)
        self.clf=nn.Sequential(nn.Linear(128,256),nn.ReLU(),
                                nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.clf(self.encoder(x))


# ─── Évaluation rapide ───
@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval(); corr=tot=0
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        if isinstance(model,ArchipelPhase2):
            preds=model(x)["output"].argmax(1)
        else:
            preds=model(x).argmax(1)
        corr+=(preds==y).sum().item(); tot+=y.size(0)
    model.train()
    return corr/tot


def main():
    device="cpu"; torch.manual_seed(123)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    mnist_train=datasets.MNIST("./data",train=True,download=True,transform=transform)
    mnist_test=datasets.MNIST("./data",train=False,download=True,transform=transform)
    train_ld=DataLoader(mnist_train,batch_size=64,shuffle=True,num_workers=0)
    test_ld=DataLoader(mnist_test,batch_size=512,shuffle=False,num_workers=0)

    epochs=2

    # ── Archipel ──
    print("="*55)
    print("ARCHIPEL — 2 époques MNIST (batch=64)")
    print("="*55)
    model=MNISTArchipel(); model.train()
    opt=optim.Adam(model.parameters(),lr=1e-3)
    courant=Courant(num_islands=4)
    model._dataloader_for_distillation=train_ld

    t0=time.time()
    logs,_=train_loop_lifecycle(model,train_ld,opt,courant,epochs=epochs,device=device,log_every=50)
    t1=time.time()
    acc_arch=evaluate(model,test_ld,device)
    b_arch=sum(1 for l in logs if l.get('event')=='birth')
    d_arch=sum(1 for l in logs if l.get('event')=='death')
    final_loss=logs[-1]['loss']
    print(f"Train: {t1-t0:.0f}s | Loss: {final_loss:.4f} | TestAcc: {acc_arch:.4f} | b:{b_arch} d:{d_arch}")

    # ── MLP Baseline ──
    print("\n"+"="*55)
    print("MLP BASELINE — 2 époques MNIST")
    print("="*55)
    mlp=MLPBaseline(); mlp.train()
    opt2=optim.Adam(mlp.parameters(),lr=1e-3)
    crit=nn.CrossEntropyLoss()
    t0=time.time()
    for ep in range(epochs):
        for xb,yb in train_ld:
            xb=xb.to(device); yb=yb.to(device)
            opt2.zero_grad(); loss=crit(mlp(xb),yb); loss.backward(); opt2.step()
        print(f"  Epoch {ep+1}/{epochs}  loss={loss.item():.4f}")
    t1=time.time()
    acc_mlp=evaluate(mlp,test_ld,device)
    print(f"Train: {t1-t0:.0f}s | Loss: {loss.item():.4f} | TestAcc: {acc_mlp:.4f}")

    # ── Comparaison ──
    print("\n"+"="*55)
    print("RÉSULTATS")
    print("="*55)
    print(f"{'':30} {'Archipel':>10} {'MLP':>10}")
    print(f"{'Test Accuracy':30} {acc_arch:>10.4f} {acc_mlp:>10.4f}")
    delta=acc_arch-acc_mlp
    if delta>0.01: print(f"\n✅ Archipel surpasse MLP de +{delta:.4f}")
    elif delta<-0.01: print(f"\n⚠️  MLP surpasse Archipel de {abs(delta):.4f}")
    else: print(f"\n→ Équivalents (delta={delta:+.4f})")

    out={"archipel":{"test_acc":round(acc_arch,4),"loss_final":round(final_loss,4),"births":b_arch,"deaths":d_arch,"train_time_s":round(t1-t0,1)},
         "mlp":{"test_acc":round(acc_mlp,4),"loss_final":round(loss.item(),4),"train_time_s":round(t1-t0,1)},
         "epochs":epochs}
    out_path=ROOT/"baseline_comparison.json"
    with open(out_path,'w') as f: json.dump(out,f,indent=2)
    print(f"\nRésultats: {out_path}")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--batch-size",type=int,default=64)
    args=parser.parse_args()
    main()
