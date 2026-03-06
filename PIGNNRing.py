"""
PIGNNRing
Physics-Informed Graph Neural Network for Inverse Design of Integrated Photonic Biosensors

This script demonstrates a simplified Physics-Informed Graph Neural Network (PI-GNN)
for inverse design of a microring resonator biosensor operating near 1550 nm.

The photonic structure is represented as a graph of interacting regions
(ring, bus, coupler, analyte, substrate). A GNN surrogate learns the mapping
between geometry parameters and spectral response, enabling efficient
inverse design while respecting physical constraints.

Reference
---------
Torabi, Y., Ekhteraei, A., & Khajezadeh, M. (2026).
Physics-Informed Graph Neural Network for Inverse Design of Integrated Photonic Biosensors.
arXiv: https://arxiv.org/abs/2602.19082
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Setup
# =========================================================

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Physical constants and design targets
# =========================================================

LAMBDA_TARGET_NM = 1550.0

N0 = 1.3109
N_CORE = 1.57

# Geometry ranges (µm)
R_RANGE_UM = (0.0, 500.0)
W_RANGE_UM = (1.5, 3.0)
H_RANGE_UM = (1.5, 3.0)
G_RANGE_UM = (0.15, 0.30)

# Target geometry
TARGET_R_UM = 200.0
TARGET_W_UM = 2.0
TARGET_H_UM = 2.0
TARGET_G_UM = 0.25

# Sensitivity
S_NM_PER_RIU = 60.0

TARGET_DLAM_NM = 0.01
DELTA_N = TARGET_DLAM_NM / S_NM_PER_RIU
N1 = N0 + DELTA_N


# =========================================================
# Physics proxy model
# =========================================================

def neff_model(R, w, h, g, n_clad):
    """
    Smooth proxy model for effective refractive index.

    neff increases with waveguide confinement and decreases
    with larger coupling gaps.
    """

    w01 = (w - W_RANGE_UM[0]) / (W_RANGE_UM[1] - W_RANGE_UM[0] + 1e-9)
    h01 = (h - H_RANGE_UM[0]) / (H_RANGE_UM[1] - H_RANGE_UM[0] + 1e-9)
    g01 = (g - G_RANGE_UM[0]) / (G_RANGE_UM[1] - G_RANGE_UM[0] + 1e-9)

    overlap = 0.18 + 0.28*(1-w01) + 0.28*(1-h01) + 0.20*(1-g01)
    overlap = torch.clamp(overlap, 0.12, 0.88)

    neff = (1-overlap)*N_CORE + overlap*n_clad

    # bending correction
    neff = neff - 2e-4/(R/200.0 + 1e-6)

    return neff


def resonance_lambda(R, neff, m):
    """Microring resonance condition."""
    return (2 * math.pi * (R*1000.0) * neff) / m


# =========================================================
# Graph definition
# =========================================================

NODE_TYPES = {
    "ring":0,
    "bus":1,
    "coupler":2,
    "analyte":3,
    "substrate":4
}

EDGE_LIST = [
    ("ring","coupler"),("coupler","ring"),
    ("bus","coupler"),("coupler","bus"),
    ("ring","analyte"),("analyte","ring"),
    ("ring","substrate"),("substrate","ring")
]


def build_graph(R, w, h, g, n_clad):
    """
    Construct graph tensors for the photonic structure.
    """

    onehot = torch.eye(5, device=R.device)

    globals_ = torch.stack([R, w, h, g, n_clad]).view(1,5).repeat(5,1)

    x = torch.cat([onehot, globals_], dim=1)

    edge_index = torch.tensor(
        [[NODE_TYPES[u] for (u,v) in EDGE_LIST],
         [NODE_TYPES[v] for (u,v) in EDGE_LIST]],
        device=R.device
    )

    w01 = (w-W_RANGE_UM[0])/(W_RANGE_UM[1]-W_RANGE_UM[0])
    h01 = (h-H_RANGE_UM[0])/(H_RANGE_UM[1]-H_RANGE_UM[0])
    g01 = (g-G_RANGE_UM[0])/(G_RANGE_UM[1]-G_RANGE_UM[0])

    k_cpl = torch.clamp(1-g01,0,1)
    k_an  = torch.clamp(1-0.5*(w01+h01),0,1)
    k_sub = torch.tensor(0.30,device=R.device)

    edge_attr = []

    for (u,v) in EDGE_LIST:

        if "coupler" in (u,v):
            et = torch.tensor([1.,0.,0.],device=R.device)
            k = k_cpl

        elif "analyte" in (u,v):
            et = torch.tensor([0.,1.,0.],device=R.device)
            k = k_an

        else:
            et = torch.tensor([0.,0.,1.],device=R.device)
            k = k_sub

        edge_attr.append(torch.cat([et,k.view(1)],dim=0))

    edge_attr = torch.stack(edge_attr)

    return x, edge_index, edge_attr


# =========================================================
# PI-GNN Model
# =========================================================

class PIGNN(nn.Module):

    def __init__(self, hidden=64, steps=2):

        super().__init__()

        self.steps = steps

        self.node_embed = nn.Linear(10, hidden)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden*2 + 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.node_update = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )


    def forward(self, x, edge_index, edge_attr):

        h = self.node_embed(x)

        for _ in range(self.steps):

            src = edge_index[0]
            dst = edge_index[1]

            m = self.edge_mlp(torch.cat([h[dst],h[src],edge_attr],dim=1))

            agg = torch.zeros_like(h)
            agg.index_add_(0,dst,m)

            h = self.node_update(torch.cat([h,agg],dim=1))

        hg = h.mean(dim=0,keepdim=True)

        out = self.head(hg).squeeze(0)

        return out


# =========================================================
# Dataset generation
# =========================================================

def sample(lo, hi, n):
    return lo + (hi-lo)*torch.rand(n)


def make_dataset(N=2500):

    R = sample(*R_RANGE_UM, N)
    w = sample(*W_RANGE_UM, N)
    h = sample(*H_RANGE_UM, N)
    g = sample(*G_RANGE_UM, N)

    neff = neff_model(R,w,h,g,torch.tensor(N0))

    lam0 = resonance_lambda(R,neff,1)

    X = torch.stack([R,w,h,g],dim=1)

    y = torch.stack([lam0,neff,neff],dim=1)

    return X,y


# =========================================================
# Training
# =========================================================

def train():

    X,y = make_dataset()

    model = PIGNN().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(80):

        loss_sum = 0

        for i in range(len(X)):

            R,w,h,g = X[i].to(DEVICE)

            xg,ei,ea = build_graph(R,w,h,g,torch.tensor(N0,device=DEVICE))

            pred = model(xg,ei,ea)

            loss = F.mse_loss(pred, y[i].to(DEVICE))

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

        if epoch % 10 == 0:

            print(f"epoch {epoch}  loss {loss_sum:.3e}")

    return model


# =========================================================
# Inverse design
# =========================================================

def inverse_design(model, steps=200):

    z = torch.zeros(4,device=DEVICE,requires_grad=True)

    opt = torch.optim.Adam([z],lr=0.05)

    def bound(v,lo,hi):
        return lo+(hi-lo)*torch.sigmoid(v)

    for step in range(steps):

        R = bound(z[0],*R_RANGE_UM)
        w = bound(z[1],*W_RANGE_UM)
        h = bound(z[2],*H_RANGE_UM)
        g = bound(z[3],*G_RANGE_UM)

        xg,ei,ea = build_graph(R,w,h,g,torch.tensor(N0,device=DEVICE))

        lam0,_,_ = model(xg,ei,ea)

        loss = (lam0-LAMBDA_TARGET_NM)**2

        opt.zero_grad()
        loss.backward()
        opt.step()

    return R.item(),w.item(),h.item(),g.item()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print("Training PI-GNN surrogate")

    model = train()

    print("Running inverse design")

    sol = inverse_design(model)

    print("Optimized geometry (µm)")

    print("R, w, h, g =", sol)
