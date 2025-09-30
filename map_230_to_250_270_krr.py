#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_230_to_250_270_krr.py
- 5쌍 학습용: (230_i, 250_i, 270_i), i=1..5
- 새로운 230_new가 들어오면 → (250_pred, 270_pred) 추정.
- 입력 CSV 예시(각 5행):
    train_230.csv / train_250.csv / train_270.csv
    각 행: x1,y1,x2,y2,...  (같은 포인트 수)
- 옵션: 포인트 수가 다르면 자동 호장 재표본으로 통일.
"""

import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

DEGREE = 3
SPLINE_SAMPLES = 400
N_CTRL = None  # None이면 자동 통일

# ----- B-spline -----
def open_uniform_knot_vector(n_ctrl, degree):
    knots = np.concatenate([
        np.zeros(degree + 1),
        np.arange(1, n_ctrl - degree),
        np.full(degree + 1, n_ctrl - degree)
    ])
    return knots / np.max(knots)

def bspline_basis(i, degree, knots, t):
    if degree == 0:
        last = (i + 1 == len(knots) - 1)
        if (knots[i] <= t < knots[i+1]) or (last and np.isclose(t, knots[i+1])):
            return 1.0
        return 0.0
    v=0.0
    d1=knots[i+degree]-knots[i]
    if d1>1e-12: v+=(t-knots[i])/d1*bspline_basis(i,degree-1,knots,t)
    d2=knots[i+degree+1]-knots[i+1]
    if d2>1e-12: v+=(knots[i+degree+1]-t)/d2*bspline_basis(i+1,degree-1,knots,t)
    return v

def bspline_curve(ctrl, degree, knots, ts):
    out = np.zeros((len(ts), 2), float)
    for j,t in enumerate(ts):
        wsum = np.zeros(2)
        for i in range(len(ctrl)):
            w = bspline_basis(i, degree, knots, t)
            if w>0: wsum += w*ctrl[i]
        out[j]=wsum
    return out

# ----- 유틸 -----
def read_matrix(path):
    M=[]
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln or ln.startswith("#"): continue
            toks=[t for t in ln.replace(",", " ").split() if t]
            vals=list(map(float,toks))
            if len(vals)<2: continue
            M.append(np.array(vals,float))
    if not M: raise RuntimeError(f"no rows in {path}")
    return M  # list of 1D arrays (flattened)

def arclen_resample(poly, n):
    P=poly.reshape(-1,2)
    seg=np.linalg.norm(np.diff(P,axis=0),axis=1)
    u=np.zeros(len(P)); u[1:]=np.cumsum(seg)
    L=u[-1]
    if L<=1e-9: return np.repeat(P[:1], n, axis=0).reshape(-1)
    u/=L
    s=np.linspace(0,1,n,endpoint=True)
    x=np.interp(s,u,P[:,0]); y=np.interp(s,u,P[:,1])
    return np.stack([x,y],axis=1).reshape(-1)

def unify_counts_list(list_of_flat, n_ctrl=None):
    if n_ctrl is None:
        # 최빈 포인트 수
        counts={}
        for v in list_of_flat:
            counts[len(v)//2]=counts.get(len(v)//2,0)+1
        n_ctrl=max(counts,key=counts.get)
    out=[]
    for v in list_of_flat:
        if len(v)//2 != n_ctrl:
            v = arclen_resample(v, n_ctrl)
        out.append(v)
    return n_ctrl, out

def to_ctrl(v): return v.reshape(-1,2)

def plot_pair(ctrl230, ctrlT, title):
    fig, ax = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
    for k,(ctrl,name) in enumerate([(ctrl230,"230 input"), (ctrlT,title)]):
        ax[k].set_aspect("equal","datalim"); ax[k].grid(True,alpha=0.3); ax[k].invert_yaxis()
        K=open_uniform_knot_vector(len(ctrl), DEGREE)
        t=np.linspace(0,1,SPLINE_SAMPLES,endpoint=False)
        C=bspline_curve(ctrl,DEGREE,K,t)
        ax[k].plot(C[:,0],C[:,1],lw=2); ax[k].scatter(ctrl[:,0],ctrl[:,1],s=18)
        ax[k].set_title(name)
    plt.show()

def main():
    # 학습 데이터 파일(각 5행 가정)
    p230="train_230.csv"; p250="train_250.csv"; p270="train_270.csv"
    if not all(os.path.isfile(p) for p in [p230,p250,p270]):
        print("학습 CSV가 필요합니다:\n - train_230.csv\n - train_250.csv\n - train_270.csv\n각 행: x1,y1,x2,y2,...  (5행 권장)")
        sys.exit(1)

    X_list = read_matrix(p230)
    Y250_list = read_matrix(p250)
    Y270_list = read_matrix(p270)

    # 포인트 수 통일
    n230, X_list = unify_counts_list(X_list, N_CTRL)
    n250, Y250_list = unify_counts_list(Y250_list, n230)  # 230과 동일 개수로 맞춤
    n270, Y270_list = unify_counts_list(Y270_list, n230)

    X = np.vstack(X_list)                # (5, 2N)
    Y250 = np.vstack(Y250_list)          # (5, 2N)
    Y270 = np.vstack(Y270_list)          # (5, 2N)

    # 커널 릿지 회귀(RBF): α는 정규화(스무딩), γ는 길이척도(=1/(2ℓ²))
    krr250 = KernelRidge(alpha=1e-1, kernel="rbf", gamma=1e-4)
    krr270 = KernelRidge(alpha=1e-1, kernel="rbf", gamma=1e-4)
    krr250.fit(X, Y250)
    krr270.fit(X, Y270)

    # 새로운 230 입력 파일
    new230_path = "new_230.csv"
    if not os.path.isfile(new230_path):
        print(f"새로운 230 컨트롤포인트 파일이 필요합니다: {new230_path}\n(한 행: x1,y1,x2,y2,...)")
        sys.exit(0)

    new230 = np.array(read_matrix(new230_path)[0])
    if len(new230)//2 != n230:
        new230 = arclen_resample(new230, n230)

    pred250 = krr250.predict(new230[None,:])[0]
    pred270 = krr270.predict(new230[None,:])[0]

    # 저장
    with open("pred_250.csv","w",encoding="utf-8",newline="") as f:
        csv.writer(f).writerow([f"{v:.6f}" for v in pred250])
    with open("pred_270.csv","w",encoding="utf-8",newline="") as f:
        csv.writer(f).writerow([f"{v:.6f}" for v in pred270])
    print("[OK] saved → pred_250.csv, pred_270.csv")

    # 시각화
    plot_pair(to_ctrl(new230), to_ctrl(pred250), "250 predicted")
    plot_pair(to_ctrl(new230), to_ctrl(pred270), "270 predicted")

if __name__ == "__main__":
    main()
