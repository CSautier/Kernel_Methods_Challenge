import numpy as np
import torch

from utils import load_X

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def torch_binom(N, k):
    mask = N >= k
    N = (mask * N).type(torch.float32)
    k = (mask * k).type(torch.float32)
    a = torch.lgamma(N + 1) - torch.lgamma((N - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask


def gkm(X_i, X_j, k, l):
    pad = torch.all(X_i[:, None] != 0, dim=-1)*torch.all(X_j[None] != 0, dim=-1)
    M = torch.sum(X_i[:, None] != X_j[None], dim=-1)
    return torch_binom(l-M, k)*pad


def main():

    X_paths = [f'data/Xtr{i}.csv' for i in range(3)]
    X = load_X(X_paths)

    X2_paths = [f'data/Xte{i}.csv' for i in range(3)]
    X2 = load_X(X2_paths)
    X = np.concatenate((X,X2))

    X = torch.tensor(X, dtype=torch.float32, device=device)
    n, d = X.shape
    K = torch.zeros((n,n), device=device, dtype=torch.float32)
    k = 6
    l = None

    for p in range(3):
        t, v = p*2000, 6000+p*1000
        X_paths = [f"data/Xt{t}{p}.csv" for t in ["r", "e"]]
        X = load_X(X_paths)
        X = torch.tensor(X, dtype=torch.float32, device=device)
        n, d = X.shape
        print(f"on data {p}...")
        for i in range(d - k):
            for j in range(d - k):
                X_i, X_j = X[:, i : i + k], X[:, j : j + k]
                if l is not None:
                    _K = gkm(X_i, X_j, k, l)
                    K[t:t+2000, t:t+2000] += _K[:2000, :2000]
                    K[v:v+1000, t:t+2000] += _K[2000:, :2000]
                    K[t:t+2000, v:v+1000] += _K[:2000, 2000:]
                    K[v:v+1000, v:v+1000] += _K[2000:, 2000:]
                else:
                    _K = torch.all(X_i[None] == X_j[:, None], dim=-1) * \
                         torch.all(X_i[None] != 0, dim=-1) * torch.all(X_j[:, None] != 0, dim=-1)
                    K[t:t + 2000, t:t + 2000] += _K[:2000, :2000]
                    K[v:v + 1000, t:t + 2000] += _K[2000:, :2000]
                    K[t:t + 2000, v:v + 1000] += _K[:2000, 2000:]
                    K[v:v + 1000, v:v + 1000] += _K[2000:, 2000:]
            if i % 10 == 0:
                print(i)

    name = f"K{k}.npy"
    if l is not None:
      name = f"K{k}-{l}.npy"
    print(f"Saving in {name}.")
    np.save(name, K.cpu().numpy())


if __name__ == "__main__":
    main()
