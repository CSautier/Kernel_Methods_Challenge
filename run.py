import numpy as np
import torch

from utils import load_X, load_Y, solve_svm_dual


def torch_binom(N, k):
    mask = N >= k
    N = (mask * N).type(torch.float32)
    k = (mask * k).type(torch.float32)
    a = torch.lgamma(N + 1) - torch.lgamma((N - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask


def gkm(X_i, X_j, k, l):
    pad = torch.all(X_i[:, None] != 0, dim=-1) * torch.all(X_j[None] != 0, dim=-1)
    M = torch.sum(X_i[:, None] != X_j[None], dim=-1)
    return torch_binom(l - M, k) * pad


def compute_kernel(k, l=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Computing kernel")
    K = torch.zeros((9000, 9000), device=device, dtype=torch.float32)
    for p in range(3):
        t, v = p*2000, 6000+p*1000
        X_paths = [f"data/Xt{t}{p}.csv" for t in ["r", "e"]]
        X = load_X(X_paths)
        X = torch.tensor(X, dtype=torch.float32, device=device)
        n, d = X.shape
        print(f"on data {p}...")
        if l is not None:
            for i in range(d - l):
                for j in range(d - l):
                    X_i, X_j = X[:, i : i + l], X[:, j : j + l]
                    _K = gkm(X_i, X_j, k, l)
                    K[t:t+2000, t:t+2000] += _K[:2000, :2000]
                    K[v:v+1000, t:t+2000] += _K[2000:, :2000]
                    K[t:t+2000, v:v+1000] += _K[:2000, 2000:]
                    K[v:v+1000, v:v+1000] += _K[2000:, 2000:]
        else:
            for i in range(d - k):
                for j in range(d - k):
                    X_i, X_j = X[:, i : i + k], X[:, j : j + k]
                    _K = torch.all(X_i[None] == X_j[:, None], dim=-1) * \
                         torch.all(X_i[None] != 0, dim=-1) * torch.all(X_j[:, None] != 0, dim=-1)
                    K[t:t + 2000, t:t + 2000] += _K[:2000, :2000]
                    K[v:v + 1000, t:t + 2000] += _K[2000:, :2000]
                    K[t:t + 2000, v:v + 1000] += _K[:2000, 2000:]
                    K[v:v + 1000, v:v + 1000] += _K[2000:, 2000:]
    return K.cpu().numpy()


def compute_kernel2(k=8, l=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Computing kernel")
    K = torch.zeros((9000, 9000), device=device, dtype=torch.float32)
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
                _K = gkm(X_i, X_j, k, l)
                K[t:t+2000, t:t+2000] += _K[:2000, :2000]
                K[v:v+1000, t:t+2000] += _K[2000:, :2000]
                K[t:t+2000, v:v+1000] += _K[:2000, 2000:]
                K[v:v+1000, v:v+1000] += _K[2000:, 2000:]
    return K.cpu().numpy()


def solve_svm(Ks):
    # Regularization parameter
    l = 1e-5
    Y_list = []

    for m, _K in enumerate(Ks):
        print(f"Parsing dataset {m}")

        Y_paths = [f"data/Ytr{m}.csv"]
        Y = load_Y(Y_paths)
        Y[Y == 0] = -1

        t = m * 2000
        Ktr = _K[t : t + 2000, t : t + 2000]

        print("Running dual SVM...")
        mu, nu, res = solve_svm_dual(Ktr, Y, l)
        a = np.diag(Y).dot(mu) / (2 * l)
        print("Predicting on test set... \n")
        v = m * 1000 + 6000
        Kte = _K[v : v + 1000, t : t + 2000]
        Y_pred = Kte.dot(a)
        Y_pred[Y_pred < 0] = 0
        Y_pred[Y_pred > 0] = 1

        Y_list.append(Y_pred)

    return np.concatenate(Y_list)


def main():

    K1 = compute_kernel(6,9)
    K2 = compute_kernel(6)
    K3 = compute_kernel(6,8)

    np.save("K1.npy", K1)
    np.save("K2.npy", K2)
    np.save("K3.npy", K3)
    # K1 = np.load("K1.npy")
    # K2 = np.load("K2.npy")
    # K3 = np.load("K3.npy")

    ###### Compute the kernel for m=0 #######
    D = np.sqrt(np.diag(K1))
    K1 = np.divide(K1, D[:, None])
    K1 = np.divide(K1, D[None, :])
    s = 0.75
    Km0 = np.exp((K1 - 1) / (s ** 2))

    ###### Compute the kernel for m=1 #######
    D = np.sqrt(np.diag(K2))
    K2 = np.divide(K2, D[:, None])
    K2 = np.divide(K2, D[None, :])
    s = 0.45
    _K1 = np.exp((K1 - 1) / (s ** 2))
    _K2 = np.exp((K2 - 1) / (s ** 2))
    Km1 = 0.5 * _K1 + 0.5 * _K2

    ###### Compute the kernel for m=2 #######
    D = np.sqrt(np.diag(K3))
    K3 = np.divide(K3, D[:, None])
    K3 = np.divide(K3, D[None, :])
    s = 0.66
    _K1 = np.exp((K1 - 1) / (s ** 2))
    _K3 = np.exp((K3 - 1) / (s ** 2))
    Km2 = 0.5 * _K1 + 0.5 * _K3

    Yte = solve_svm([Km0, Km1, Km2])

    print("Saving predictions...")
    res = np.vstack((np.arange(3000), Yte)).T.astype(int)
    np.savetxt(
        f"data/Yte.csv",
        res,
        header="Id,Bound",
        delimiter=",",
        fmt="%d",
        comments=''
    )


if __name__ == "__main__":
    main()

    """
    Note : Our best results were obtained with the kernels given by K1 = compute_kernel2(6,9) (as done here)
    and K2 = K3 = compute_kernel2(8,13) for the two others m, with s = 0.5 and 0.62 respectively.
    However, we later found a bug in this second function, and chose instead to produce this script to reproduce our
    best result, instead of the submitted one, as it is very close and (hopefully) glitch-free.
    """
