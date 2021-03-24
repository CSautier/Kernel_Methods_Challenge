import cvxpy as cp
import numpy as np

from utils import load_Y, solve_svm_dual

def main():

    paths = [
        ['k6-9_test.npy'],
        ['K6_test.npy', 'k6-9_test.npy'],
        ['k6-8_test.npy', 'k6-9_test.npy'],
    ]
    Y_list = []
    s_list = [0.75, 0.45, 0.66]
    for m, (s, path) in enumerate(zip(s_list, paths)):

        K = np.load(path[0])
        D = np.sqrt(np.diag(K))
        K = np.divide(K, D[:,None])
        K = np.divide(K, D[None, :])

        if len(path)>1:
            K1 = np.load(path[1])
            D = np.sqrt(np.diag(K1))
            K1 = np.divide(K1, D[:,None])
            K1 = np.divide(K1, D[None, :])

        w = np.load(f'w-{m}.npy')
        v = m*1000+6000
        t = m*2000
        _K = np.exp((K-1)/(s**2))
        _K = _K[v:v+1000,t:t+2000]

        if len(path)>1:
            _K1 = np.exp((K1-1)/(s**2))
            _K1 = _K1[v:v+1000,t:t+2000]
            _K = 0.5*_K + 0.5*_K1

        Y_pred = _K.dot(w)
        Y_pred[Y_pred<0] = 0
        Y_pred[Y_pred>0] = 1
        Y_list.append(Y_pred)

    Y_pred = np.concatenate(Y_list)
    ind = np.arange(3000)
    res = np.vstack((ind,Y_pred)).T.astype(int)
    np.savetxt(
        f"data/Yte.csv",
        res,
        delimiter=',',
        fmt='%d'
    )

if __name__ == "__main__":
    main()
