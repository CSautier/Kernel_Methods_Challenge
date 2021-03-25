import argparse
import numpy as np

from utils import train_test_split, load_X, load_Y, solve_svm_dual, solve_MKL, euclidean_proj_simplex

parser = argparse.ArgumentParser(description="Gapped K-mer SVM training.")
parser.add_argument("-k", "--kernels", type=str, nargs='+', help='Paths to kernels', required=True)
parser.add_argument("--split", type=float, help="Train-test split", required=False, default=0.8)
parser.add_argument("--lr", type=float, help="Learning rate", required=False, default=0.1)
parser.add_argument("--n-iter", type=int, help="Number of iterations", required=False, default=5)
parser.add_argument("--test", help="Whether to use all data for training.", action='store_true')
args = parser.parse_args()

def main():

    for m in range(3):
        Y_paths = [f'data/Ytr{i}.csv' for i in range(m,m+1)]
        Y = load_Y(Y_paths)
        Y[Y==0] = -1

        lr = args.lr
        n_iter = args.n_iter

        kernels = [np.load(p)[m*2000:(m+1)*2000,m*2000:(m+1)*2000] for p in args.kernels]
        nk = len(kernels)
        n = len(Y)

        f = open(f'logs-{m}.txt', 'w')

        for s in [0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78,
                  0.80]:

            K_list = []
            for K in kernels:
                D = np.sqrt(np.diag(K))
                K = np.divide(K, D[:,None])
                K = np.divide(K, D[None, :])
                if s>0:
                    K = np.exp((K-1)/(s**2))
                K_list.append(K)

            for l in [1e-4]:
                print(f"s: {s}, l:{l}")
                acc_mean = 0

                for p in range(10):

                    a,b = p*n//10, (p+1)*n//10 if p!=9 else None
                    tids = np.ones(n).astype(bool)
                    tids[a:b] = False
                    Y_train, Y_val = Y[tids], Y[~tids]
                    Kt_arr = np.array([K[tids][:, tids] for K in K_list])
                    Kv_arr = np.array([K[~tids][:, tids] for K in K_list])

                    eta = np.ones(nk)/nk

                    for k in range(n_iter):

                        Kt = np.sum(eta[:,None,None]*Kt_arr, axis=0)
                        gamma, res = solve_MKL(Kt, Y_train, l)
                        grad = -l*gamma @ Kt_arr @ gamma
                        eta_argmax = np.argmax(eta)
                        grad_max = grad[eta_argmax]
                        D = grad - grad_max
                        D[eta_argmax] = np.sum(grad_max - grad)
                        eta -= lr*D
                        w = gamma
                        Kv = np.sum(eta[:,None,None]*Kv_arr, axis=0)
                        Y_pred = Kv.dot(w)
                        acc = np.sum(np.sign(Y_val)==np.sign(Y_pred))/len(Y_val)
                        print(f"{a}-{b}: {round(acc, 4)}")

                    acc_mean += acc

                acc_mean /= 10
                acc_mean = round(acc_mean, 4)

                print(f"s: {s}, l:{l}: {acc_mean} \n.")
                f.write(' '.join(map(str, [s, l, acc_mean])) + '\n')

        f.close()


if __name__ == "__main__":
    main()
