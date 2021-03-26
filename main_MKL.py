import argparse
import numpy as np

from utils import load_Y, solve_MKL

parser = argparse.ArgumentParser(description="Gapped K-mer SVM training.")
parser.add_argument("-k", "--kernels", type=str, nargs='+', help='Paths to kernels', required=True)
parser.add_argument("--split", type=float, help="Train-test split", required=False, default=0.8)
parser.add_argument("-s", "--sigma", type=float, nargs='+', help="Exponential parameter", required=False, default=1)
parser.add_argument("-l", type=float, help="Regularization parameter", required=False, default=5e-5)
parser.add_argument("--lr", type=float, help="Learning rate", required=False, default=0.1)
parser.add_argument("--n-iter", type=int, help="Number of iterations", required=False, default=5)
parser.add_argument("-m", type=int, help="Number of iterations", required=False, default=0)
parser.add_argument("--test", help="Whether to use all data for training.", action='store_true')
args = parser.parse_args()


def main():

    test = args.test
    m = args.m
    Y_paths = [f'data/Ytr{m}.csv']
    Y = load_Y(Y_paths)
    Y[Y==0] = -1

    n = len(Y)
    sigma = args.sigma
    l = args.l
    lr = args.lr
    n_iter = args.n_iter

    print(f"Running MKL for {n_iter} iterations with learning rate {lr} and penalty {l}.")

    # Loading data

    kernels = [np.load(p)[m*2000:(m+1)*2000,m*2000:(m+1)*2000] for p in args.kernels]
    nk = len(kernels)
    K_list = []
    for K, s in zip(kernels, sigma):
        D = np.sqrt(np.diag(K))
        K = np.divide(K, D[:,None])
        K = np.divide(K, D[None, :])
        if s>0:
            K = np.exp((K-1)/(s**2))
        K_list.append(K)

    if not test:
        # Splitting dataset into train and test set
        n = len(Y)
        ind = np.arange(n)
        np.random.shuffle(ind)
        p = int(n*args.split)
        Y_train, Y_val = Y[ind][:p], Y[ind][p:]
        Kt_arr = np.array([K[ind[:p]][:,ind[:p]] for K in K_list])
        Kv_arr = np.array([K[ind[p:]][:, ind[:p]] for K in K_list])
    else:
        # Using all the data for training
        Y_train = Y
        Kt_arr = np.array([K[:n,:n] for K in K_list])

    eta = np.ones(nk)/nk

    # Iteratively solve the MKL problem

    for k in range(n_iter):
        Kt = np.sum(eta[:,None,None]*Kt_arr, axis=0)
        gamma, res = solve_MKL(Kt, Y_train, l)
        grad = -l*gamma @ Kt_arr @ gamma
        eta_argmax = np.argmax(eta)
        grad_max = grad[eta_argmax]
        D = grad - grad_max
        D[eta_argmax] = np.sum(grad_max - grad)
        # Update the weights
        eta -= lr*D

        print(f"Weights: {eta}.")
        w = gamma
        if not test:
            print("Predicting on validation set.")
            Kv = np.sum(eta[:,None,None]*Kv_arr, axis=0)
            Y_pred = Kv.dot(w)
            print(Y_val[:10], Y_pred[:10])
            acc = np.sum(np.sign(Y_val)==np.sign(Y_pred))/len(Y_val)
            print(f"Accuracy: {round(acc, 4)}")
        else:
            print("Predicting on training set.")
            Kt = np.sum(eta[:, None, None] * Kt_arr, axis=0)
            Y_pred = Kt.dot(w)
            acc = np.sum(np.sign(Y_train) == np.sign(Y_pred)) / len(Y_train)
            print(f"Accuracy: {round(acc, 4)}")
    if test:
        np.save(f'w-{m}.npy', w)
    print('\n')


if __name__ == "__main__":
    main()
