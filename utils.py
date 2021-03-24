import numpy as np
import cvxpy as cp

def train_test_split(X, Y, s):
    n = len(Y)
    ind = np.arange(n)
    np.random.shuffle(ind)
    return X[ind][:int(n*s)], X[ind][int(n*s):], Y[ind][:int(n*s)], Y[ind][int(n*s):]

def load_Y(paths):
    data = [
        list(map(
            lambda x: x.split(',')[1],
            open(path).read().splitlines()[1:]
        ))
        for path in paths
    ]
    return np.array(np.concatenate(data)).astype(float)

def load_X(paths, seqlen=120):
    data = []
    for path in paths:
        s = map(list, open(path).read().splitlines()[1:])
        s = [list(map(ord, l))+[0]*(seqlen - len(l)) for l in s]
        data.append(s)
    data = np.concatenate(data)
    return data.astype(np.uint8)
    #ords = [65, 67, 71, 84]
    #res = np.zeros((len(data), 4))
    #for i, u in enumerate(ords):
    #    res[res==u] = i
    #return res

def load_no_padding(paths):
    data = []
    for path in paths:
        s = map(list, open(path).read().splitlines()[1:])
        s = [np.array(list(map(ord, l))) for l in s]
        data+= s
    return data

def solve_svm_primal(K, y, lbd):
    n = len(K)
    xi, alph = cp.Variable(n), cp.Variable(n)
    constraints = [xi >= 1-cp.multiply(y, K@alph), xi >= 0]
    objective = cp.Minimize(cp.sum(xi)/n + lbd * cp.quad_form(alph, K))
    prob = cp.Problem(objective, constraints)
    res = prob.solve(verbose=True)
    return xi.value, alph.value, res

def solve_svm_dual(K, y, lbd):
    n = len(K)
    mu, nu = cp.Variable(n), cp.Variable(n)
    constraints = [mu >= 0, nu >= 0, mu + nu == 1/n]
    objective = cp.Maximize(cp.sum(mu) - cp.quad_form(cp.diag(y)@mu, K)/(4*lbd))
    prob = cp.Problem(objective, constraints)
    res = prob.solve(verbose=False)
    return mu.value, nu.value, res

def solve_MKL(K, y, lbd):
    n = len(K)
    gamma = cp.Variable(n)
    constraints = [2*lbd*cp.multiply(y, gamma)>=0, 2*lbd*cp.multiply(y, gamma)<=1]
    objective = cp.Maximize(2*lbd*cp.sum(cp.multiply(y, gamma)) - lbd * cp.quad_form(gamma, K))
    prob = cp.Problem(objective, constraints)
    res = prob.solve(verbose=False)
    return gamma.value, res

def euclidean_proj_simplex(v, s=1):
    n, = v.shape  # will raise ValueError if v is not 1-D
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho+1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w
