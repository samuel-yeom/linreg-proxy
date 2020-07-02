from __future__ import print_function

import numpy as np
import proxy, process
import argparse

np.set_printoptions(threshold=100, suppress=True)

'''
Prints information about the proxy described by alpha.

X, z (== gender_m or race_blk), and colnames are the outputs of the process
methods above. beta and alpha are 1-D arrays such that X.shape[1] ==
beta.shape[0] == alpha.shape[0]. Each entry in these two input arrays
corresponds to a column of X. beta describes the model coefficients, and alpha
describes the proxy coefficients.
'''
def print_proxy_info(X, z, colnames, beta, alpha, n_alpha=None, verbose=True):
    n_alpha = len(alpha) if n_alpha is None else n_alpha
    
    #deal with alpha-coefficients that didn't fully converge
    for i in range(len(alpha)):
        if abs(alpha[i]) < 1e-6:
            alpha[i] = 0.0
        if abs(alpha[i] - 1.0) < 1e-6:
            alpha[i] = 1.0
    
    association, influence = proxy.get_assoc_infl(X, z, beta, alpha)
    print('Association: {:.6f}'.format(association))
    print('Influence:   {:.6f}'.format(influence))
    
    cov = np.cov(z, X, rowvar=False)[0, 1:] #covariance between z and columns of X
    sort_idx = np.argsort(np.abs(alpha * beta))[::-1]
    
    print('{} of {} attributes'.format(np.count_nonzero(alpha), len(alpha)))

    if verbose:
        max_lencolname = max(map(len, colnames))
        fmt_string = 'alpha{:' + str(max_lencolname+2) + '} = {:.6f} (beta: {:>6.3f}, cov: {:>6.3f})'
        for i in sort_idx[:n_alpha]:
            if alpha[i] == 0.0:
                continue
            print(fmt_string.format('('+colnames[i]+')', alpha[i], beta[i], cov[i]))

def run(args):
    exact = not args.approx
    epsilon = args.epsilon
    
    if args.dataset == 'ssl':
        X, y, gender_m, race_blk, colnames = process.process_chicago_ssl()
    elif args.dataset == 'cc':
        X, y, gender_m, race_blk, colnames = process.process_communities()
    z = race_blk
    X = process.scale(X)
    
    print('Standard deviation of y: {:.6f}'.format(np.std(y)))
    model = process.train_model(X, y)
    beta = model.coef_
    
    epsilon_str = 'epsilon == {:.6f}'.format(epsilon)
    print('\n{:*^80}'.format(epsilon_str))
    
    alphas = proxy.find_proxy(X, z, beta, epsilon, exact, direction='both')
    print('Positively correlated proxy:')
    print_proxy_info(X, z, colnames, beta, alphas[0], n_alpha=5)
    print('\nNegatively correlated proxy:')
    print_proxy_info(X, z, colnames, beta, alphas[1], n_alpha=5)
    
    return X, y, z, colnames, beta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['ssl', 'cc'])
    parser.add_argument('-a', '--approx', action='store_true',
                        help='Run the approximate, rather than exact, proxy-finding procedure.')
    parser.add_argument('-e', '--epsilon', action='store', type=float, default=0.05,
                        help='Association threshold for finding proxies.')
    args = parser.parse_args()
    
    X, y, z, colnames, beta = run(args)
