import numpy as np
from scipy.linalg import cholesky

def get_assoc_infl(X, z, beta, alpha=None):
    '''
    Returns the association and influence of the component
        ```P == alpha_1 * beta_1 * X_1 + ... + alpha_d * beta_d * X_d```
    of the linear regression model
        ```model == beta_1 * X_1 + ... + beta_d * X_d```
    as defined in the paper "Hunting for Discriminatory Proxies in Linear
    Regression Models" by Yeom et al., 2018.

    Parameters
    ----------
    X : 2-D numpy array with shape `(n,d)`
        input features; rows are data points and columns are features
    z : 1-D numpy array with shape `(n,)`
        protected attribute
    beta : 1-D numpy array with shape `(d,)`
        coefficients of the model
    alpha : None or 1-D numpy array with shape `(d,)`, optional
        coefficients used to defined the component. The default is None.
        If None, the component is assumed to be the entire model.

    Returns
    -------
    association : float
        association of the component with the protected attribute
    influence : float
        influence of the component on the model

    '''
    d = X.shape[1] #this is called n in the paper
    if alpha is None:
        alpha = np.ones(d)
    
    assert X.shape[0] == z.shape[0]
    assert X.shape[1] == beta.shape[0] == alpha.shape[0]
    
    cov = np.cov(z, X, rowvar=False)
    xcov = cov[1:,1:]
    
    #compute influence
    model_variance = np.linalg.multi_dot((beta, xcov, beta))
    p_variance = np.linalg.multi_dot((alpha*beta, xcov, alpha*beta))
    influence = p_variance / model_variance
    
    #compute association
    cov_pz = np.dot(alpha*beta, cov[0,1:])
    z_variance = cov[0,0]
    association = cov_pz ** 2 / (p_variance * z_variance)
    
    return association, influence

def find_proxy(X, z, beta, epsilon, exact=True, direction='pos', verbose=False):
    '''
    Implements the proxy-finding procedures described in "Hunting for
    Discriminatory Proxies in Linear Regression Models" by Yeom et al., 2018.
    The optimizer tries to maximize the influence of the component subject to
    the constraint that its association must be at least `epsilon`.

    Parameters
    ----------
    X : 2-D numpy array with shape `(n,d)`
        input features; rows are data points and columns are features
    z : 1-D numpy array with shape `(n,)`
        protected attribute
    beta : 1-D numpy array with shape `(d,)`
        coefficients of the model
    epsilon : float
        association threshold; must be between 0 and 1
    exact : bool, optional
        If True, runs the exact optimization problem (Problem 1 in the paper).
        If False, runs the approximate optimization problem (Problem 2).
        The default is True.
    direction : {'pos', 'neg', 'both'}, optional
        If 'pos', searches for proxies that are positively correlated with `z`.
        If 'neg', searches for negatively correlated proxies.
        If 'both', searches for both.
    verbose : bool, optional
        If True, prints Gurobi's outputs. The default is False.

    Returns
    -------
    alphas : list of 1-D numpy arrays
        Each array has shape `(d,)` and represents the alpha-coefficients
        that characterize a proxy. If `direction` is 'pos' or 'neg', the list
        contains one array. If `direction` is 'both', the list contains two
        arrays, the first of which is positively correlated with `z` and the
        second of which is negatively correlated.
    '''
    import gurobipy
    from gurobipy import GRB
    
    assert 0 <= epsilon <= 1
    assert X.shape[0] == z.shape[0]
    assert X.shape[1] == beta.shape[0]
    assert direction in ['pos', 'neg', 'both']
    
    d = X.shape[1] #this is called n in the paper
    
    cov = np.cov(z, X, rowvar=False)
    
    #columns of the basis correspond to z, beta_1 * X_1, ..., beta_d * X_d
    basis = cholesky(cov) * np.concatenate((np.ones(1), beta))
    z_vector = basis[:,0]
    betaX_vectors = basis[:,1:] #A' in the paper
    
    if direction == 'pos':
        s_list = [1]
    elif direction == 'neg':
        s_list = [-1]
    elif direction == 'both':
        s_list = [1, -1]
    
    alphas = []
    for s in s_list:
        m = gurobipy.Model()
        m.Params.OutputFlag = 1 if verbose else 0
        m.Params.LogFile = '' #do not write log to file
        if exact:
            m.Params.NonConvex = 2 #allow non-convex quadratic objectives
            m.Params.TimeLimit = 30 #seconds
        
        alpha = m.addMVar(d, lb=0, ub=1)
        p_vector = m.addMVar(d+1, lb=-GRB.INFINITY)
        m.addConstr(p_vector == betaX_vectors @ alpha)
        rhs = m.addMVar(1, lb=0, ub=GRB.INFINITY)
        m.addConstr(s * np.sqrt(epsilon) * np.linalg.norm(z_vector) * rhs == z_vector @ p_vector)
        m.addConstr(p_vector @ p_vector <= rhs @ rhs)
        if exact:
            m.setObjective(p_vector @ p_vector, GRB.MAXIMIZE)
        else:
            m.setObjective(np.linalg.norm(betaX_vectors, axis=0) @ alpha, GRB.MAXIMIZE)
        m.optimize()
        
        alphas.append(alpha.X)
    
    return alphas

def find_proxy_cvxopt(X, z, beta, epsilon, exact=True, direction='pos', verbose=False):
    if exact:
        return cvxopt_exact(X, z, beta, epsilon, direction, verbose)
    else:
        return cvxopt_approx(X, z, beta, epsilon, direction, verbose)

def cvxopt_approx(X, z, beta, epsilon, direction='pos', verbose=False):
    from cvxopt import solvers, matrix
    
    assert 0 <= epsilon <= 1
    assert X.shape[0] == z.shape[0]
    assert X.shape[1] == beta.shape[0]
    assert direction in ['pos', 'neg', 'both']
    
    d = X.shape[1] #this is called n in the paper
    
    cov = np.cov(z, X, rowvar=False)
    xcov = cov[1:,1:]
    
    #each column of basis corresponds to z, beta_1 * X_1, ..., beta_d * X_d
    basis = cholesky(cov) * np.concatenate((np.ones(1), beta))
    
    #stretch (or compress) along the z-axis
    #to make the angle at the vertex of the "association cone" 90 degrees
    stretch_factor = np.sqrt((1-epsilon) / epsilon)
    
    if direction == 'pos':
        s_list = [1]
    elif direction == 'neg':
        s_list = [-1]
    elif direction == 'both':
        s_list = [1, -1]
    
    solvers.options['show_progress'] = verbose
    
    alphas = []
    for s in s_list:
        stretched_basis = np.copy(basis)
        stretched_basis[0] *= s * stretch_factor #reflect (or don't) along the z-axis
        
        c = np.abs(beta) * np.sqrt(xcov.diagonal()) #c_i == |beta_i| * std(X_i)
        G = np.concatenate((-np.eye(d), np.eye(d), -stretched_basis[:,1:]))
        h = np.concatenate((np.zeros(d), np.ones(d), np.zeros(d+1)))
        dims = {'l': 2*d, 'q': [d+1], 's': []}
        
        ans_dict = solvers.conelp(matrix(-c), matrix(G), matrix(h), dims)
        
        alpha = np.squeeze(np.array(ans_dict['x']))
        if ans_dict['status'] != 'optimal':
            if s == 1:
                print('Positively correlated proxy did not converge to the optimum\n')
            elif s == -1:
                print('Negatively correlated proxy did not converge to the optimum\n')
        
        alphas.append(alpha)
    
    return alphas

def cvxopt_exact(X, z, beta, epsilon, direction='pos', verbose=False):
    from cvxopt import solvers, matrix
    
    assert 0 <= epsilon <= 1
    assert X.shape[0] == z.shape[0]
    assert X.shape[1] == beta.shape[0]
    assert direction in ['pos', 'neg', 'both']
    
    d = X.shape[1] #this is called n in the paper
    
    cov = np.cov(z, X, rowvar=False)
    xcov = cov[1:,1:]
    
    #each column of basis corresponds to z, beta_1 * X_1, ..., beta_d * X_d
    basis = cholesky(cov) * np.concatenate((np.ones(1), beta))
    
    #stretch (or compress) along the z-axis
    #to make the angle at the vertex of the "association cone" 90 degrees
    stretch_factor = np.sqrt((1-epsilon) / epsilon)
    
    if direction == 'pos':
        s_list = [1]
    elif direction == 'neg':
        s_list = [-1]
    elif direction == 'both':
        s_list = [1, -1]
    
    solvers.options['show_progress'] = verbose
    
    alphas = []
    for s in s_list:
        stretched_basis = np.copy(basis)
        stretched_basis[0] *= s * stretch_factor #reflect (or don't) along the z-axis
        
        def F(*args): #negative of the influence of component P
            assert len(args) <= 2
            
            betaxcov = np.outer(beta, beta) * xcov
            if len(args) == 0:
                return (0, matrix(np.random.rand(d)))
            
            else:
                alpha = np.squeeze(np.array(args[0]))
                
                objective = -np.linalg.multi_dot((alpha, betaxcov, alpha))
                f = matrix(objective)
                
                d_objective = -np.dot(betaxcov + betaxcov.T, alpha)
                Df = matrix(d_objective.reshape((1, d)))
                
                if len(args) == 1:
                    return (f, Df)
                
                else:
                    z = args[1]
                    
                    dd_objective = -(betaxcov + betaxcov.T)
                    H = matrix(z[0] * dd_objective)
                    return (f, Df, H)
        
        G = np.concatenate((-np.eye(d), np.eye(d), -stretched_basis[:,1:]))
        h = np.concatenate((np.zeros(d), np.ones(d), np.zeros(d+1)))
        dims = {'l': 2*d, 'q': [d+1], 's': []}
        
        ans_dict = solvers.cp(F, matrix(G), matrix(h), dims)
        
        alpha = np.squeeze(np.array(ans_dict['x']))
        if ans_dict['status'] != 'optimal':
            if s == 1:
                print('Positively correlated proxy did not converge to the optimum\n')
            elif s == -1:
                print('Negatively correlated proxy did not converge to the optimum\n')
        
        alphas.append(alpha)
    
    return alphas
