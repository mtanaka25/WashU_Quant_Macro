#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW4.py

is the python class for the assignment #4 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Sep 20, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import time
from copy import deepcopy
from scipy.stats import norm
from scipy.optimize import fsolve, root, bisect

# =-=-=-=-=-=-=-=-= functions used in HW4 =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def plot_Q2_graph(model1, model2, a2plot, fname):
    if model1.n_grids_a != model2.n_grids_a:
        raise Exception('Two instances seem to have been solved in different ways.')
    
    a_idx = get_nearest_idx(a2plot, model1.a_grid)
    
    model1_data = model1.policy_func[:, a_idx]
    model2_data = model2.policy_func[:, a_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(model1.y_grid, model1_data, lw=1.5, c='blue', ls='dashed',
             label='Model w/ taxation')
    ax.plot(model2.y_grid, model2_data, lw=3.0, c='red', 
             label='Model w/o taxation')
    ax.set_ylabel("asset holding tomorrow (a')")
    ax.set_xlabel("income today (y)")
    ax.legend(frameon = False)
    fig.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0)

        
def get_nearest_idx(x, array):
    nearest_idx = np.abs(array - x).argmin()
    return nearest_idx
    

# =-=-=-=-=-=-=-=-= classes for interpolation =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class piecewise_intrpl: # pair-wised linear interpolation
    def __init__(self, x, fx):
        if x is list:
            x = np.array(x)
        if fx is list:
            fx = np.array(fx)
        
        self.x  = x
        self.fx = fx
    
    def __call__(self, x_new):
        if np.isscalar(x_new):
            fx_bar = self._fit_univariate(x_new)
        else:
            fx_bar = self._fit_multivariate(x_new)
        return fx_bar
    
    def _fit_univariate(self, x_new):
        j  = sum(self.x <= x_new)
        
        phi = np.zeros((len(self.x), ))
        if self.x[-1] <= x_new:
            phi[-1]   = (x_new - self.x[-2])/(self.x[-1] - self.x[-2])
            phi[-2] = (self.x[-1] -  x_new)/(self.x[-1] - self.x[-2])
        elif x_new <= self.x[0]:
            phi[1]   = (x_new - self.x[0])/(self.x[1] - self.x[0])
            phi[0] = (self.x[1] -  x_new)/(self.x[1] - self.x[0])
        else:
            phi[j]   = (x_new - self.x[j-1])/(self.x[j] - self.x[j-1])
            phi[j-1] = (self.x[j] -  x_new)/(self.x[j] - self.x[j-1])

        fx_bar = np.sum(phi * self.fx)
        
        return fx_bar

    def _fit_multivariate(self, x_new):
        fx_bar = [self._fit_univariate(x_new_i) for x_new_i in x_new]
        return fx_bar
    
    
class rbf_intrpl: # radial basis function interpolation
    def __init__(self, x, fx, eps=0.01):
        self.eps = eps
        self.x   = x
        self._solve_for_omega(x, fx)
              
    def __call__(self, x_new):
        if np.isscalar(x_new):
            fx_bar = self._fit_univariate(x_new)
        else:
            fx_bar = self._fit_multivariate(x_new)
        return fx_bar
    
    def _rbf(self, x_1, x_2=0): # radial basis function
        distance = abs(x_1 - x_2)
        phi = np.exp(- self.eps * distance**2)
        return phi
        
    def _solve_for_omega(self, x, fx):
        coef_mat = [
            [self._rbf(x[i], x[j]) for i in range(len(x))]
            for j in range(len(x))
            ]
        omega = np.linalg.solve(coef_mat, fx)
        self.omega = omega
    
    def _fit_univariate(self, x_new):
        fx_bar = [self._rbf(x_new, self.x[i])*self.omega[i]
                  for i in range(len(self.x))]
        fx_bar = np.sum(fx_bar)
        return fx_bar
    
    def _fit_multivariate(self, x_new):
        fx_bar = [self._fit_univariate(x_new_i) for x_new_i in x_new]
        return fx_bar


# =-=-=-=-=-= classes for income fluctuation problem =-=-=-=-=-=-=-=-=-=-=-=-=-
class IFP: # Income fluctuation problem
    def __init__(self,
                 beta      = 0.9400, # discount factor
                 sig       = 2.0000, # inverse IES 
                 R         = 1.0100, # gross interest rate
                 a_lb      = 0.0000, # borrowing limit
                 rho       = 0.9900, # AR coefficient of income process
                 var_eps   = 0.0426, # variance of income shock
                 n_grids_y = 17, # # of gird points for y
                 n_grids_a = 500, # # of gird points for a 
                 a_range   = (0, 350), # range of a interval
                 Omega     = 3.0000, # half of interval range (for tauchen)
                 ):
        # calculate sigma_y based on rho and var_eps
        sig_eps = np.sqrt(var_eps)
        sig_lny = sig_eps * (1 - rho**2)**(-1/2)
        var_lny = sig_lny**2
        
        # prepare the grid points for a
        if a_range != a_lb:
            a_range = (a_lb, a_range[1])
        
        a_grid = np.linspace(a_range[0], a_range[1], n_grids_a)
        
        # Store the given parameters as instance attributes
        self.beta      = beta
        self.sig       = sig
        self.R         = R
        self.a_lb      = a_lb
        self.rho       = rho
        self.var_eps   = var_eps
        self.sig_eps   = sig_eps
        self.var_lny   = var_lny
        self.sig_lny   = sig_lny
        self.n_grids_y = n_grids_y
        self.n_grids_a = n_grids_a
        self.a_grid    = a_grid
        self.Omega     = Omega
    
    
    def discretize(self, method,
                is_write_out_result = True,
                is_quiet = False): 
        if method in ['tauchen', 'Tauchen', 'T', 't']:
            if not is_quiet:
                print("\n Discretizing the income process by Tauchen method")
            self._tauchen_discretize(is_write_out_result)
        
        elif method in ['rouwenhorst', 'Rouwenhorst', 'R', 'r']:
            if not is_quiet:
                print("\n Discretizing the income process by Rouwenhorst method")
            self._rouwenhorst_discretize(is_write_out_result)
            
        else:
            raise Exception('"method" input much be "Tauchen" or "Rouwenhorst."')
    
    
    def _tauchen_discretize(self, is_write_out_result):
        # nested function to compute i-j element of the transition matrix
        def tauchen_trans_mat_ij(self, i, j, lny_grid, h):
            if j == 0:
                trans_mat_ij = norm.cdf((lny_grid[j] - self.rho*lny_grid[i] + h/2)/self.sig_eps)
            elif j == (self.n_grids_y-1):
                trans_mat_ij = 1 - norm.cdf((lny_grid[j] - self.rho*lny_grid[i] - h/2)/self.sig_eps)
            else:
                trans_mat_ij = ( norm.cdf((lny_grid[j] - self.rho*lny_grid[i] + h/2)/self.sig_eps)
                               - norm.cdf((lny_grid[j] - self.rho*lny_grid[i] - h/2)/self.sig_eps))
            return trans_mat_ij
        
        # Prepare gird points
        lny_N    = self.Omega * self.sig_lny
        lny_grid = np.linspace(-lny_N, lny_N, self.n_grids_y)
        y_grid   = np.exp(lny_grid)
        
        # Calculate the step size
        h = (2 * lny_N)/(self.n_grids_y-1)
        
        # Construct the transition matrix
        trans_mat = [ 
            [tauchen_trans_mat_ij(i, j, lny_grid, h) 
             for j in range(self.n_grids_y)
            ]
            for i in range(self.n_grids_y)
            ]
            
        if is_write_out_result:
            np.savetxt('Tauchen_y_grid.csv', y_grid, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Tauchen_trans_mat.csv', trans_mat, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
        
        self.lny_grid, self.y_grid, self.trans_mat, self.step_size =\
            lny_grid, y_grid, np.array(trans_mat), h
    
    
    def _rouwenhorst_discretize(self, is_write_out_result):
        # Prepare y gird points
        lny_N    = self.sig_lny * np.sqrt(self.n_grids_y - 1)
        lny_grid = np.linspace(-lny_N, lny_N, self.n_grids_y)
        y_grid   = np.exp(lny_grid)
        
        # Calculate the step size
        h = (2 * lny_N)/(self.n_grids_y-1)
        
        # parameter necessary for Rouwenhorst recursion
        pi = 0.5 * (1 + self.rho)
        
        # N = 2
        Pi_N = np.array([[pi, 1 - pi],
                         [1 - pi, pi]])
        
        for n in range(3, self.n_grids_y+1, 1):
            Pi_pre = deepcopy(Pi_N)
            Pi_N1, Pi_N2, Pi_N3, Pi_N4 = \
                np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))
            
            Pi_N1[:n-1, :n-1] = Pi_N2[:n-1, 1:n] = \
                Pi_N3[1:n, 1:n] = Pi_N4[1:n, :n-1] = Pi_pre
            
            Pi_N = (pi * Pi_N1
                    + (1 - pi) * Pi_N2
                    + pi * Pi_N3
                    + (1 - pi) * Pi_N4
            )
            # Divide all but the top and bottom rows by two so that the 
            # elements in each row sum to one (Kopecky & Suen[2010, RED]).
            Pi_N[1:-1, :] *= 0.5
            
        if is_write_out_result:
            np.savetxt('Rouwenhorst_y_grid.csv', y_grid, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Rouwenhorst_trans_mat.csv', Pi_N, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')

        self.lny_grid, self.y_grid, self.trans_mat, self.step_size \
            = lny_grid, y_grid, Pi_N, h
        
    
    def muc(self, y_td, a_td, a_tmrw): # marginal utility of consumption
        # Calculate consumption as the residual in the budget constraint
        c_td = self.R*a_td + y_td - a_tmrw
        # If consumption is negative, assign NaN
        
        if type(c_td) is np.float64:
            if c_td <= 0:
                c_td = np.nan
        else:
            c_td[c_td <= 0] = np.nan

        # Compute the marginal utility of consumption
        muc_td = c_td**(-self.sig) 
        
        return muc_td

    def get_nearest_a_idx(self, continuous_a):
        nearest_idx = get_nearest_idx(continuous_a, self.a_grid)
        return nearest_idx    
    
    def _check_borrowing_constraint(self, y_td_idx, a_td_idx, a_tmrw_idx_mat):
    # Today's state variables (scalers)
        y_td, a_td  = self.y_grid[y_td_idx], self.a_grid[a_td_idx]
        
        # Optimal asset holdings tommorrow (scaler)
        a_tmrw_idx = 0
        a_tmrw = self.a_grid[a_tmrw_idx]

        # Possible asset holdings day after tomorrow (1D array)
        a_dat_idx_vec = a_tmrw_idx_mat[:, a_tmrw_idx]
        a_dat_vec = self.a_grid[a_dat_idx_vec]
        
        # Left-hand side of the Euler equation (scaler)
        LHS = self.muc(y_td = y_td,
                       a_td = a_td,
                       a_tmrw = a_tmrw)
        
        # Right-hand side of the Euler equation  (1D array)
        muc_tmrw_vec = self.muc(y_td = self.y_grid,
                                a_td = a_tmrw,
                                a_tmrw = a_dat_vec) # vector
        RHS = self.beta * self.R * self.trans_mat[y_td_idx, :] @ muc_tmrw_vec # scaler
        
        # The difference between LHS and RHS (EE = Euler equation)
        EE_resid = LHS - RHS
               
        return EE_resid
    
    
    def _optimal_a_tmrw(self, y_td_idx, a_td_idx, a_tmrw_idx_mat):
        # Today's state variables (scalers)
        y_td, a_td = self.y_grid[y_td_idx], self.a_grid[a_td_idx]
        
        # Possible asset holdings tommorrow (horizontal vector)
        a_tmrw_idx_vec = np.arange(self.n_grids_a).reshape(1, -1)
        a_tmrw_vec = (self.a_grid).reshape(1, -1)
        
        # Possible asset holdings day after tomorrow (matrix)
        a_dat_idx_mat = (a_tmrw_idx_mat)[:, a_tmrw_idx_vec.flatten()]
        a_dat_mat = self.a_grid[a_dat_idx_mat]
        
        # Possbile values of left-hand side of the Euler equation
        LHS = self.muc(y_td = y_td,
                       a_td = a_td,
                       a_tmrw = a_tmrw_vec)
        
        # Possbile values of right-hand side of the Euler equation
        muc_tmrw_mat = self.muc(y_td = (self.y_grid).reshape(-1,1),
                                a_td = a_tmrw_vec,
                                a_tmrw = a_dat_mat) 
        trans_prob = self.trans_mat[y_td_idx, :].reshape(1, -1)
        RHS = self.beta * self.R * trans_prob @ muc_tmrw_mat
        
        # The difference between LHS and RHS (EE = Euler equation)
        EE_resid_vec = LHS - RHS

        # find the value for a' most consistent with the Euler equation
        # and return its index
        # Note: Monotonicity of marginal utility is exploited.
        optimal_a_idx = np.nanargmin(abs(EE_resid_vec))        
        return optimal_a_idx
    
    
    def policy_func_iter(self,
                         tol = 1E-5,
                         max_iter = 10000,
                         fname = 'policy_func_result'
                         ):
        a_idx_vec = np.arange(self.n_grids_a).reshape(1, -1) # horizontal direction: asset

        # Initial guess for the policy function (Use the 45 degree line)
        a_hat_idx_mat = np.tile(a_idx_vec, (self.n_grids_y, 1))
        
        # Initialize while loop
        diff, iteration = 100, 0
        
        tic = time.perf_counter()
        while (diff > tol) & (iteration <= max_iter):
            # Prepare a guess for the policy function
            a_hat_idx_guess_mat = deepcopy(a_hat_idx_mat)
                        
            # Check if the borrowing constraint is binding
            for j in range(self.n_grids_a): # loop wrt grid points for a
                for i in reversed(range(self.n_grids_y)): # loop wrt grid points for y
                    # Check if the borrowing constraint is binding
                    EE_resid = self._check_borrowing_constraint(y_td_idx  = i,   
                                                                a_td_idx  = j,
                                                                a_tmrw_idx_mat = a_hat_idx_guess_mat)
                    if EE_resid > 0:
                        a_hat_idx_mat[:i, j] = 0
                        break
                    else:
                        a_hat_idx_mat[i, j] = \
                            self._optimal_a_tmrw(y_td_idx = i,
                                                 a_td_idx = j,
                                                 a_tmrw_idx_mat = a_hat_idx_guess_mat)
            
            diff = np.max(abs(self.a_grid[a_hat_idx_mat] - self.a_grid[a_hat_idx_guess_mat]))
            iteration += 1
            if iteration % 10 == 0:
                print('\n Iteration {:５d}: Diff =  {:.3f}'.format(iteration, diff))
        toc = time.perf_counter()
        
        # Store the result as instance attributes
        self.elasped_time     = toc - tic
        self.policy_idx_func = a_hat_idx_mat
        self.policy_func     = self.a_grid[a_hat_idx_mat]
            
        # Plot the result
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(self.a_grid, self.policy_func[0, :], 
                     '-', lw = 1.5, color = 'red', label='Policy function: y_1')
        plt.plot(self.a_grid, self.policy_func[7, :], 
                     '--', lw = 1.5, color = 'orange', label='Policy function: y_8')
        plt.plot(self.a_grid, self.policy_func[16, :], 
                     '-', lw = 2, color = 'blue', label='Policy function: y_17')
        plt.plot(self.a_grid, self.a_grid, 
                     ls='dotted', lw = 1, color = 'gray', label='45 degree line')
        plt.legend(frameon = False)
        fig1.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0)
    
    
    def policy_func_iter_w_intrpl(self,
                                  tol = 1E-5,
                                  max_iter = 100000,
                                  intrpl_method = 'piecewise',
                                  solving_method = 'bisection',
                                  fname = None
                                  ):
        if intrpl_method == 'piecewise':
            interpolate = piecewise_intrpl
        elif intrpl_method == 'radial_basis':
            interpolate = rbf_intrpl
        else:
            raise Exception('Choose a method from "piecewise" and "radial_basis".')
       
        a_idx_vec = np.arange(self.n_grids_a).reshape(1, -1) # horizontal direction: asset
        
        # Initial guess for the policy function (Use the 45 degree line)
        a_hat_idx_mat   = np.tile(a_idx_vec, (self.n_grids_y, 1))
        
        # Initialize while loop
        diff, iteration = 100, 0
        
        tic = time.perf_counter()
        while (diff > tol) & (iteration <= max_iter):
            # Prepare a guess for the policy function
            a_hat_idx_guess_mat = deepcopy(a_hat_idx_mat)
            a_hat_guess_mat     = self.a_grid[a_hat_idx_guess_mat]
            
            a_hat_intrpl =[
                interpolate(self.a_grid, a_hat_guess_mat[i, :])
                for i in range(self.n_grids_y)
                ]
            
            self.policy_func_tmp = a_hat_guess_mat
            self.a_hat_intrpl = a_hat_intrpl
            
            # Check if the borrowing constraint is binding
            for j in range(self.n_grids_a): # loop wrt grid points for a
                for i in reversed(range(self.n_grids_y)): # loop wrt grid points for y
                    # Check if the borrowing constraint is binding
                    EE_resid = self._check_borrowing_constraint(y_td_idx  = i,   
                                                                a_td_idx  = j,
                                                                a_tmrw_idx_mat = a_hat_idx_guess_mat)
                        
                    if EE_resid > 0:
                        a_hat_idx_mat[:i, j] = 0
                        break
                    else:
                        a_hat_idx_mat[i, j] = \
                            self._optimal_a_tmrw_intrpl(y_td_idx = i,
                                                        a_td_idx = j,
                                                        solver = solving_method)

            diff = np.max(abs(self.a_grid[a_hat_idx_mat] - a_hat_guess_mat))
            iteration += 1
            if iteration % 10 == 0:
                print('\n Iteration {:5d}: Diff =  {:.3f}'.format(iteration, diff))
        toc = time.perf_counter()
        self.elasped_time = toc - tic
        
        self.policy_idx_func = a_hat_idx_mat
        self.policy_func     = self.a_grid[a_hat_idx_mat]
        
        # Plot the result
        if fname is None:
            fname = 'Q1' + intrpl_method +'_' + solving_method + '.png'
        
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(self.a_grid, self.policy_func[0, :], 
                     '-', lw = 1.5, color = 'red', label='Policy function: y_1')
        plt.plot(self.a_grid, self.policy_func[7, :], 
                     '--', lw = 1.5, color = 'orange', label='Policy function: y_8')
        plt.plot(self.a_grid, self.policy_func[16, :], 
                     '-', lw = 2, color = 'blue', label='Policy function: y_17')
        plt.plot(self.a_grid, self.a_grid, 
                     ls='dotted', lw = 1, color = 'gray', label='45 degree line')
        plt.legend(frameon = False)
        fig1.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0)
    
    
    def _optimal_a_tmrw_intrpl(self, 
                               y_td_idx, 
                               a_td_idx, 
                               solver = 'bisection'):
        y_td, a_td  = self.y_grid[y_td_idx], self.a_grid[a_td_idx]
            
        def objective_func(a_tmrw):
            if a_tmrw is np.ndarray:
                a_tmrw = a_tmrw[0,]
                        
            LHS = self.muc(y_td, a_td, a_tmrw)
            
            a_dat_vec = [self.a_hat_intrpl[i](a_tmrw)
                         for i in range(self.n_grids_y)]
            
            a_dat_vec = np.array(a_dat_vec).reshape(17, -1)
            muc_vec = self.muc(
                y_td   = (self.y_grid).reshape(-1,1),
                a_td   = a_tmrw,
                a_tmrw = a_dat_vec
                )
            trans_prob = self.trans_mat[y_td_idx, :].reshape(1, -1)
                            
            RHS = self.beta * self.R * trans_prob @ muc_vec
            
            diff = (LHS - RHS).reshape(1,)
            
            if np.isnan(diff):
                diff = 9999.
            return diff
        
        # Find optimal a' (continuous) 
        calc_EE_resid = objective_func
        
        if solver == 'bisection':
            optimal_a_tmrw, _ = self.bisection_solver(
                func  = calc_EE_resid,
                x_min = self.a_grid[0],
                x_max = self.a_grid[-1]
                )
            
        elif solver == 'scipy_bisection':
            a_min = self.a_grid[0]
            a_max = self.a_grid[-1]
            if calc_EE_resid(a_max) <= 0:
                optimal_a_tmrw = a_max
            else:
                optimal_a_tmrw = bisect(f = calc_EE_resid,
                                        a = a_min, b = a_max, xtol=1E-5,
                                        rtol=1E-5, disp=False)
        
        elif solver == 'Newton':
            # Set the inital value according to the old policy function
            a_conjecture = self.policy_func_tmp[y_td_idx, a_td_idx]
            # run Newton method
            optimal_a_tmrw, _ = self.newton_solver(func  = calc_EE_resid,
                                                   x_0   = a_conjecture)
        elif solver == 'fsolve':
            # Set the inital value according to the old policy function
            a_conjecture = self.policy_func_tmp[y_td_idx, a_td_idx]
            # run solve fuhcion
            optimized_result = fsolve(func = calc_EE_resid,
                                    x0   = a_conjecture)
            optimal_a_tmrw = optimized_result[0]
        else:
            # Set the inital value according to the old policy function
            a_conjecture = self.policy_func_tmp[y_td_idx, a_td_idx]
            # run solve fuhcion
            optimized_result = root(fun    = calc_EE_resid,
                                    x0     = a_conjecture,
                                    method = solver)
            optimal_a_tmrw = optimized_result.x[0]        
        
        # Find optimal a' (discrete) 
        opt_a_idx = self.get_nearest_a_idx(optimal_a_tmrw)
           
        return opt_a_idx


    def bisection_solver(self, 
                         func,  # function to be solved
                         x_min,  # lower end of initial interval 
                         x_max,  # upper end of initial interval
                         max_iter = 10000, # Maximum number of iterations
                         tol = 1E-5      # torelance level in convergence test
                         ):
        x_a, x_b = x_min, x_max # just rename the both ends of interval
        f_a, f_b = func(x_a), func(x_b) # calculate function values
        
        idx = -1
        while np.isnan(f_b):
            idx -= 1
            x_b = self.a_grid[idx]
            f_b = func(x_b)      
        x_max = deepcopy(x_b)
        # check if the initial interval is appropriate for bisection
        if f_a * f_b > 0:
            return x_b, f_b
        
        # initialize while-loop           
        diff, iteration = min(abs(f_a), abs(f_b)), 0
        
        if diff <= tol:
            if abs(f_a) < abs(f_b):
                return x_a, f_a
            else:
                return x_b, f_b
        
        while (diff > tol) and (iteration <= max_iter):
            # updated point
            x_c = np.mean([x_a, x_b])
            f_c = func(x_c)
                        
            # update the end point which has the same sign with x_c
            if f_a * f_c > 0:
                x_a, f_a = x_c, f_c
            else:
                x_b, f_b = x_c, f_c
            
            diff = min([abs(f_c), abs(x_a - x_b)]) 
            iteration += 1
        
        if diff > tol:
            raise Exception("Bisection failed to find the solution within MaxIter.")
        
        return x_c, f_c
    
    
    
    def newton_solver(self,
                      func, # function to be solved
                      x_0, # initial value
                      max_iter = 10000,  # step size when numerically differencing
                      tol      = 1E-5, # Maximum number of iterations
                      delta_x  = 0.01  # torelance level in convergence test
                      ):
        
        # set the initial value as the tentative champion
        x_post, f_post = x_0, func(x_0)
        
        # initialize while-loop
        diff, iteration = abs(f_post), 0
        hist = []
        
        while (diff > tol) and (iteration <= max_iter):
            x_pre, f_pre = deepcopy(x_post), deepcopy(f_post)
            
            # calculate numerical derivative at x = x_pre
            f_dx = func(x_pre+delta_x)
            df = (f_dx - f_pre)/delta_x
            
            # update x
            x_post = x_pre - f_pre/df
            f_post = func(x_post)
            if x_post > self.a_grid[-1]:
                x_post = self.a_grid[-1]
                f_post = func(x_post)
                diff = 0
                break
            
            elif x_post < self.a_grid[0]:
                x_post = self.a_grid[0]
                f_post = func(x_post)
                diff = 0
                break
            
            diff = abs(f_post)
            iteration += 1
            hist.append([x_post, diff])
            
        if diff > tol:
            raise Exception("Bisection failed to find the solution within MaxIter.")
        
        return x_post, f_post
    
    
    def endogenous_grid_method(self,
                                tol = 1E-5,
                                max_iter = 1,
                                intrpl_method = 'piecewise',
                                fname = 'endogenous_grid.png'
                                ):

        def backward_induction_for_a(y_td_idx, a_tmrw_idx, optimal_a_mapping):
            y_td, a_tmrw = self.y_grid[y_td_idx], self.a_grid[a_tmrw_idx]
            
            # Possible asset holdings day after tomorrow (matrix)
            a_dat_idx_vec = (optimal_a_mapping)[:, a_tmrw_idx]
            a_dat_vec = self.a_grid[a_dat_idx_vec].reshape(-1, 1)

            # Possible marginal utility tomorrow
            muc_tmrw_vec = self.muc(y_td = (self.y_grid).reshape(-1,1),
                                    a_td = a_tmrw,
                                    a_tmrw = a_dat_vec) 
            

            # transition probabilities conditional on realized y today
            trans_prob = self.trans_mat[y_td_idx, :]
            
            # Expected (and subjectively discounted) marginal utility tomrrow 
            RHS = np.asscalar(self.beta * self.R * trans_prob @ muc_tmrw_vec)
  
            # today's consumnption consistent with the above RHS
            c_td = RHS**(-1/self.sig) 
        
            # calculate today's asset holding as the residual of budget constraint
            a_td = (c_td + a_tmrw - y_td)/self.R

            return a_td
        
        if intrpl_method == 'piecewise':
            interpolate = piecewise_intrpl
        elif intrpl_method == 'radial_basis':
            interpolate = rbf_intrpl
        else:
            raise Exception('Choose a method from "piecewise" and "radial_basis".')
        
        # Initial guess for the policy function (Use the 45 degree line)
        a_idx_vec = np.arange(self.n_grids_a).reshape(1, -1) # horizontal direction: asset
        a_hat_idx_mat = np.tile(a_idx_vec, (self.n_grids_y, 1))
        
        
        # Initialize while loop
        diff, iteration = 100, 0
        
        tic = time.perf_counter()
        
        while (diff > tol) & (iteration < max_iter):
            # Prepare a guess for the policy function
            a_hat_idx_guess_mat = deepcopy(a_hat_idx_mat)

            a_td_mat = np.array([
                [backward_induction_for_a(i, j, a_hat_idx_guess_mat) for j in range(self.n_grids_a)]
                for i in range(self.n_grids_y)
                ])

            for i in range(self.n_grids_y): # loop wrt grid points for y
                # Constructing the bijection from a to a'
                mapping_i = interpolate(x = a_td_mat[i, :],
                                        fx = self.a_grid)
                
                optimal_a_tmrw = mapping_i(self.a_grid)
                
                optimal_a_tmrw_idx = np.array([
                    self.get_nearest_a_idx(a_tmrw_i)
                    for a_tmrw_i in optimal_a_tmrw
                    ])

                a_hat_idx_mat[i, :] = optimal_a_tmrw_idx
            
            diff = np.max(abs(self.a_grid[a_hat_idx_mat] - self.a_grid[a_hat_idx_guess_mat]))
            iteration += 1
            if iteration % 10 == 0:
                print('\n Iteration {:５d}: Diff =  {:.3f}'.format(iteration, diff))
        toc = time.perf_counter()
        
        # Store the result as instance attributes
        self.elasped_time = toc - tic
        self.policy_idx_func = a_hat_idx_mat
        self.policy_func     = self.a_grid[a_hat_idx_mat]

        # Plot the result
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.a_grid, self.policy_func[0, :], 
                     '-', lw = 1.5, color = 'red', label='Policy function: y_1')
        plt.plot(self.a_grid, self.policy_func[7, :], 
                     '--', lw = 1.5, color = 'orange', label='Policy function: y_8')
        plt.plot(self.a_grid, self.policy_func[16, :], 
                     '-', lw = 2, color = 'blue', label='Policy function: y_17')
        plt.plot(self.a_grid, self.a_grid, 
                     ls='dotted', lw = 1, color = 'gray', label='45 degree line')
        plt.legend(frameon = False)
        fig.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0)
    
        
class IFP_w_taxation(IFP):
    def __init__(self,
                 beta      = 0.9400, # discount factor
                 sig       = 2.0000, # inverse IES 
                 R         = 1.0100, # gross interest rate
                 a_lb      = 0.0000, # borrowing limit
                 rho       = 0.9900, # AR coefficient of income process
                 var_eps   = 0.0426, # variance of income shock
                 n_grids_y = 17, # # of gird points for y
                 n_grids_a = 500, # # of gird points for a 
                 a_range   = (0, 350), # range of a interval 
                 Omega     = 3.0000, # half of interval range (for tauchen))
                 b0        = 0.258, # parameter in income tax function
                 b1        = 0.768, # parameter in income tax function
                 b2        = 0.491, # parameter in income tax function
                 b3        = 0.144  # parameter in income tax function
                 ):
        
            super().__init__(beta      = beta,
                             sig       = sig,
                             R         = R,
                             a_lb      = a_lb,
                             rho       = rho,
                             var_eps   = var_eps,
                             n_grids_y = n_grids_y,
                             n_grids_a = n_grids_a,
                             a_range   = a_range,
                             Omega     = Omega
                             )
            self.b0, self.b1, self.b2, self.b3 = b0, b1, b2, b3
            
    def tax(self, income):
        # Calculate income tax
        tau = (self.b0 * (income - (income**(-self.b1) + self.b2)**(-1/self.b1))
              + self.b3 * income)
        return tau
    
    def plot_income_tax(self):
        tau = self.tax(self.y_grid)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('income')
        ax.set_ylabel('income tax')
        ax.plot(self.y_grid, tau, c='red')
        fig.savefig('Q2a.png', dpi=150, bbox_inches='tight', pad_inches=0)
        
        
    def muc(self, y_td, a_td, a_tmrw): # marginal utility of consumption
        # Calculate consumption as the residual in the budget constraint
        c_td = self.R*a_td + y_td - self.tax(y_td) - a_tmrw
        # If consumption is negative, assign NaN        
        if type(c_td) is np.float64:
            if c_td <= 0:
                c_td = np.nan
        else:
            c_td[c_td <= 0] = np.nan

        # Compute the marginal utility of consumption
        muc_td = c_td**(-self.sig) 
        
        return muc_td
    
    
    def endogenous_grid_method(self,
                                tol = 1E-5,
                                max_iter = 1,
                                intrpl_method = 'piecewise'
                                ):

        def backward_induction_for_a(y_td_idx, a_tmrw_idx, optimal_a_mapping):
            y_td, a_tmrw = self.y_grid[y_td_idx], self.a_grid[a_tmrw_idx]
            
            # Possible asset holdings day after tomorrow (matrix)
            a_dat_idx_vec = (optimal_a_mapping)[:, a_tmrw_idx]
            a_dat_vec = self.a_grid[a_dat_idx_vec].reshape(-1, 1)

            # Possible marginal utility tomorrow
            muc_tmrw_vec = self.muc(y_td = (self.y_grid).reshape(-1,1),
                                    a_td = a_tmrw,
                                    a_tmrw = a_dat_vec) 

            # transition probabilities conditional on realized y today
            trans_prob = self.trans_mat[y_td_idx, :]
            
            # Expected (and subjectively discounted) marginal utility tomrrow 
            RHS = np.asscalar(self.beta * self.R * trans_prob @ muc_tmrw_vec)
  
            # today's consumnption consistent with the above RHS
            c_td = RHS**(-1/self.sig) 
        
            # calculate today's asset holding as the residual of budget constraint
            a_td = (c_td + a_tmrw + self.tax(y_td) - y_td)/self.R

            return a_td
        
        if intrpl_method == 'piecewise':
            interpolate = piecewise_intrpl
        elif intrpl_method == 'radial_basis':
            interpolate = rbf_intrpl
        else:
            raise Exception('Choose a method from "piecewise" and "radial_basis".')
        
        # Initial guess for the policy function (Use the 45 degree line)
        a_idx_vec = np.arange(self.n_grids_a).reshape(1, -1) # horizontal direction: asset
        a_hat_idx_mat = np.tile(a_idx_vec, (self.n_grids_y, 1))
        
        
        # Initialize while loop
        diff, iteration = 100, 0
        
        tic = time.perf_counter()
        
        while (diff > tol) & (iteration < max_iter):
            # Prepare a guess for the policy function
            a_hat_idx_guess_mat = deepcopy(a_hat_idx_mat)

            a_td_mat = np.array([
                [backward_induction_for_a(i, j, a_hat_idx_guess_mat) for j in range(self.n_grids_a)]
                for i in range(self.n_grids_y)
                ])

            for i in range(self.n_grids_y): # loop wrt grid points for y
                # Constructing the bijection from a to a'
                mapping_i = interpolate(x = a_td_mat[i, :],
                                        fx = self.a_grid)
                
                optimal_a_tmrw = mapping_i(self.a_grid)
                
                optimal_a_tmrw_idx = np.array([
                    self.get_nearest_a_idx(a_tmrw_i)
                    for a_tmrw_i in optimal_a_tmrw
                    ])

                a_hat_idx_mat[i, :] = optimal_a_tmrw_idx
            
            diff = np.max(abs(self.a_grid[a_hat_idx_mat] - self.a_grid[a_hat_idx_guess_mat]))
            iteration += 1
            if iteration % 10 == 0:
                print('\n Iteration {:５d}: Diff =  {:.3f}'.format(iteration, diff))
        toc = time.perf_counter()
        
        # Store the result as instance attributes
        self.elasped_time = toc - tic
        self.policy_idx_func = a_hat_idx_mat
        self.policy_func     = self.a_grid[a_hat_idx_mat]