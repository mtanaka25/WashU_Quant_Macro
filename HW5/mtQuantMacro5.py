#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW5.py

is the python class for the assignment #5 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Oct 5, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from copy import deepcopy
from scipy.stats import norm
import time
from datetime import timedelta

# =-=-=-=-=-=-=-=-= classes for interpolation =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class PiecewiseIntrpl: # piecewised linear interpolation on a 1D grid
    def __init__(self, x, fx):
        if x is list:
            x, fx = np.array(x), np.array(fx)
        if x.shape[0] < x.shape[1]:
            x, fx = x.T, fx.T
        
        self.x  = x
        self.fx = fx
    
    def __call__(self, x_new):
        if np.isscalar(x_new):
            fx_bar = self._fit_scalar(x_new)
        else:
            if x_new is list:
                x_new = np.array(x_new)
            
            is_transposed = False
            if x_new.shape[0] < x_new.shape[1]:
                x_new = x_new.T
                is_transposed = True
                
            fx_bar = self._fit_array(x_new)
            
            if is_transposed:
                fx_bar = fx_bar.T
        return fx_bar
    
    def _fit_scalar(self, x_new):
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
        
    def _fit_array(self, x_new):
        fx_bar = [self._fit_scalar(x_new_i) for x_new_i in x_new]
        return fx_bar

class PiecewiseIntrpl_MeshGrid:
    """
    This class implements the piece-wise linear interpolation over
    the mesh grid of 2 dimensions.
    This class can handles functions of R^2 -> R
    
    Args for __init__:
        x1_grid: the grid of the 1st dependend variable [numpy array]
        
        x2_grid: the grid of the 2nd dependend variable [numpy array]
        
        fx: independent variable [numpy array]
                
    Arg for __call__:
        x_new: points on which you need interpolated values [numpy array]
            x_new meeds to have the same number of dependent variables as x has.
            You may also use a list for x_new. In that case, this method automatically
            convert it into a numpy array.
            
    Return from __call__:
        phi: interpolated values [numpy array]
        
    """
    def __init__(self, x1_grid, x2_grid, fx):
        self.x1_grid, self.x2_grid = x1_grid.flatten(), x2_grid.flatten()
        self.fx = fx
    
    def __call__(self, x1, x2):
        if np.isscalar(x1) & np.isscalar(x2):
            fx_bar = self._fit_single_point(x1, x2)
        else:
            fx_bar = self._fit_multiple_points(x1, x2)
        return fx_bar
    
    def _fit_single_point(self, x1_new, x2_new):
        def get_marginal_weight_vector(x, x_hat, j):
            phi = np.zeros((len(x), ))
            if x[-1] <= x_hat:
                phi[-1]   = (x_hat - x[-2])/(x[-1] - x[-2])
                phi[-2] = (x[-1] -  x_hat)/(x[-1] - x[-2])
            elif x_hat <= x[0]:
                phi[1]   = (x_hat - x[0])/(x[1] - x[0])
                phi[0] = (x[1] -  x_hat)/(x[1] - x[0])
            else:
                phi[j]   = (x_hat - x[j-1])/(x[j] - x[j-1])
                phi[j-1] = (x[j] -  x_hat)/(x[j] - x[j-1])
            return phi       
        j1  = sum(self.x1_grid <= x1_new)
        j2  = sum(self.x2_grid <= x2_new)
        
        phi1 = get_marginal_weight_vector(self.x1_grid, x1_new, j1)
        phi2 = get_marginal_weight_vector(self.x2_grid, x2_new, j2)
        
        phi1 = phi1.reshape(-1, 1)
        phi2 = phi2.reshape(1, -1)
        phi  = phi1 @ phi2
        
        fx_bar = np.sum(phi * self.fx)
        
        return fx_bar
    
    def _fit_multiple_points(self, x1, x2):
        if np.isscalar(x1):
            x1 = [x1]
        if np.isscalar(x2):
            x2 = [x2]
             
        fx_bar = np.array([
            [self._fit_single_point(x1[i], x2[j]) for j in range(len(x2))]
            for i in range(len(x1))
            ])
        return fx_bar
    
    def calc_partial_derivative(self, x1, x2, dim=0, dx=1E-5):
        if dim == 0:
            def f_prime(x1_k, x2_l):
                f_plus  = self.__call__(x1_k+dx, x2_l)
                f_minus = self.__call__(x1_k-dx, x2_l)
                f_prm = (f_plus - f_minus)/(2*dx)
                return f_prm
        else:
            def f_prime(x1_k, x2_l):
                f_plus  = self.__call__(x1_k, x2_l+dx)
                f_minus = self.__call__(x1_k, x2_l-dx)
                f_prm = (f_plus - f_minus)/(2*dx)
                return f_prm
        if np.isscalar(x1):
            x1 = [x1]
        if np.isscalar(x2):
            x2 = [x2]
        f_prm_vec = np.array([
                [f_prime(x1[i], x2[j]) for j in range(len(x2))] for i in range(len(x1))
                ])
        return f_prm_vec


class RBFIntrpl: # radial basis function interpolation
    """
    "RBFIntrpl" class implements the radial basis function interpolation.
    This class can handles functions of R -> R
    
    Args for __init__:
        x: dependent variable [array like]
        
        fx: independent variable [array like]
        
        
    Arg for __call__:
        x_new: points on which you need interpolated values [scalar/array like]
            
    Return from __call__:
        fx_bar: interpolated values [scalar/array like]
        
    """
    def __init__(self, x, fx, eps=0.01):
        self.eps = eps
        self.x   = x
        self._solve_for_omega(x, fx)
              
    def __call__(self, x_new):
        if np.isscalar(x_new):
            fx_bar = self._fit_single_point(x_new)
        else:
            fx_bar = self._fit_multiple_points(x_new)
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
    
    def _fit_single_point(self, x_new):
        fx_bar = [self._rbf(x_new, self.x[i])*self.omega[i]
                  for i in range(len(self.x))]
        fx_bar = np.sum(fx_bar)
        return fx_bar
    
    def _fit_multiple_points(self, x_new):
        fx_bar = [self._fit_single_point(x_new_i) for x_new_i in x_new]
        return fx_bar

class RBFIntrpl_MeshGrid: # radial basis function interpolation
    """
    "RBFIntrpl" class implements the radial basis function interpolation
    over the mesh grid of 2 dimensions.
    This class can handles functions of R ^2-> R
    
    Args for __init__:
        x1_grid: the grid of the 1st dependend variable [numpy array]
        
        x1_grid: the grid of the 2nd dependend variable [numpy array]
        
        fx: independent variable [numpy array]
        
        eps: decay parameter in the radial basis function [scalar]
        
        
    Arg for __call__:
        x_new: points on which you need interpolated values [scalar/array like]
            
    Return from __call__:
        fx_bar: interpolated values [scalar/array like]
        
    """
    def __init__(self, x1_grid ,x2_grid, fx, eps=1):
        x1_grid = x1_grid.flatten()
        x2_grid = x2_grid.flatten()
        fx_flatten = sum((fx.T).tolist(), [])
        self.eps = eps
        self.x1_grid, self.x2_grid = x1_grid, x2_grid
        self._solve_for_omega(x1_grid, x2_grid, fx_flatten)      

              
    def __call__(self, x1, x2):
        if np.isscalar(x1) & np.isscalar(x2):
            fx_bar = self._fit_single_point(np.array([x1, x2]))
        else:
            fx_bar = self._fit_multiple_points(x1, x2)
        return fx_bar
    
    def _rbf(self, x, x0): # radial basis function
        distance = abs(x - x0)
        phi = np.exp(- self.eps * np.sum(distance**2))
        return phi
        
    def _solve_for_omega(self, x1, x2, fx):
        coef_mat = np.array([
            [self._rbf(np.array([x1[k], x2[l]]), np.array([x1[i], x2[j]])) for j in range(len(x2)) for i in range(len(x1))]
             for l in range(len(x2)) for k in range(len(x1))
            ])
        omega = np.linalg.solve(coef_mat, fx)
        coef_inv = np.linalg.inv(coef_mat)
        omega2 = coef_inv @ np.array(fx).reshape(-1, 1)
        self.coef_mat = coef_mat
        self.omega = omega
        self.omega2 = omega2
    
    def _fit_single_point(self, x_new):
        x_new = np.array(x_new)
        phi_vec = np.array([
                   self._rbf(x_new, np.array([self.x1_grid[i], self.x2_grid[j]]))
                   for j in range(len(self.x2_grid)) for i in range(len(self.x1_grid)) 
                   ])        
        fx_bar = np.sum(phi_vec * self.omega)
        return fx_bar
    
    def _fit_multiple_points(self, x1, x2):
        if np.isscalar(x1):
            x1 = [x1]
        if np.isscalar(x2):
            x2 = [x2]
        fx_bar = np.array([
            [self._fit_single_point([x1[i], x2[j]]) for j in range(len(x2))]
            for i in range(len(x1))
            ])
        return fx_bar
    
    def calc_partial_derivative(self, 
                                x1, # point(s) on which the partial derivative is calculated (dim 0)
                                x2, # point(s) on which the partial derivative is calculated (dim 1)
                                dim = 0 # For which variable, is partical derivative calculated?
                                ):
        if dim == 0:
            def f_prime(x1_k, x2_l):
                x_new = np.array([x1_k, x2_l])
                phi_prm_vec = np.array([
                           -2 * self.eps * (x1_k - self.x1_grid[i])*self._rbf(x_new, np.array([self.x1_grid[i], self.x2_grid[j]]))
                           for j in range(len(self.x2_grid)) for i in range(len(self.x1_grid)) 
                           ])
                f_prm = np.sum(phi_prm_vec * self.omega)
                return f_prm
        else:
            def f_prime(x1_k, x2_l):
                x_new = np.array([x1_k, x2_l])
                phi_prm_vec = np.array([
                           -2 * self.eps * (x2_l - self.x2_grid[j])*self._rbf(x_new, np.array([self.x1_grid[i], self.x2_grid[j]]))
                           for j in range(len(self.x2_grid)) for i in range(len(self.x1_grid)) 
                           ])
                f_prm = np.sum(phi_prm_vec * self.omega)
                return f_prm
        if np.isscalar(x1):
            x1 = [x1]
        if np.isscalar(x2):
            x2 = [x2]
        f_prm_vec = np.array([
                [f_prime(x1[i], x2[j]) for j in range(len(x2))] for i in range(len(x1))
                ])
        return f_prm_vec
    
# =-=-=-=-=-=-= classe for discretized AR(1) process =-=-=-=-=-=-=-=-=-=-=-=-=
class AR1_process:
    def __init__(self,
                rho = 0.9000, # AR coefficient
                sig = 0.0080, # size of shock
                varname = 'x'  # variable name
                ):
        self.rho = rho
        self.sig = sig
        self.varname = varname
    
    def discretize(self, 
                   method,
                   N = 100, # number of grid points
                   Omega = 3, # scale parameter for Tauchen's grid range
                   is_write_out_result = True,
                   is_quiet = False): 
        if method in ['tauchen', 'Tauchen', 'T', 't']:
            if not is_quiet:
                print("Discretizing the AR(1) process by Tauchen method...\n")
            self._tauchen_discretize(N, Omega, is_write_out_result)
        elif method in ['rouwenhorst', 'Rouwenhorst', 'R', 'r']:
            if not is_quiet:
                print("Discretizing the income process by Rouwenhorst method...\n")
            self._rouwenhorst_discretize(N, is_write_out_result)
        else:
            raise Exception('"method" must be "Tauchen" or "Rouwenhorst."')
    
    def _tauchen_discretize(self,N, Omega, is_write_out_result):
        # nested function to compute i-j element of the transition matrix
        def tauchen_trans_mat_ij(i, j, x_grid, h):
            if j == 0:
                trans_mat_ij = norm.cdf((x_grid[j] - self.rho*x_grid[i] + h/2)/self.sig)
            elif j == (N-1):
                trans_mat_ij = 1 - norm.cdf((x_grid[j] - self.rho*x_grid[i] - h/2)/self.sig)
            else:
                trans_mat_ij = ( norm.cdf((x_grid[j] - self.rho*x_grid[i] + h/2)/self.sig)
                               - norm.cdf((x_grid[j] - self.rho*x_grid[i] - h/2)/self.sig))
            return trans_mat_ij
        
        # Prepare gird points
        sig_x  = self.sig * (1 - self.rho**2)**(-1/2)
        x_max  = Omega * sig_x
        x_grid = np.linspace(-x_max, x_max, N)
        
        # Calculate the step size
        h = (2 * x_max)/(N - 1)
        
        # Construct the transition matrix
        trans_mat = [ 
            [tauchen_trans_mat_ij(i, j, x_grid, h) for j in range(N)]
            for i in range(N)
            ]
            
        if is_write_out_result:
            np.savetxt('Tauchen_{0}_grid.csv'.format(self.varname), x_grid, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Tauchen_trans_mat.csv', trans_mat, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
        
        # Store the result as the instance's attributes
        self.__dict__['{0}_grid'.format(self.varname)] = x_grid
        self.trans_mat, self.step_size = np.array(trans_mat), h
    
    def _rouwenhorst_discretize(self, N, is_write_out_result):
        # Prepare gird points
        sig_x  = self.sig * (1 - self.rho**2)**(-1/2)
        x_max  = sig_x * np.sqrt(N - 1)
        x_grid = np.linspace(-x_max, x_max, N)
        
        # Calculate the step size
        h = (2 * x_max)/(N-1)
        
        # parameter necessary for Rouwenhorst recursion
        pi = 0.5 * (1 + self.rho)
        
        # N = 2
        Pi_N = np.array([[pi, 1 - pi],
                         [1 - pi, pi]])
        
        for n in range(3, N+1, 1):
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
            np.savetxt('Rouwenhorst_{0}_grid.csv'.format(self.varname), x_grid, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Rouwenhorst_trans_mat.csv', Pi_N, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            
        # Store the result as the instance's attributes
        self.__dict__['{0}_gird'.format(self.varname)] = x_grid
        self.trans_mat, self.step_size = Pi_N, h

# =-=-=-=-=-=-=-=-=-=-=-=-=- helpful functions =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def get_nearest_idx(x, array):
    nearest_idx = np.abs(array - x).argmin()
    return nearest_idx

def get_nearest_idx_vec(x_vec, array):
    if x_vec == np.ndarray:
        x_vec = x_vec.flatten()
    nearest_index_vec = [get_nearest_idx(x_vec_i, array) for x_vec_i in x_vec]
    nearest_index_vec = np.array(nearest_index_vec)
    return nearest_index_vec


# =-=-=-=-=-= classes for the simplified Arellano 2008 model=-=-=-=-=-=-=-=-=
class SimplifiedArellano2008:
    def __init__(self,
                 beta      = 0.8800, # discount factor
                 gamma     = 1.5000, # inverse IES 
                 theta     = 0.5000, # inverse Frisch elasticity
                 r         = 0.0300, # world net interest rate
                 phi       = 0.9300, # productivity cap in defaulting state
                 rho       = 0.9000, # AR coefficient of income process
                 sig_eps   = 0.0080, # variance of income shock
                 N_A       = 31, # # of gird points for A
                 N_b       = 51, # # of gird points for b 
                 b_range   = (-0.41, 0), # range of b interval
                 kappa     = 0.03, 
                 Omega     = 3.0000, # half of interval range (for tauchen)
                 ):
        
        # Create an instance for the productivity process
        lnA_AR1 = AR1_process(rho = rho,
                              sig = sig_eps,
                              varname = 'lnA')
        
        b_grid = np.linspace(b_range[0], b_range[1], N_b)
        
        # Store the given parameters as instance attributes
        self.beta      = beta
        self.gamma     = gamma
        self.theta     = theta
        self.r         = r
        self.phi       = phi
        self.N_A       = N_A
        self.N_b       = N_b
        self.b_grid    = b_grid
        self.kappa     = kappa    
        self.Omega     = Omega
        self.lnA_AR1   = lnA_AR1
    
    
    def discretize_lnA_process(self, 
                               method,
                               N = None,
                               is_write_out_result = True,
                               is_quiet = False): 
        if type(N) == type(None):
            N = self.N_A
        
        self.lnA_AR1.discretize(method = method,
                                N = N,
                                Omega = self.Omega,
                                is_write_out_result = is_write_out_result,
                                is_quiet = is_quiet)
        
        A_grid = np.exp(self.lnA_AR1.lnA_grid)
        self.trans_mat = deepcopy(self.lnA_AR1.trans_mat)
        self.A_grid = A_grid
    
    
    def VD(self, A_idx, V_tmrw, is_interpolated=False): # value of default today
        # today's productivity level
        A = self.A_grid[A_idx]
        # The instantaneous payoff
        inst_payoff = self.utility(A = A, 
                                   b = 0, 
                                   b_prime = 0, 
                                   q = 0,
                                   isdefault = True)
        # Pick up transition matrix
        if is_interpolated:
            Pi = self.trans_mat_finer
        else:
            Pi = self.trans_mat
        # expected value
        E_value = Pi[A_idx, :] @ V_tmrw[:, -1]
        # The value of default today
        V_D = inst_payoff + self.beta * E_value
        return V_D
    
    
    def VG(self, A_idx, b_idx, q, V_tmrw, is_interpolated=False): # value of repaying today
        # today's productivity level and borrowing
        A = self.A_grid[A_idx]
        b = self.b_grid[b_idx]
        # Possible borrowing
        if is_interpolated:
            b_prime_vec = self.b_finer_grid.reshape(1, -1)
        else:
            b_prime_vec = self.b_grid.reshape(1, -1)
        # instantaneous payoff
        inst_payoff_vec = self.utility(A = A, 
                                       b = b, 
                                       b_prime = b_prime_vec, 
                                       q = q[A_idx, :])
        # Pick up transition matrix
        if is_interpolated:
            Pi = self.trans_mat_finer
        else:
            Pi = self.trans_mat
        # expected value
        E_value_vec = Pi[A_idx, :] @ V_tmrw
        # possible VGs
        VG_vec = inst_payoff_vec + self.beta * E_value_vec            
        if all(np.isnan(VG_vec.flatten())):
            V_G, opt_b = -1000, np.nan
        else:
            # take the maximum
            V_G, argmax_idx = np.nanmax(VG_vec), np.nanargmax(VG_vec)
            opt_b = b_prime_vec[0, argmax_idx]
        return V_G, opt_b
    
    def utility(self, A, b, b_prime, q, isdefault=False):
        #productivity
        if isdefault:
            tfp = min(A, self.phi)
        else:
            tfp = A
        # optimal labor input
        l  = tfp**(1/self.theta)
        # consumption
        c = tfp * l + b - q * b_prime
        # the inside of () in the CES utility
        inside_u = c - l**(1+self.theta)/(1+self.theta)
        # If the inside of () is not greater than zero, assign a penalty value
        if np.isscalar(inside_u):
            if inside_u <= 0:
                inside_u = np.nan
        else:
            inside_u[inside_u <= 0] = np.nan
        # calculate utility
        u = inside_u**(1 - self.gamma) / (1 - self.gamma)
        return u
    
    
    def muc(self, A, b, b_prime, q, isdefault=False):
        #productivity
        if isdefault:
            tfp = min(A, self.phi)
        else:
            tfp = A
        # optimal labor input
        l  = tfp**(1/self.theta)
        # consumption
        c = tfp * l + b - q * b_prime
        # the inside of () in the CES utility
        inside_u = c - l**(1+self.theta)/(1+self.theta)
        # If the inside of () is not greater than zero, assign a penalty value
        if np.isscalar(inside_u):
            if inside_u <= 0:
                inside_u = np.nan
        else:
            inside_u[inside_u <= 0] = np.nan
        # calculate utility
        u = inside_u**(- self.gamma)
        return u
    
    
    def E_V(self, V_G, V_D, is_wo_shock=False):
        if is_wo_shock:
            E_V = np.max([V_G, V_D], axis = 0)
        else:
            E_V = V_D + self.kappa * np.log(
                np.exp((V_G - V_D)/self.kappa) + 1
                )
        return E_V 
    
    
    def D_star(self, V_G, V_D, is_wo_shock): # default rate
        if is_wo_shock:
            D_star = np.argmax([V_G, V_D], axis = 0)
        else:
            expVG_VD = np.exp((V_G - V_D)/self.kappa)
            D_star = 1 / (1 + expVG_VD)
        return D_star
    
    
    def q(self, A_idx, b_idx, D_star_prime):
        D_star_prime_vec = D_star_prime[:, b_idx]
        q = self.trans_mat[A_idx, :] @ (1 - D_star_prime_vec) / (1 + self.r)
        return q
    
    
    def value_func_iter(self,
                        V_init   = None,
                        q_init   = None,
                        max_iter = 1000,
                        tol      = 1E-5,
                        is_wo_shock = False
                        ):
        # if the inital values for V and q are not given, use zero matricies
        if type(V_init) == type(None):
            V_init = np.zeros((self.N_A, self.N_b))
        if type(q_init) == type(None):
            q_init = np.zeros((self.N_A, self.N_b))
        
        # initialize the whlile loop
        diff = 999.
        iteration = 0
        V_t   = deepcopy(V_init)
        q_tm1 = deepcopy(q_init)
        
        tic = time.process_time()
        while (iteration < max_iter) & (diff > tol):
            # Taking V_{t+1} (V_tp1) and q_t, solve for the period-t values and q_{t-1}
            # Load the values in the previous loop
            V_tp1 = deepcopy(V_t)
            q_t   = deepcopy(q_tm1)
            
            # Value of default in period t
            VD_t = [self.VD(a_idx, V_tp1) for a_idx in range(self.N_A)]
            VD_t = np.tile(np.array(VD_t).reshape(-1, 1), (1, self.N_b))
            
            # Value of repaying in period t and optimal b_{t+1}
            VG_and_b_prime = [
                [self.VG(a_idx, b_idx, q_t, V_tp1) for b_idx in range(self.N_b)]
                for a_idx in range(self.N_A)
                ]
            VG_and_b_prime = np.array(VG_and_b_prime)
            VG_t = VG_and_b_prime[:, :, 0]
            b_tp1 = VG_and_b_prime[:, :, 1]
                        
            # Maximized value in period t
            V_t = self.E_V(V_G = VG_t, V_D = VD_t, is_wo_shock=is_wo_shock)
            
            # Default rate
            D_t = self.D_star(V_G = VG_t, V_D = VD_t, is_wo_shock=is_wo_shock)
            
            # Bond price at t-1
            q_tm1 = [
                [self.q(a_idx, b_idx, D_t) for b_idx in range(self.N_b)]
                for a_idx in range(self.N_A)
                ]
            q_tm1 = np.array(q_tm1)
            
            diff = np.max([np.nanmax(abs(V_tp1 - V_t)), np.nanmax(abs(q_t - q_tm1))])
            iteration += 1
        toc = time.process_time()
        elapsed_time = toc - tic
        
        return V_t, b_tp1, D_t, q_tm1, VD_t, VG_t, elapsed_time, iteration
    
    
    def value_func_iter_w_intrpl(self,
                                V_init    = None,
                                q_init    = None,
                                max_iter  = 1000,
                                tol       = 1E-5,
                                N_A_finer = 200,
                                N_b_finer = 1000,
                                is_wo_shock = False
                                ):
        if type(V_init) == type(None):
            V_init = np.zeros((self.N_A, self.N_b))
        if type(q_init) == type(None):
            q_init = np.zeros((self.N_A, self.N_b))
        
        # back up the old transition matrix and grid points
        A_grid_original, trans_mat_original = deepcopy(self.A_grid), deepcopy(self.trans_mat)
        
        # Rerun Tauchen's method to obtain the finer transition matrix
        # -- Be careful that self.A_gird and self.trans_mat would be overwritten 
        self.discretize_lnA_process(method = 'Tauchen',
                                    N = N_A_finer,
                                    is_write_out_result = False,
                                    is_quiet = True)
        
        # exchange the variable names
        self.A_grid, self.A_finer_grid = A_grid_original, self.A_grid
        
        # Prepare the finer grid for b
        self.b_finer_grid = np.linspace(self.b_grid[0], self.b_grid[-1], N_b_finer)
        
        # Reduce the size of the transition matrix to N_A * N_A_finer
        A_grid_correspondence = get_nearest_idx_vec(self.A_grid, self.A_finer_grid)
        self.trans_mat = self.trans_mat[A_grid_correspondence, :]
        
        # exchange the variable names
        self.trans_mat, self.trans_mat_finer = trans_mat_original, self.trans_mat
        # initialize the whlile loop
        diff = 999.
        iteration = 0
        V_t   = deepcopy(V_init)
        q_tm1 = deepcopy(q_init)
        
        tic = time.process_time()
        while (iteration < max_iter) & (diff > tol):
            # Load and interpolate the values in the previous loop
            V_tp1 = deepcopy(V_t)
            q_t   = deepcopy(q_tm1)
            V_intrpl = PiecewiseIntrpl_MeshGrid(self.A_grid, self.b_grid, V_tp1)
            q_intrpl = PiecewiseIntrpl_MeshGrid(self.A_grid, self.b_grid, q_t)
            V_tp1_finer = V_intrpl(self.A_finer_grid, self.b_finer_grid)
            q_t_finer = q_intrpl(self.A_grid, self.b_finer_grid)
            
            # Value of default in period t
            VD_t = [self.VD(a_idx, V_tp1_finer, is_interpolated=True) for a_idx in range(self.N_A)]
            VD_t = np.tile(np.array(VD_t).reshape(-1, 1), (1, self.N_b))
            
            # Value of repaying in period t and optimal b_{t+1}
            VG_and_b_prime = [
                [self.VG(a_idx, b_idx, q_t_finer, V_tp1_finer, is_interpolated=True) for b_idx in range(self.N_b)]
                for a_idx in range(self.N_A)
                ]
            VG_and_b_prime = np.array(VG_and_b_prime)
            VG_t = VG_and_b_prime[:, :, 0]
            b_tp1 = VG_and_b_prime[:, :, 1]
            
            # Maximized value in period t
            V_t = self.E_V(V_G = VG_t, V_D = VD_t, is_wo_shock = is_wo_shock)
            
            # Default rate
            D_t = self.D_star(V_G = VG_t, V_D = VD_t, is_wo_shock = is_wo_shock)
            
            # Bond price at t-1
            q_tm1 = [
                [self.q(a_idx, b_idx, D_t) for b_idx in range(self.N_b)]
                for a_idx in range(self.N_A)
                ]
            q_tm1 = np.array(q_tm1)
            
            diff = np.max([np.max(abs(V_tp1 - V_t)), np.max(abs(q_t - q_tm1))])
            iteration += 1
        toc = time.process_time()
        elapsed_time = toc - tic
        
        return V_t, b_tp1, D_t, q_tm1, VD_t, VG_t, elapsed_time, iteration
    
    
    def Euler_eq_iter_w_intrpl(self,
                               V_init    = None,
                               q_init    = None,
                               max_iter  = 1000,
                               tol       = 1E-5,
                               N_b_finer = 1000,
                               is_wo_shock = False
                               ):
        if type(V_init) == type(None):
            V_init = np.zeros((self.N_A, self.N_b))
        if type(q_init) == type(None):
            q_init = np.zeros((self.N_A, self.N_b))
        
        # Prepare the finer grid for b
        self.b_finer_grid = np.linspace(self.b_grid[0], self.b_finer_grid[-1], N_b_finer)
        db = self.b_finer_grid[1] - self.b_finer_grid[0]
        # initialize the whlile loop
        diff = 999.
        iteration = 0
        V_t   = deepcopy(V_init)
        q_tm1 = deepcopy(q_init)
        
        tic = time.process_time()
        while (iteration < max_iter) & (diff > tol):
            # Load and interpolate the values in the previous loop
            V_tp1 = deepcopy(V_t)
            q_t   = deepcopy(q_tm1)
            V_intrpl = PiecewiseIntrpl_MeshGrid(self.A_grid, self.b_grid, V_tp1)
            q_intrpl = PiecewiseIntrpl_MeshGrid(self.A_grid, self.b_grid, q_t)
            V_tp1_finer = V_intrpl(self.A_grid, self.b_finer_grid)
            q_t_finer   = q_intrpl(self.A_grid, self.b_finer_grid)
                        
            # Calculate the partial derivatives wrt b
            dVdb = np.array(
                V_intrpl.calc_partial_derivative(self.A_grid, self.b_finer_grid, dim=1, dx=db)
                )
            dqdb =  np.array(
                q_intrpl.calc_partial_derivative(self.A_grid, self.b_finer_grid, dim=1, dx=db)
                )
            
            # Calculate the optimal b'
            b_tp1_and_idx = [
                [self.argmin_resid_of_Euler_eq(a_idx, b_idx, q_t_finer, dVdb, dqdb, is_interpolated=True)  for b_idx in range(self.N_b)]
                for a_idx in range(self.N_A)
                ]
            b_tp1_and_idx = np.array(b_tp1_and_idx)
            b_tp1 = b_tp1_and_idx[:, :, 0]
            b_tp1_idx = b_tp1_and_idx[:, :, 1]
            
            adjusted_V_tp1 = np.array([
                [V_tp1_finer[a_idx, int(b_tp1_idx[a_idx, b_idx])]  for b_idx in range(self.N_b)]
                for a_idx in range(self.N_A)
                ])
            adjusted_q_t = np.array([
                [q_t_finer[a_idx, int(b_tp1_idx[a_idx, b_idx])]  for b_idx in range(self.N_b)]
                for a_idx in range(self.N_A)
                ])
            
            # Value of repaying
            VG_t = self.utility(self.A_grid.reshape(-1, 1),
                                self.b_grid.reshape(1, -1),
                                b_tp1, 
                                adjusted_q_t,
                                isdefault=False)
            VG_t = VG_t + self.beta * self.trans_mat @ adjusted_V_tp1
            VG_t[np.isnan(VG_t)] = -1000
            
            # Value of default in period t
            VD_t = [self.VD(a_idx, V_tp1) for a_idx in range(self.N_A)]
            VD_t = np.tile(np.array(VD_t).reshape(-1, 1), (1, self.N_b))
            
            # Maximized value in period t
            V_t = self.E_V(V_G = VG_t, V_D = VD_t, is_wo_shock = is_wo_shock)
            
            # Default rate
            D_t = self.D_star(V_G = VG_t, V_D = VD_t, is_wo_shock = is_wo_shock)
            
            # Bond price at t-1
            q_tm1 = [
                [self.q(a_idx, b_idx, D_t) for b_idx in range(self.N_b)]
                for a_idx in range(self.N_A)
                ]
            q_tm1 = np.array(q_tm1)
            
            diff = np.max([np.max(abs(V_tp1 - V_t)), np.max(abs(q_t - q_tm1))])
            iteration += 1
        toc = time.process_time()
        elapsed_time = toc - tic
        
        return V_t, b_tp1, D_t, q_tm1, VD_t, VG_t, elapsed_time, iteration
    
    def argmin_resid_of_Euler_eq(self, A_idx, b_idx, q, dVdb, dqdb, is_interpolated=False):
        # today's productivity level and borrowing
        A = self.A_grid[A_idx]
        b = self.b_grid[b_idx]
        # Possible borrowing
        if is_interpolated:
            b_prime_vec = self.b_finer_grid.reshape(1, -1)
        else:
            b_prime_vec = self.b_grid.reshape(1, -1)
        # Calculate in advance the marginal utility of consumption
        muc_vec = self.muc(A, b, b_prime_vec, q[A_idx, :], isdefault=False)
        # The left-hand side of Euler equation
        LHS = muc_vec * (q[A_idx, :] + dqdb[A_idx, :] * b_prime_vec)
        LHS[(q[A_idx, :] < 1E-1).reshape(1, -1)] = np.nan
        # The right-hand side of Euler equation
        RHS = self.beta * self.trans_mat[A_idx, :] @ dVdb
        # difference of the two
        resid = abs(LHS - RHS)
        # argmin of difference
        if all(np.isnan(resid.flatten())):
            opt_b_idx = 0
            opt_b = np.nan
        else:
            opt_b_idx = np.nanargmin(resid)
            opt_b = b_prime_vec[0, opt_b_idx]
        # return optimal b vec
        return opt_b, opt_b_idx
    
    
    def solve_problem_a(self):
        # After the world ends, everything is worthless.
        V_terminal = np.zeros((self.N_A, self.N_b))
        q_T = np.zeros((self.N_A, self.N_b))
        
        V_T, _, _, q_Tm1, _, _, elapsed_time, _ = self.value_func_iter(V_init = V_terminal,
                                                                       q_init = q_T,
                                                                       max_iter = 1)
        
        print('elapsed time = {0}\n'.format(timedelta(seconds = elapsed_time)))
        
        # Store the result as instance's attributes
        self.V_T, self.q_Tm1 = V_T, q_Tm1
    
    
    def solve_problem_b(self, 
                        eps=3000,
                        N_finer = 1000,
                        A_fix = 19,
                        b_fix = 29):
        
        # prepare the finer grids
        A_finer_grid = np.linspace(self.A_grid[0], self.A_grid[-1], N_finer)
        b_finer_grid = np.linspace(self.b_grid[0], self.b_grid[-1], N_finer)
        
        # interpolate by RBF interpolation
        print("Interpolating V_T by RBF interpolation...\n")
        tic = time.process_time()
        V_intrpl = RBFIntrpl_MeshGrid(x1_grid = self.A_grid,
                                      x2_grid = self.b_grid,
                                      fx = self.V_T,
                                      eps = eps)
        toc = time.process_time()
        print('elapsed time = {0}\n'.format(timedelta(seconds = toc - tic)))
                
        print("Interpolating q_{T-1} by RBF interpolation...\n")
        tic = time.process_time()
        q_intrpl = RBFIntrpl_MeshGrid(x1_grid = self.A_grid,
                                      x2_grid = self.b_grid,
                                      fx = self.q_Tm1,
                                      eps = eps)
        toc = time.process_time()
        print('elapsed time = {0}\n'.format(timedelta(seconds = toc - tic)))
                
        # store the result as instances
        self.b_finer_grid = b_finer_grid
        self.A_finer_grid = A_finer_grid
        self.Q1b_V_intrpl = V_intrpl
        self.Q1b_q_intrpl = q_intrpl
        
        # Find index whose corresponding value is close to A_20 (b_30)
        A_finer_idx = get_nearest_idx(self.A_grid[A_fix], A_finer_grid)
        b_finer_idx = get_nearest_idx(self.b_grid[b_fix], b_finer_grid)
        
        # Calculate the interpolated values
        print("Calculating the interpolated values...\n")
        tic = time.process_time()
        V_of_b_given_A = (V_intrpl(x1 = A_finer_grid[A_finer_idx],
                                   x2 = b_finer_grid)).flatten()
        V_of_A_given_b = (V_intrpl(x1 = A_finer_grid,
                                   x2 = b_finer_grid[b_finer_idx])).flatten()
        q_of_b_given_A = (q_intrpl(x1 = A_finer_grid[A_finer_idx],
                                   x2 = b_finer_grid)).flatten()
        q_of_A_given_b = (q_intrpl(x1 = A_finer_grid,
                                   x2 = b_finer_grid[b_finer_idx])).flatten()
        toc = time.process_time()
        print('elapsed time = {0}\n'.format(timedelta(seconds = toc - tic)))
        
        # Plot result
        fig, ax = plt.subplots(2, 2, figsize=(12, 16))
        ax[0, 0].plot(self.b_grid, self.V_T[A_fix, :],
                  c ='red', lw = 0, marker = "o", label='Original')
        ax[0, 0].plot(b_finer_grid, V_of_b_given_A ,
                      c ='orange', label='Interpolated')
        ax[0, 0].set_xlabel('b')
        ax[0, 0].set_title('$V_T(b | A)$')
        ax[0, 0].set_ylim([-3.9, -3.3])
        ax[0, 0].legend(frameon=False)
        
        ax[0, 1].plot(self.A_grid, self.V_T[:, b_fix],
                      c ='red', lw = 0, marker = "o", label='Original')
        ax[0, 1].plot(A_finer_grid, V_of_A_given_b,
                      c ='orange', label='Interpolated')
        ax[0, 1].set_xlabel('A')
        ax[0, 1].set_title('$V_T(A | b)$')
        ax[0, 1].set_ylim([-3.9, -3.3])
        ax[0, 1].legend(frameon=False)
        
        ax[1, 0].plot(self.b_grid, self.q_Tm1[A_fix, :],
                      c ='red', lw = 0, marker = "o", label='Original')
        ax[1, 0].plot(b_finer_grid, q_of_b_given_A ,
                      c ='orange', label='Interpolated')
        ax[1, 0].set_xlabel("$b'$")
        ax[1, 0].set_title('$q_{T-1}$' + "$(b' | A)$")
        ax[1, 0].set_ylim([-0.05, 1.05])
        ax[1, 0].legend(frameon=False)
        
        ax[1, 1].plot(self.A_grid, self.q_Tm1[:, 29],
                      c ='red', lw = 0, marker = "o", label='Original')
        ax[1, 1].plot(A_finer_grid, q_of_A_given_b,
                      c ='orange', label='Interpolated')
        ax[1, 1].set_xlabel("$A$")
        ax[1, 1].set_title('$q_{T-1}$' + "$(A | b')$")
        ax[1, 1].set_ylim([-0.05, 1.05])
        ax[1, 1].legend(frameon=False)
        
        plt.savefig('Q1(b).png', dpi = 150, bbox_inches='tight', pad_inches=0)
        
        
    def solve_problem_c(self, 
                        N_finer = 1000,
                        A_fix = 19,
                        b_fix = 29):
        b_finer_grid = self.b_finer_grid
        A_finer_grid = self.A_finer_grid
        
        # interpolate by RBF interpolation
        print("Interpolating V_T by piecewise interpolation...\n")
        tic = time.process_time()
        V_intrpl = PiecewiseIntrpl_MeshGrid(x1_grid = self.A_grid,
                                            x2_grid = self.b_grid,
                                            fx = self.V_T)
        toc = time.process_time()
        print('elapsed time = {0}\n'.format(timedelta(seconds = toc - tic)))
        tic = time.process_time()
        print("Interpolating q_{T-1} by piecewise interpolation...\n")
        q_intrpl = PiecewiseIntrpl_MeshGrid(x1_grid = self.A_grid,
                                            x2_grid = self.b_grid,
                                            fx = self.q_Tm1)
        toc = time.process_time()
        print('elapsed time = {0}\n'.format(timedelta(seconds = toc - tic)))
        
        # store the result as instances
        self.Q1c_V_intrpl = V_intrpl
        self.Q1c_q_intrpl = q_intrpl
        
        # Find index whose corresponding value is close to A_20 (b_30)
        A_finer_idx = get_nearest_idx(self.A_grid[A_fix], A_finer_grid)
        b_finer_idx = get_nearest_idx(self.b_grid[b_fix], b_finer_grid)
        
        # Calculate the interpolated values
        print("Calculating the interpolated values...\n")
        tic = time.process_time()
        V_of_b_given_A = (V_intrpl(x1 = A_finer_grid[A_finer_idx],
                                   x2 = b_finer_grid)).flatten()
        V_of_A_given_b = (V_intrpl(x1 = A_finer_grid,
                                   x2 = b_finer_grid[b_finer_idx])).flatten()
        q_of_b_given_A = (q_intrpl(x1 = A_finer_grid[A_finer_idx],
                                   x2 = b_finer_grid)).flatten()
        q_of_A_given_b = (q_intrpl(x1 = A_finer_grid,
                                   x2 = b_finer_grid[b_finer_idx])).flatten()
        toc = time.process_time()
        print('elapsed time = {0}\n'.format(timedelta(seconds = toc - tic)))
                
        # Plot result
        fig, ax = plt.subplots(2, 2, figsize=(12, 16))
        ax[0, 0].plot(self.b_grid, self.V_T[A_fix, :],
                  c ='red', lw = 0, marker = "o", label='Original')
        ax[0, 0].plot(b_finer_grid, V_of_b_given_A ,
                      c ='orange', label='Interpolated')
        ax[0, 0].set_xlabel('b')
        ax[0, 0].set_title('$V_T(b | A)$')
        ax[0, 0].set_ylim([-3.9, -3.3])
        ax[0, 0].legend(frameon=False)
        
        ax[0, 1].plot(self.A_grid, self.V_T[:, b_fix],
                      c ='red', lw = 0, marker = "o", label='Original')
        ax[0, 1].plot(A_finer_grid, V_of_A_given_b,
                      c ='orange', label='Interpolated')
        ax[0, 1].set_xlabel('A')
        ax[0, 1].set_title('$V_T(A | b)$')
        ax[0, 1].set_ylim([-3.9, -3.3])
        ax[0, 1].legend(frameon=False)
        
        ax[1, 0].plot(self.b_grid, self.q_Tm1[A_fix, :],
                      c ='red', lw = 0, marker = "o", label='Original')
        ax[1, 0].plot(b_finer_grid, q_of_b_given_A ,
                      c ='orange', label='Interpolated')
        ax[1, 0].set_xlabel("$b'$")
        ax[1, 0].set_title('$q_{T-1}$' + "$(b' | A)$")
        ax[1, 0].set_ylim([-0.05, 1.05])
        ax[1, 0].legend(frameon=False)
        
        ax[1, 1].plot(self.A_grid, self.q_Tm1[:, 29],
                      c ='red', lw = 0, marker = "o", label='Original')
        ax[1, 1].plot(A_finer_grid, q_of_A_given_b,
                      c ='orange', label='Interpolated')
        ax[1, 1].set_xlabel("$A$")
        ax[1, 1].set_title('$q_{T-1}$' + "$(A | b')$")
        ax[1, 1].set_ylim([-0.05, 1.05])
        ax[1, 1].legend(frameon=False)
        
        plt.savefig('Q1(c).png', dpi = 150, bbox_inches='tight', pad_inches=0)
        
    def solve_problem_d(self,
                        A_fix = 19,
                        b_fix = 29):
        # Find index whose corresponding value is close to A_20 (b_30)
        b_finer_grid = self.b_finer_grid
        A_finer_grid = self.A_finer_grid
        A_finer_idx = get_nearest_idx(self.A_grid[A_fix], A_finer_grid)
        b_finer_idx = get_nearest_idx(self.b_grid[b_fix], b_finer_grid)
        A20 = A_finer_grid[A_finer_idx]
        b30 = b_finer_grid[b_finer_idx] 
        
        # Partial derivatives: RBF
        print("Calculating the partial derivatives with RBF interpolation...\n")
        tic = time.process_time()
        V_prime_of_b_RBF = self.Q1b_V_intrpl.calc_partial_derivative(A20, self.b_finer_grid, dim=1)
        V_prime_of_A_RBF = self.Q1b_V_intrpl.calc_partial_derivative(self.A_finer_grid, b30, dim=0)
        q_prime_of_b_RBF = self.Q1b_q_intrpl.calc_partial_derivative(A20, self.b_finer_grid, dim=1)
        q_prime_of_A_RBF = self.Q1b_q_intrpl.calc_partial_derivative(self.A_finer_grid, b30, dim=0)
        toc = time.process_time()
        print('elapsed time = {0}\n'.format(timedelta(seconds = toc - tic)))
        
        # Partial derivatives: Piecewise linear
        print("Calculating the partial derivatives with piecewise interpolation...\n")
        tic = time.process_time()
        V_prime_of_b_PWL = self.Q1c_V_intrpl.calc_partial_derivative(A20, self.b_finer_grid, dim=1)
        V_prime_of_A_PWL = self.Q1c_V_intrpl.calc_partial_derivative(self.A_finer_grid, b30, dim=0)
        q_prime_of_b_PWL = self.Q1c_q_intrpl.calc_partial_derivative(A20, self.b_finer_grid, dim=1)
        q_prime_of_A_PWL = self.Q1c_q_intrpl.calc_partial_derivative(self.A_finer_grid, b30, dim=0)
        toc = time.process_time()
        print('elapsed time = {0}\n'.format(timedelta(seconds = toc - tic)))
        
        # Plot result
        fig, ax = plt.subplots(2, 2, figsize=(12, 16))
        ax[0, 0].plot(b_finer_grid, V_prime_of_b_RBF.flatten(),
                      c ='blue', ls='dashed', label='Radial basis function')
        ax[0, 0].plot(b_finer_grid, V_prime_of_b_PWL.flatten(),
                      c ='red', label='Piecewise linear')
        ax[0, 0].set_xlabel('b')
        ax[0, 0].set_title("$V'_T(b | A)$")
        ax[0, 0].set_ylim([-16, 20])
        ax[0, 0].legend(frameon=False)
        
        ax[0, 1].plot(A_finer_grid, V_prime_of_A_RBF.flatten(),
                      c ='blue', ls='dashed', label='Radial basis function')
        ax[0, 1].plot(A_finer_grid, V_prime_of_A_PWL.flatten(),
                      c ='red', label='Piecewise linear')
        ax[0, 1].set_xlabel('A')
        ax[0, 1].set_title("$V'_T(A | b)$")
        ax[0, 1].set_ylim([-16, 20])
        ax[0, 1].legend(frameon=False)
        
        ax[1, 0].plot(b_finer_grid, q_prime_of_b_RBF.flatten(),
                      c ='blue', ls='dashed', label='Radial basis function')
        ax[1, 0].plot(b_finer_grid, q_prime_of_b_PWL.flatten(),
                      c ='red', label='Piecewise linear')
        ax[1, 0].set_xlabel("$b'$")
        ax[1, 0].set_title("$q'_{T-1}(b' | A)$")
        ax[1, 0].set_ylim([-5, 36])
        ax[1, 0].legend(frameon=False)
        
        ax[1, 1].plot(A_finer_grid, q_prime_of_A_RBF.flatten(),
                      c ='blue', ls='dashed', label='Radial basis function')
        ax[1, 1].plot(A_finer_grid, q_prime_of_A_PWL.flatten(),
                      c ='red', label='Piecewise linear')
        ax[1, 1].set_xlabel("$A$")
        ax[1, 1].set_title("$q'_{T-1}(A | b')$")
        ax[1, 1].set_ylim([-5, 36])
        ax[1, 1].legend(frameon=False)     
        
        plt.savefig('Q1(d).png', dpi = 150, bbox_inches='tight', pad_inches=0)
        
        
    def solve_problem_ef(self,
                        A_fix = 19,
                        b_fix = 29):
        
        # i) Ordinary value iteration with discretization
        print("Solving for b'_T by ordinary value iteration with discretization...\n")
        V_Tm1_i, b_T_i, D_Tm1_i, q_Tm2_i, _, _, elapsed_time_i, _ = \
            self.value_func_iter(V_init = self.V_T,
                                 q_init = self.q_Tm1,
                                 max_iter = 1)
        print('elapsed time = {0}\n'.format(timedelta(seconds = elapsed_time_i)))
        self.V_Tm1_i, self.b_T_i, self.q_Tm2_i = V_Tm1_i, b_T_i, q_Tm2_i
        
        # ii) Value iteration with interpolation
        print("Solving for b'_T by value iteration with interpolation...\n")
        V_Tm1_ii, b_T_ii, D_Tm1_ii, q_Tm2_ii, _, _, elapsed_time_ii, _ = \
            self.value_func_iter_w_intrpl(V_init = self.V_T,
                                          q_init = self.q_Tm1,
                                          max_iter = 1)
        print('elapsed time = {0}\n'.format(timedelta(seconds = elapsed_time_ii)))
        self.V_Tm1_ii, self.b_T_ii, self.q_Tm2_ii = V_Tm1_ii, b_T_ii, q_Tm2_ii
        
        # iii) Euler-equation based policy iteration
        print("Solving for b'_T by Eular equation iteration with interpolation...\n")
        V_Tm1_iii, b_T_iii, D_Tm1_iii, q_Tm2_iii, _, _, elapsed_time_iii, _= \
            self.Euler_eq_iter_w_intrpl(V_init = self.V_T,
                                        q_init = self.q_Tm1,
                                        max_iter = 1)
        print('elapsed time = {0}\n'.format(timedelta(seconds = elapsed_time_iii)))
        self.V_Tm1_iii, self.b_T_iii, self.q_Tm2_iii = V_Tm1_iii, b_T_iii, q_Tm2_iii
        
        # Graphics for (e)
        fig1, ax1 = plt.subplots(2, 1, figsize=(12, 16))
        ax1[0].plot(self.b_grid,self.b_grid,
                      lw = 0.75, c = 'gray', label = '45 degree line')        
        ax1[0].plot(self.b_grid, b_T_iii[A_fix, :].flatten(),
                      c = 'blue', ls = 'dashed', lw = 3, label = 'Method (iii)')
        ax1[0].plot(self.b_grid, b_T_ii[A_fix, :].flatten(),
                      c = 'gray', lw = 2, label = 'Method (ii)')
        ax1[0].plot(self.b_grid, b_T_i[A_fix, :].flatten(),
                      c = 'red', label = 'Method (i)')
        ax1[0].set_xlabel("$b_{T-1}$")
        ax1[0].set_title('$b^{*}_{T-1}$' + "$(b_{T-1} | A = A_{20})$")
        ax1[0].set_ylim([-0.1, 0])
        ax1[0].legend(frameon=False)
        
        ax1[1].plot(self.A_grid, b_T_iii[:, b_fix].flatten(),
                      c = 'blue', ls = 'dashed', lw = 3, label = 'Method (iii)')
        ax1[1].plot(self.A_grid, b_T_ii[:, b_fix].flatten(),
                      c = 'gray', lw = 2, label = 'Method (ii)')
        ax1[1].plot(self.A_grid, b_T_i[:, b_fix].flatten(),
                      c = 'red', label = 'Method (i)')
        ax1[1].set_xlabel("$A_{T-1}$")
        ax1[1].set_title('$b^{*}_{T-1}$' + "$(A_{T-1}| b = b_{30})$")
        ax1[1].legend(frameon=False)
        
        plt.savefig('Q1(e).png', dpi = 150, bbox_inches='tight', pad_inches=0)
        
        # Graphics for (f)
        fig, ax2 = plt.subplots(2, 1, figsize=(12, 16))    
        ax2[0].plot(self.b_grid, D_Tm1_iii[A_fix, :].flatten(),
                      c = 'blue', ls = 'dashed', lw = 3, label = 'Method (iii)')
        ax2[0].plot(self.b_grid, D_Tm1_ii[A_fix, :].flatten(),
                      c = 'gray', lw = 2, label = 'Method (ii)')
        ax2[0].plot(self.b_grid, D_Tm1_i[A_fix, :].flatten(),
                      c = 'red', label = 'Method (i)')
        ax2[0].set_xlabel("$b_{T-1}$")
        ax2[0].set_title('$D_{T-1}$' + "$(b_{T-1} | A = A_{20})$")
        ax2[0].set_ylim([-0.05, 1.05])
        ax2[0].legend(frameon=False)
        
        ax2[1].plot(self.A_grid, D_Tm1_iii[:, b_fix].flatten(),
                      c = 'blue', ls = 'dashed', lw = 3, label = 'Method (iii)')
        ax2[1].plot(self.A_grid, D_Tm1_ii[:, b_fix].flatten(),
                      c = 'gray', lw = 2, label = 'Method (ii)')
        ax2[1].plot(self.A_grid, D_Tm1_i[:, b_fix].flatten(),
                      c = 'red', label = 'Method (i)')
        ax2[1].set_xlabel("$A_{T-1}$")
        ax2[1].set_title('$D_{T-1}$' + "$(A_{T-1}| b = b_{30})$")
        ax2[1].set_ylim([-0.05, 1.05])
        ax2[1].legend(frameon=False)
        
        plt.savefig('Q1(f).png', dpi = 150, bbox_inches='tight', pad_inches=0)
        
        
    def solve_problem_2(self,
                        method = ['method1', 'method2', 'method3'],
                        A_fix = (2, 12, 22),
                        b_fix = (9, 29, 49)
                        ):
        
        if 'method1' in method:
            # i) Ordinary value iteration with discretization
            print("Solving the model by ordinary value iteration with discretization...\n")
            _, b_i, _, q_i, _, _, elapsed_time_i, iteration_i = \
                self.value_func_iter(V_init = self.V_Tm1_i,
                                     q_init = self.q_Tm2_i,
                                     max_iter = 10000
                                     )
            print('elapsed time = {0}'.format(timedelta(seconds = elapsed_time_i)))
            print('# of iteration = {0}\n'.format(iteration_i))    
            
            # Graphics
            fig1, ax1 = plt.subplots(3, 1, figsize=(12, 16))
            ax1[0].plot(self.b_grid,self.b_grid,
                        lw = 0.75, c = 'gray', label = '45 degree line')        
            ax1[0].plot(self.b_grid, b_i[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax1[0].plot(self.b_grid, b_i[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax1[0].plot(self.b_grid, b_i[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax1[0].set_xlabel("$b$")
            ax1[0].set_title('$b^{*}$' + "$(b | A)$")
            ax1[0].set_ylim([-0.1, 0])
            ax1[0].legend(frameon=False)
            
            ax1[1].plot(self.A_grid, b_i[:, b_fix[0]].flatten(),
                          c = 'blue', ls = 'dashed', lw = 3, label = 'Low $b$')
            ax1[1].plot(self.A_grid, b_i[:, b_fix[1]].flatten(),
                          c = 'gray', lw = 2, label = 'Mid $b$')
            ax1[1].plot(self.A_grid, b_i[:, b_fix[2]].flatten(),
                          c = 'red', label = 'High $b$')
            ax1[1].set_xlabel("$A$")
            ax1[1].set_title('$b^{*}$' + "$(A | b)$")
            ax1[1].legend(frameon=False)
            
            ax1[2].plot(self.b_grid, q_i[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax1[2].plot(self.b_grid, q_i[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax1[2].plot(self.b_grid, q_i[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax1[2].set_xlabel("$b'$")
            ax1[2].set_title('$q$' + "$(b' | A)$")
            ax1[2].legend(frameon=False)
            plt.savefig('Q2_method1.png', dpi = 150, bbox_inches='tight', pad_inches=0)
            
        if 'method2' in method:
            # ii) Value iteration with interpolation
            print("Solving the model by value iteration with interpolation...\n")
            _, b_ii, _, q_ii, _, _, elapsed_time_ii, iteration_ii = \
                self.value_func_iter_w_intrpl(V_init = self.V_Tm1_ii,
                                              q_init = self.q_Tm2_ii,
                                              max_iter = 10000)
            print('elapsed time = {0}'.format(timedelta(seconds = elapsed_time_ii)))
            print('# of iteration = {0}\n'.format(iteration_ii))    
            
            # Graphics
            fig2, ax2 = plt.subplots(3, 1, figsize=(12, 16))
            ax2[0].plot(self.b_grid,self.b_grid,
                        lw = 0.75, c = 'gray', label = '45 degree line')        
            ax2[0].plot(self.b_grid, b_ii[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax2[0].plot(self.b_grid, b_ii[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax2[0].plot(self.b_grid, b_ii[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax2[0].set_xlabel("$b$")
            ax2[0].set_title('$b^{*}$' + "$(b | A)$")
            ax2[0].set_ylim([-0.1, 0])
            ax2[0].legend(frameon=False)
            
            ax2[1].plot(self.A_grid, b_ii[:, b_fix[0]].flatten(),
                          c = 'blue', ls = 'dashed', lw = 3, label = 'Low $b$')
            ax2[1].plot(self.A_grid, b_ii[:, b_fix[1]].flatten(),
                          c = 'gray', lw = 2, label = 'Mid $b$')
            ax2[1].plot(self.A_grid, b_ii[:, b_fix[2]].flatten(),
                          c = 'red', label = 'High $b$')
            ax2[1].set_xlabel("$A$")
            ax2[1].set_title('$b^{*}$' + "$(A | b)$")
            ax2[1].legend(frameon=False)
            
            ax2[2].plot(self.b_grid, q_ii[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax2[2].plot(self.b_grid, q_ii[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax2[2].plot(self.b_grid, q_ii[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax2[2].set_xlabel("$b'$")
            ax2[2].set_title('$q$' + "$(b' | A)$")
            ax2[2].legend(frameon=False)
            plt.savefig('Q2_method2.png', dpi = 150, bbox_inches='tight', pad_inches=0)
            
        if 'method3' in method:
            # iii) Euler-equation based policy iteration
            print("Solving the model by Eular equation iteration with interpolation...\n")
            _, b_iii, _, q_iii, _, _, elapsed_time_iii, iteration_iii = \
                self.Euler_eq_iter_w_intrpl(V_init = self.V_Tm1_iii,
                                            q_init = self.q_Tm2_iii,
                                            max_iter = 10000)
            print('elapsed time = {0}'.format(timedelta(seconds = elapsed_time_iii)))
            print('# of iteration = {0}\n'.format(iteration_iii))    
            
            # Graphics
            fig3, ax3 = plt.subplots(3, 1, figsize=(12, 16))
            ax3[0].plot(self.b_grid,self.b_grid,
                        lw = 0.75, c = 'gray', label = '45 degree line')        
            ax3[0].plot(self.b_grid, b_iii[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax3[0].plot(self.b_grid, b_iii[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax3[0].plot(self.b_grid, b_iii[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax3[0].set_xlabel("$b$")
            ax3[0].set_title('$b^{*}$' + "$(b | A)$")
            ax3[0].set_ylim([-0.1, 0])
            ax3[0].legend(frameon=False)
            
            ax3[1].plot(self.A_grid, b_iii[:, b_fix[0]].flatten(),
                          c = 'blue', ls = 'dashed', lw = 3, label = 'Low $b$')
            ax3[1].plot(self.A_grid, b_iii[:, b_fix[1]].flatten(),
                          c = 'gray', lw = 2, label = 'Mid $b$')
            ax3[1].plot(self.A_grid, b_iii[:, b_fix[2]].flatten(),
                          c = 'red', label = 'High $b$')
            ax3[1].set_xlabel("$A$")
            ax3[1].set_title('$b^{*}$' + "$(A | b)$")
            ax3[1].legend(frameon=False)
            
            ax3[2].plot(self.b_grid, q_iii[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax3[2].plot(self.b_grid, q_iii[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax3[2].plot(self.b_grid, q_iii[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax3[2].set_xlabel("$b'$")
            ax3[2].set_title('$q$' + "$(b' | A)$")
            ax3[2].legend(frameon=False)
            plt.savefig('Q2_method3.png', dpi = 150, bbox_inches='tight', pad_inches=0)

    def solve_problem_2_wo_shock(self,
                        method = ['method1', 'method2', 'method3'],
                        A_fix = (2, 12, 22),
                        b_fix = (9, 29, 49)
                        ):
        
        if 'method1' in method:
            # i) Ordinary value iteration with discretization
            print("Solving the model by ordinary value iteration with discretization...\n")
            _, b_i, _, q_i, _, _, elapsed_time_i, iteration_i = \
                self.value_func_iter(max_iter = 10000,
                                     is_wo_shock=True)
            print('elapsed time = {0}'.format(timedelta(seconds = elapsed_time_i)))
            print('# of iteration = {0}\n'.format(iteration_i))            
            # Graphics
            fig1, ax1 = plt.subplots(3, 1, figsize=(12, 16))
            ax1[0].plot(self.b_grid,self.b_grid,
                        lw = 0.75, c = 'gray', label = '45 degree line')        
            ax1[0].plot(self.b_grid, b_i[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax1[0].plot(self.b_grid, b_i[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax1[0].plot(self.b_grid, b_i[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax1[0].set_xlabel("$b$")
            ax1[0].set_title('$b^{*}$' + "$(b | A)$")
            ax1[0].set_ylim([-0.1, 0])
            ax1[0].legend(frameon=False)
            
            ax1[1].plot(self.A_grid, b_i[:, b_fix[0]].flatten(),
                          c = 'blue', ls = 'dashed', lw = 3, label = 'Low $b$')
            ax1[1].plot(self.A_grid, b_i[:, b_fix[1]].flatten(),
                          c = 'gray', lw = 2, label = 'Mid $b$')
            ax1[1].plot(self.A_grid, b_i[:, b_fix[2]].flatten(),
                          c = 'red', label = 'High $b$')
            ax1[1].set_xlabel("$A$")
            ax1[1].set_title('$b^{*}$' + "$(A | b)$")
            ax1[1].legend(frameon=False)
            
            ax1[2].plot(self.b_grid, q_i[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax1[2].plot(self.b_grid, q_i[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax1[2].plot(self.b_grid, q_i[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax1[2].set_xlabel("$b'$")
            ax1[2].set_title('$q$' + "$(b' | A)$")
            ax1[2].legend(frameon=False)
            plt.savefig('Q2_method1_wo_shock.png', dpi = 150, bbox_inches='tight', pad_inches=0)
                        
        if 'method2' in method:
            # ii) Value iteration with interpolation
            print("Solving the model by value iteration with interpolation...\n")
            _, b_ii, _, q_ii, _, _, elapsed_time_ii, iteration_ii = \
                self.value_func_iter_w_intrpl(max_iter = 10000,
                                              is_wo_shock = True)
            print('elapsed time = {0}'.format(timedelta(seconds = elapsed_time_ii)))
            print('# of iteration = {0}\n'.format(iteration_ii))
                        
            # Graphics
            fig2, ax2 = plt.subplots(3, 1, figsize=(12, 16))
            ax2[0].plot(self.b_grid,self.b_grid,
                        lw = 0.75, c = 'gray', label = '45 degree line')        
            ax2[0].plot(self.b_grid, b_ii[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax2[0].plot(self.b_grid, b_ii[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax2[0].plot(self.b_grid, b_ii[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax2[0].set_xlabel("$b$")
            ax2[0].set_title('$b^{*}$' + "$(b | A)$")
            ax2[0].set_ylim([-0.1, 0])
            ax2[0].legend(frameon=False)
            
            ax2[1].plot(self.A_grid, b_ii[:, b_fix[0]].flatten(),
                          c = 'blue', ls = 'dashed', lw = 3, label = 'Low $b$')
            ax2[1].plot(self.A_grid, b_ii[:, b_fix[1]].flatten(),
                          c = 'gray', lw = 2, label = 'Mid $b$')
            ax2[1].plot(self.A_grid, b_ii[:, b_fix[2]].flatten(),
                          c = 'red', label = 'High $b$')
            ax2[1].set_xlabel("$A$")
            ax2[1].set_title('$b^{*}$' + "$(A | b)$")
            ax2[1].legend(frameon=False)
            
            ax2[2].plot(self.b_grid, q_ii[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax2[2].plot(self.b_grid, q_ii[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax2[2].plot(self.b_grid, q_ii[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax2[2].set_xlabel("$b'$")
            ax2[2].set_title('$q$' + "$(b' | A)$")
            ax2[2].legend(frameon=False)
            plt.savefig('Q2_method2_wo_shock.png', dpi = 150, bbox_inches='tight', pad_inches=0)
            
        if 'method3' in method:
            # iii) Euler-equation based policy iteration
            print("Solving the model by Eular equation iteration with interpolation...\n")
            _, b_iii, _, q_iii, _, _, elapsed_time_iii, iteration_iii = \
                self.Euler_eq_iter_w_intrpl(max_iter = 10000,
                                            is_wo_shock = True)
            print('elapsed time = {0}'.format(timedelta(seconds = elapsed_time_iii)))
            print('# of iteration = {0}\n'.format(iteration_iii))
                        
            # Graphics
            fig3, ax3 = plt.subplots(3, 1, figsize=(12, 16))
            ax3[0].plot(self.b_grid,self.b_grid,
                        lw = 0.75, c = 'gray', label = '45 degree line')        
            ax3[0].plot(self.b_grid, b_iii[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax3[0].plot(self.b_grid, b_iii[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax3[0].plot(self.b_grid, b_iii[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax3[0].set_xlabel("$b$")
            ax3[0].set_title('$b^{*}$' + "$(b | A)$")
            ax3[0].set_ylim([-0.1, 0])
            ax3[0].legend(frameon=False)
            
            ax3[1].plot(self.A_grid, b_iii[:, b_fix[0]].flatten(),
                          c = 'blue', ls = 'dashed', lw = 3, label = 'Low $b$')
            ax3[1].plot(self.A_grid, b_iii[:, b_fix[1]].flatten(),
                          c = 'gray', lw = 2, label = 'Mid $b$')
            ax3[1].plot(self.A_grid, b_iii[:, b_fix[2]].flatten(),
                          c = 'red', label = 'High $b$')
            ax3[1].set_xlabel("$A$")
            ax3[1].set_title('$b^{*}$' + "$(A | b)$")
            ax3[1].legend(frameon=False)
            
            ax3[2].plot(self.b_grid, q_iii[A_fix[0], :].flatten(),
                        c = 'blue', ls = 'dashed', lw = 3, label = 'Low $A$')
            ax3[2].plot(self.b_grid, q_iii[A_fix[1], :].flatten(),
                        c = 'gray', lw = 2, label = 'Mid $A$')
            ax3[2].plot(self.b_grid, q_iii[A_fix[2], :].flatten(),
                        c = 'red', label = 'High $A$')
            ax3[2].set_xlabel("$b'$")
            ax3[2].set_title('$q$' + "$(b' | A)$")
            ax3[2].legend(frameon=False)
            plt.savefig('Q2_method3_wo_shock.png', dpi = 150, bbox_inches='tight', pad_inches=0)