#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW5.py

is the python class for the assignment #4 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Oct 5, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
from math import isqrt
from this import d
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import time
from copy import deepcopy
from scipy.stats import norm
from scipy.optimize import fsolve, root, bisect


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
    
    def calc_partial_derivative(self, x1, x2, dim=0, dx=0.001):
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
                print("\n Discretizing the AR(1) process by Tauchen method")
            self._tauchen_discretize(N, Omega, is_write_out_result)
        elif method in ['rouwenhorst', 'Rouwenhorst', 'R', 'r']:
            if not is_quiet:
                print("\n Discretizing the income process by Rouwenhorst method")
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
                               is_write_out_result = True,
                               is_quiet = False): 
        
        self.lnA_AR1.discretize(method = method,
                                N = self.N_A,
                                Omega = self.Omega,
                                is_write_out_result = is_write_out_result,
                                is_quiet = is_quiet)
        
        A_grid = np.exp(self.lnA_AR1.lnA_grid)
        self.trans_mat = deepcopy(self.lnA_AR1.trans_mat)
        self.A_grid = A_grid
    
    def VD(self, A_idx, V_tmrw): # value of default today
        # today's productivity level
        A = self.A_grid[A_idx]
        # The instantaneous payoff
        inst_payoff = self.utility(A = A, 
                                   b = 0, 
                                   b_prime = 0, 
                                   q = 0,
                                   isdefault = True)
        # expected value
        E_value = self.trans_mat[A_idx, :] @ V_tmrw[:, -1]
        # The value of default today
        V_D = inst_payoff + self.beta * E_value
        return V_D
    
    def VG(self, A_idx, b_idx, q, V_tmrw): # value of repaying today
        # today's productivity level and borrowing
        A = self.A_grid[A_idx]
        b = self.b_grid[b_idx]
        # Possible borrowing
        b_prime_vec = self.b_grid.reshape(1, -1)
        # instantaneous payoff
        inst_payoff_vec = self.utility(A = A, 
                                       b = b, 
                                       b_prime = b_prime_vec, 
                                       q = q)
        # expected value
        E_value_vec = self.trans_mat[A_idx, :] @ V_tmrw
        # possible VGs
        VG_vec = inst_payoff_vec + self.beta * E_value_vec
        # take the maximum
        V_G, argmax_idx = np.nanmax(VG_vec), np.nanargmax(VG_vec)
        return V_G, b_prime_vec[argmax_idx]
    
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
                inside_u = 1E-5
        else:
            inside_u[inside_u <= 0] = 1E-5
        # calculate utility
        u = inside_u**(1 - self.gamma) / (1 - self.gamma)
        return u
    
    def E_V(self, V_G, V_D):
        E_V = self.kappa * np.log(
            np.exp(V_G/self.kappa) + np.exp(V_D/self.kappa)
            )
        return E_V 
    
    def D_star(self, V_G, V_D): # default rate
        expVD = np.exp(V_D/self.kappa)
        expVG = np.exp(V_G/self.kappa)
        D_star = expVD / (expVD + expVG)
        return D_star
    
    def q(self, A_idx, b_idx, D_star_prime):
        D_star_prime_vec = D_star_prime[:, b_idx]
        q = self.trans_mat[A_idx, :] @ (1 - D_star_prime_vec) / (1 + self.r)
        return q
    
    def solve_two_period_DP(self):
        # After the world ends, everything is worthless.
        V_terminal = np.zeros((self.N_A, self.N_b))
        q_T = np.zeros((self.N_A, self.N_b))
        
        # Solve for period T -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # Value of default in period T
        VD_T = [self.VD(a_idx, V_terminal) for a_idx in range(self.N_A)]
        VD_T = np.tile(np.array(VD_T).reshape(-1, 1), (1, self.N_b))
        # Value of repaying in period T
        VG_T = [
            [self.VG(a_idx, b_idx, q_T, V_terminal)[0] for b_idx in range(self.N_b)]
            for a_idx in range(self.N_A)
            ]
        VG_T = np.array(VG_T)
        # Maximized value in period T
        V_T = self.E_V(V_G = VG_T, V_D = VD_T)
        # Default rate
        D_T = self.D_star(V_G = VG_T, V_D = VD_T)
        
        # Solve for period T-1 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        q_Tm1 = [
            [self.q(a_idx, b_idx, D_T) for b_idx in range(self.N_b)]
            for a_idx in range(self.N_A)
            ]
        q_Tm1 = np.array(q_Tm1)
        
        # Store the result as instance's attributes
        self.VD_T, self.VG_T, self.V_T, self.D_T, self.q_Tm1 =\
            VD_T, VG_T, V_T, D_T, q_Tm1
            


        
    def solve_problem_b(self, 
                        eps=3000,
                        N_finer = 1000,
                        A_fix = 19,
                        b_fix = 29):
        
        # prepare the finer grids
        A_finer_grid = np.linspace(self.A_grid[0], self.A_grid[-1], N_finer)
        b_finer_grid = np.linspace(self.b_grid[0], self.b_grid[-1], N_finer)
        
        
        # interpolate by RBF interpolation
        V_intrpl = RBFIntrpl_MeshGrid(x1_grid = self.A_grid,
                                      x2_grid = self.b_grid,
                                      fx = self.V_T,
                                      eps = eps)
        q_intrpl = RBFIntrpl_MeshGrid(x1_grid = self.A_grid,
                                      x2_grid = self.b_grid,
                                      fx = self.q_Tm1,
                                      eps = eps)
        
        # store the result as instances
        self.b_finer_grid = b_finer_grid
        self.A_finer_grid = A_finer_grid
        self.Q1b_V_intrpl = V_intrpl
        self.Q1b_q_intrpl = q_intrpl
        
        # Find index whose corresponding value is close to A_20 (b_30)
        A_finer_idx = get_nearest_idx(self.A_grid[A_fix], A_finer_grid)
        b_finer_idx = get_nearest_idx(self.b_grid[b_fix], b_finer_grid)
        
        # Calculate the interpolated values
        V_of_b_given_A = (V_intrpl(x1 = A_finer_grid[A_finer_idx],
                                   x2 = b_finer_grid)).flatten()
        V_of_A_given_b = (V_intrpl(x1 = A_finer_grid,
                                   x2 = b_finer_grid[b_finer_idx])).flatten()
        q_of_b_given_A = (q_intrpl(x1 = A_finer_grid[A_finer_idx],
                                   x2 = b_finer_grid)).flatten()
        q_of_A_given_b = (q_intrpl(x1 = A_finer_grid,
                                   x2 = b_finer_grid[b_finer_idx])).flatten()
        
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
        V_intrpl = PiecewiseIntrpl_MeshGrid(x1_grid = self.A_grid,
                                            x2_grid = self.b_grid,
                                            fx = self.V_T)
        q_intrpl = PiecewiseIntrpl_MeshGrid(x1_grid = self.A_grid,
                                            x2_grid = self.b_grid,
                                            fx = self.q_Tm1)
        
        # store the result as instances
        self.Q1c_V_intrpl = V_intrpl
        self.Q1c_q_intrpl = q_intrpl
        
        # Find index whose corresponding value is close to A_20 (b_30)
        A_finer_idx = get_nearest_idx(self.A_grid[A_fix], A_finer_grid)
        b_finer_idx = get_nearest_idx(self.b_grid[b_fix], b_finer_grid)
        
        # Calculate the interpolated values
        V_of_b_given_A = (V_intrpl(x1 = A_finer_grid[A_finer_idx],
                                   x2 = b_finer_grid)).flatten()
        V_of_A_given_b = (V_intrpl(x1 = A_finer_grid,
                                   x2 = b_finer_grid[b_finer_idx])).flatten()
        q_of_b_given_A = (q_intrpl(x1 = A_finer_grid[A_finer_idx],
                                   x2 = b_finer_grid)).flatten()
        q_of_A_given_b = (q_intrpl(x1 = A_finer_grid,
                                   x2 = b_finer_grid[b_finer_idx])).flatten()
        
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
        V_prime_of_b_RBF = self.Q1b_V_intrpl.calc_partial_derivative(A20, self.b_finer_grid, dim=1)
        V_prime_of_A_RBF = self.Q1b_V_intrpl.calc_partial_derivative(self.A_finer_grid, b30, dim=0)
        q_prime_of_b_RBF = self.Q1b_q_intrpl.calc_partial_derivative(A20, self.b_finer_grid, dim=1)
        q_prime_of_A_RBF = self.Q1b_q_intrpl.calc_partial_derivative(self.A_finer_grid, b30, dim=0)

        # Partial derivatives: Piecewise linear
        V_prime_of_b_PWL = self.Q1c_V_intrpl.calc_partial_derivative(A20, self.b_finer_grid, dim=1)
        V_prime_of_A_PWL = self.Q1c_V_intrpl.calc_partial_derivative(self.A_finer_grid, b30, dim=0)
        q_prime_of_b_PWL = self.Q1c_q_intrpl.calc_partial_derivative(A20, self.b_finer_grid, dim=1)
        q_prime_of_A_PWL = self.Q1c_q_intrpl.calc_partial_derivative(self.A_finer_grid, b30, dim=0)
        
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