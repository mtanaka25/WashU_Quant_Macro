#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW9.py

is the python class for the assignment #9 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Nov 9, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tabulate import tabulate
from scipy.stats import norm
from scipy.optimize import bisect

# Load the personal Python package, which is based on the assignments #1-#7.
from mtPyTools import StopWatch
from mtPyEcon import lorenz_curve
def_tol = 1E-8
def_max_iter = 1000

class Quasi_Tauchen:
    def __init__(self,
                 x_grid = np.array([2.32, 3.23, 4.24, 5.89, 7.75]),
                 take_log = True,
                 a0  = 0.035,
                 a1  = 0.970,
                 sig = 0.110):
        """
        Parameters
        ----------
        x_grid : numpy array, optional
            grid points, by default np.array([2.32, 3.23, 4.24, 5.89, 7.75])
        take_log : bool, optional
            If True, x_grid will be taken log, by default True
        a0 : float, optional
            constant in x process, by default 0.035
        a1 : float, optional
            coeficient in z process, by default 0.970
        sig : float, optional
            standard deviation of the shock on x, by default 0.110
        """
        if take_log:
            x_grid = np.log(x_grid)
        self.x_grid = x_grid
        self.a0, self.a1, self.sig = a0, a1, sig
    
    def __call__(self, write_out=False):
        self.discretize(write_out = write_out)
        return self.trans_mat
    
    def discretize(self, write_out=False):
        # simplify variable names
        x_grid, a0, a1, sig = self.x_grid, self.a0, self.a1, self.sig
        # number of grid points
        N = len(x_grid)
        # Calculate the step sizes
        h = [x_grid[i+1] - x_grid[i] for i in range(N-1)]
        def trans_mat_ij(i, j):
            deterministic_val = a0 + a1*x_grid[i]
            if j == 0:
                trans_mat_ij = norm.cdf((x_grid[j] - deterministic_val + h[0]/2)/sig)
            elif j == (N-1):
                trans_mat_ij = 1 - norm.cdf((x_grid[j] - deterministic_val - h[-1]/2)/sig)
            else:
                trans_mat_ij = ( norm.cdf((x_grid[j] - deterministic_val + h[j]/2)/sig)
                                - norm.cdf((x_grid[j] - deterministic_val - h[j-1]/2)/sig))
            return trans_mat_ij
        # Construct the transition matrix
        trans_mat = [
            [trans_mat_ij(i, j) for j in range(N)]
            for i in range(N)
            ]
        # Write out the transition matrix in a csv file (if necessary)
        if write_out:
            np.savetxt('trans_mat.csv', trans_mat, delimiter=' & ',
                       fmt='%2.3f', newline=' \\\\\n')
        # Store the result as the instance's attributes
        self.trans_mat = np.array(trans_mat)

class Hopenhayn1992:
    """
    This class is for the model proposed by Hopenhayn (1992, econometrica)
    """
    def __init__(self,
                 z_grid = np.array([2.32, 3.23, 4.24, 5.89, 7.75]),
                 trans_mat = np.array([
                     [0.80, 0.15, 0.05, 0.00, 0.00],
                     [0.15, 0.65, 0.15, 0.05, 0.00],
                     [0.05, 0.15, 0.60, 0.15, 0.05],
                     [0.00, 0.05, 0.15, 0.65, 0.15],
                     [0.00, 0.00, 0.05, 0.15, 0.80]
                     ]),
                 G = np.array([0.720, 0.140, 0.075, 0.045, 0.020]),
                 alpha = 0.64,
                 beta = 0.94,
                 cf = 20.,
                 ce = 36.2,
                 kappa = 0.5,
                 b = 1.,
                 ):
        """
        This class is for the model proposed by Hopenhayn (1992, econometrica)
        Parameters
        ----------
        z_grid : numpy array optional
            grid points for z,
            by default
                [2.32, 3.23, 4.24, 5.89, 7.75]
        trans_mat : numpy array, optional
            transition matrix of z,
            by default
                [[0.80, 0.15, 0.05, 0.00, 0.00],
                 [0.15, 0.65, 0.15, 0.05, 0.00],
                 [0.05, 0.15, 0.60, 0.15, 0.05],
                 [0.00, 0.05, 0.15, 0.65, 0.15],
                 [0.00, 0.00, 0.05, 0.15, 0.80]]
        G : numpy array, optional
            pdf for initial z
            by default
                [0.720, 0.140, 0.075, 0.045, 0.020]
        alpha : float, optional
            concavity of production function, by default 0.64
        beta : float, optional
            discount factor, by default 0.94
        cf : float, optional
            fixed cost to operate, by default 20.
        ce : float, optional
            fixed cost to entry, by default 36.2
        kappa : float, optional
            shape parameter of Gumbel distribution, by default 0.5
        b : float, optional
            coefficient in demand function, by default 1.
        """
        self.z_grid, self.trans_mat, self.G = z_grid, trans_mat, G
        self.alpha, self.beta, self.cf, self.ce = alpha, beta, cf, ce
        self.kappa, self.b, self.Nz = kappa, b, len(z_grid)
    
    def __call__(self):
        self.solve_for_p_star()
        self.solve_for_m()
        self.calculate_stationary_stats()
    
    def solve_Bellman_eq(self, V_prime, p, w=1.0):
        """
        solve the Bellman equation for incumbents
        Parameters
        ----------
        V_prime : numpy array
            values tomorrow (conditional on continuing business)
        p : float
            price level
        w : float, optional
            wage rate. If not given, it is considered as numeraire (that is, w = 1.0)
        
        Returns
        -------
        numpy array (float)
            Today's value (under the given V', p, and w)
        numpy array (float)
            Today's optimal labor input (under the given V', p, and w)
        numpy array (float)
            Today's probability of shutting down (under the given V', p, and w)
        """
        # prepare the vector of z
        z_vec = self.z_grid.reshape(-1, 1)
        # Optimal labor input
        n_vec = (w / (self.alpha * p * z_vec))**(1/(self.alpha - 1))
        # profit today
        profit_vec = p * z_vec * n_vec**self.alpha - w * n_vec - self.cf
        # expected value tomorrow (taking expectation wrt Gumbel shocks)
        VV_prime = V_prime + self.kappa * np.log(1 + np.exp(- V_prime / self.kappa))
        # expected value tomorrow (taking expectation wrt z')
        EVV = self.trans_mat @ VV_prime
        # today's value
        V = profit_vec + self.beta * EVV
        # Probability of shutting down
        x_vec = np.exp(- V / self.kappa)/(1 + np.exp(- V / self.kappa))
        return V, n_vec, x_vec
    
    def value_func_iter(self, p, w=1.0, max_iter=def_max_iter, tol=def_tol):
        """
        solve for the stationary V under the given p and w
        Parameters
        ----------
        p : float
            price level
        w : float, optional
            wage rate. If not given, it is considered as numeraire (that is, w = 1.0)
        max_iter : int, optional
            maximum number of iterations, by default 1000
        tol : float, optional
            tolerance level in convergence check, by default 1E-5
        
        Returns
        -------
        numpy array (float)
            Today's value (under the given p and w)
        numpy array (float)
            Today's optimal labor input (under the given p and w)
        numpy array (float)
            Today's probability of shutting down (under the given p and w)
        """
        # initialize the while loop
        diff = tol + 1.
        iteration = 0
        V_updated = np.zeros((self.Nz, 1))
        # begin the while loop
        while (iteration < max_iter) & (diff > tol):
            V_guess = np.copy(V_updated)
            # Update V by solving the Bellman equation
            V_updated, policy_n, prob_shutdown = \
                self.solve_Bellman_eq(p = p, w = w, V_prime = V_guess)
            # Check convergence
            diff = max(abs(V_updated - V_guess))
            iteration += 1
        if diff > tol:
            print(f'V did not converge within {max_iter} iterations.')
        # Return the converged result
        return V_updated, policy_n, prob_shutdown
    
    def Ve(self, p, w=1.0, max_iter=def_max_iter, tol=def_tol):
        """
        Solve for the value of entry, taking p and w as given
        Parameters
        ----------
        p : float
            price level
        w : float, optional
            wage rate. If not given, it is considered as numeraire (that is, w = 1.0)
        max_iter : int, optional
            maximum number of iterations, by default 1000
        tol : float, optional
            tolerance level in convergence check, by default 1E-5

        Returns
        -------
        float
            value of entry
        """
        # solve for the stationary V
        V_prime, _, _ = self.value_func_iter(p, w, max_iter, tol)
        # expected value after entry
        VV_prime = V_prime + self.kappa * np.log(1 + np.exp(- V_prime / self.kappa))
        EVV = self.G.reshape(1, -1) @ VV_prime
        # value of entry
        return -self.ce + self.beta * EVV
    
    def solve_for_p_star(self,
                         p_a  = 0.25,
                         p_b  = 4.00,
                         w    = 1.0,
                         xtol_bisec = 2e-12,
                         rtol_bisec = 8.881784197001252e-16,
                         max_iter_bisec = 100,
                         tol  = def_tol,
                         max_iter = def_max_iter,
                         display = False
                         ):
        if display:
            print('Starting bisection...')
            stopwatch = StopWatch()
        p_star, stats = bisect(self.Ve,
                                a = p_a,
                                b = p_b,
                                args = (w, max_iter, tol),
                                xtol = xtol_bisec,
                                rtol = rtol_bisec,
                                maxiter = max_iter_bisec,
                                full_output = True
                                )
        if display:
            stopwatch.stop()
            print(f'Bisection took {stats.iterations} iterations.')
            print(f'p_star = {p_star:.4f}')
        if not stats.converged:
            print('Bisection failed to find the root within the given max_iter.')
        # calculate the value and policy function consistent with p*
        V_star, n_star, x_star = self.value_func_iter(p_star,
                                                      w = w,
                                                      max_iter = max_iter,
                                                      tol = tol)
        # store the result as instance attributes
        self.p_star, self.V_star, self.n_star, self.x_star =\
            p_star, V_star, n_star, x_star
    
    def solve_for_stationary_dist(self,
                                  m = 0.25,
                                  max_iter = def_max_iter,
                                  tol = def_tol,
                                  ):
        # Load transition matrix (and transpose it)
        trans_mat = self.trans_mat.T
        # Load the z distribution for the entrants (as a vertical vector)
        G = self.G.reshape(-1, 1)
        # Prepare a guess for the distribution (use G here)
        dist_updated = np.copy(G)
        # initialize the while loop
        diff = tol + 1.
        iteration = 0
        while (iteration < max_iter) & (diff > tol):
            dist_guess = np.copy(dist_updated)
            # update the distribution
            dist_updated = trans_mat @ ((1 - self.x_star) * dist_guess) + m * G
            # check convergence
            diff = max(abs(dist_updated - dist_guess))
            iteration += 1
        if diff > tol:
            print(f'Failed to find the stationary distribution within {max_iter}.')
        return dist_updated
    
    def solve_for_m(self,
                    m_init = 0.25,
                    max_iter = def_max_iter,
                    tol = def_tol):
        # calculate the aggregate demand
        D = self.b / self.p_star
        # calculate the stationary distribution under m_init
        dist_init = self.solve_for_stationary_dist(m = m_init,
                                                   max_iter = max_iter,
                                                   tol = tol)
        # optimal output for each z
        y_vec = self.z_grid.reshape(1, -1) * self.n_star.reshape(1, -1)**self.alpha
        # calculate the aggregate supply
        Y = y_vec @ ((1 - self.x_star)*dist_init)
        Y = Y.item()  # just transform from numpy scalar to python scalar
        # adjust m and distribution so that supply and demand are balanced
        m_star = m_init * D / Y
        dist_star = dist_init * D / Y
        # store the result as instance attributes
        self.m_star, self.stationary_dist = m_star, dist_star
    
    def calculate_stationary_stats(self, w=1., write_out=False):
        # number of firms
        total_mass = np.sum(self.stationary_dist)
        # diistribution of operating firms
        dist_operating_firms = (1 - self.x_star)*self.stationary_dist
        # number of operating firms
        operating_mass = np.sum(dist_operating_firms)
        # optimal output for each z
        y_vec = self.z_grid.reshape(1, -1) * self.n_star.reshape(1, -1)**self.alpha
        # aggregate output
        Y_star = y_vec @ dist_operating_firms
        Y_star = Y_star.item() # just transform from numpy scalar to python scalar
        # aggregate employment
        N_star = self.n_star.T @ dist_operating_firms
        N_star = N_star.item() # just transform from numpy scalar to python scalar
        # labor productivity
        A_star = Y_star/N_star
        # entry and exit rates
        exit_firms = self.x_star.T @ self.stationary_dist
        exit_firms = exit_firms.item() # just transform from numpy scalar to python scalar
        entry_rate = self.m_star / total_mass
        exit_rate  = exit_firms / total_mass
        # aggegate profit
        Pi_star = self.p_star * Y_star - w * N_star - self.cf * operating_mass
        # average firm size
        ave_size = N_star / operating_mass
        # Lorenz curve wrt employment
        lorenz_curve_for_n = lorenz_curve(fx = self.n_star.flatten(),
                                          x_dist = dist_operating_firms.flatten())
        if write_out:
            # Combine the stats
            table_data = [['p*'                          , self.p_star],
                          ['Total output'                , Y_star     ],
                          ['Total employment'            , N_star     ],
                          ['Aggregate labor productivity', A_star     ],
                          ['Aggregate profits'           , Pi_star    ],
                          ['Entry rate'                  , entry_rate ],
                          ['Exit rate'                   , exit_rate  ],
                          ['Average firm size'           , ave_size   ]]
            # Write out the table
            latex_table_code = tabulate(table_data, ['',''],
                                        tablefmt = 'latex',
                                        floatfmt = '.4f')
            f = open('latex_code_for_Q1e.txt', 'w')
            f.write(latex_table_code)
            f.close()
            # Plot the Lorenz curve
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(lorenz_curve_for_n[0], lorenz_curve_for_n[0],
                    c = 'gray', lw = 0.5, label = '45 degree line')
            ax.plot(lorenz_curve_for_n[0], lorenz_curve_for_n[1],
                    c = '#7BA23F', marker = 'o', label= 'Lorenz curve')
            ax.set_xlabel('Cumulative share of firms')
            ax.legend(frameon=False)
            plt.savefig('figQ1e.png', dpi = 100, bbox_inches='tight', pad_inches=0)
        # Store the stats as the instance attributes
        self.Y_star, self.N_star = Y_star, N_star
        self.A_star, self.Pi_star = A_star, Pi_star
        self.entry_rate, self.exit_rate = entry_rate, exit_rate
        self.ave_size, self.lorenz_curve_for_n = ave_size, lorenz_curve_for_n
    
    def solve_question_1a(self, p = 1.0):
        value, labor, shutdown = self.value_func_iter(p = p)
        # plot the result
        fig, ax = plt.subplots(3, 1, figsize=(8, 12))
        ax[0].plot(self.z_grid, value.flatten(),
                   c = '#7BA23F', marker = 'o')
        ax[0].set_ylabel('$V$')
        ax[1].plot(self.z_grid, labor.flatten(),
                   c = '#7BA23F', marker = 'o')
        ax[1].set_ylabel('Optimal labor input')
        ax[2].plot(self.z_grid, shutdown.flatten(),
                   c = '#7BA23F', marker = 'o')
        ax[2].set_xlabel('$z$')
        ax[2].set_ylabel('Prob. of shutting down')
        ax[2].set_ylim(-0.2, 1.2)
        ax[2].set_yticks([0., 0.25, 0.5, 0.75, 1.0])
        plt.savefig('figQ1a.png', dpi = 100, bbox_inches='tight', pad_inches=0)
    
    def solve_question_1b(self,
                          p_vec = [0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 4.0]):
        # Calculate Ve for given p
        Ve_vec = np.zeros((len(p_vec), ))
        for i, p_i in enumerate(p_vec):
            Ve_vec[i] = self.Ve(p = p_i)
        # plot the calculated Ve
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(np.array(p_vec), Ve_vec,
                c = '#7BA23F', marker = 'o')
        ax.set_xlabel('$p$')
        ax.set_ylabel('$V^e$')
        plt.savefig('figQ1b.png', dpi = 100, bbox_inches='tight', pad_inches=0)
        # set the initial values for bisection
        p_init_min_idx = sum(Ve_vec < 0) - 1
        p_a = p_vec[p_init_min_idx]
        p_b = p_vec[p_init_min_idx + 1]
        # finding the root for Ve = 0 by bisection
        self.solve_for_p_star(p_a = p_a, p_b = p_b, display = True)
    
    def solve_question_1c(self,
                          m_init = 0.25):
        distribution = self.solve_for_stationary_dist(m = m_init)
        # plot the distribution
        aux_x_values = np.arange(self.Nz)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.bar(aux_x_values, distribution.flatten(),
               color = '#7BA23F')
        ax.set_xticks(aux_x_values)
        ax.set_xticklabels(self.z_grid)
        ax.set_xlabel('$z$')
        ax.set_ylabel('$\Lambda(z, p^{*})$')
        plt.savefig('figQ1c.png', dpi = 100, bbox_inches='tight', pad_inches=0)
    
    def solve_question_1d_1e(self,
                             m_init = 0.25):
        self.solve_for_m(m_init = m_init)
        print(f'm_star = {self.m_star: .4f}')
        self.calculate_stationary_stats(write_out = True)
    
class MoscosoMukoyama2012(Hopenhayn1992):
    def __init__(self,
                 z_grid = np.array([2.32, 3.23, 4.24, 5.89, 7.75]),
                 trans_mat = np.array([
                     [0.80, 0.15, 0.05, 0.00, 0.00],
                     [0.15, 0.65, 0.15, 0.05, 0.00],
                     [0.05, 0.15, 0.60, 0.15, 0.05],
                     [0.00, 0.05, 0.15, 0.65, 0.15],
                     [0.00, 0.00, 0.05, 0.15, 0.80]
                     ]),
                 G = np.array([0.720, 0.140, 0.075, 0.045, 0.020]),
                 alpha = 0.64,
                 beta = 0.94,
                 cf = 20.,
                 ce = 36.2,
                 kappa = 0.5,
                 b = 1.,
                 L = 1.,
                 ):
        super().__init__(z_grid = z_grid,
                         trans_mat = trans_mat,
                         G = G,
                         alpha = alpha,
                         beta = beta,
                         cf = cf,
                         ce = ce,
                         kappa = kappa,
                         b = b)
        self.L = L
    
    def excess_labor_demand(self,
                            w_guess,
                            p_a  = 0.25,
                            p_b  = 4.00,
                            m_init = 0.25,
                            xtol_bisec = 2e-12,
                            rtol_bisec = 8.881784197001252e-16,
                            max_iter_bisec = 100,
                            tol  = def_tol,
                            max_iter = def_max_iter
                            ):
        self.solve_for_p_star(w = w_guess,
                              p_a  = p_a,
                              p_b  = p_b,
                              xtol_bisec = xtol_bisec,
                              rtol_bisec = rtol_bisec,
                              max_iter_bisec = max_iter_bisec,
                              tol  = tol,
                              max_iter = max_iter
                              )
        self.solve_for_m(m_init = m_init,
                         max_iter = max_iter,
                         tol = tol)
        self.calculate_stationary_stats(w = w_guess)
        return self.N_star - self.L
    
    def solve_for_w(self,
                    w_a = 0.5,
                    w_b = 1.2,
                    p_a  = 0.25,
                    p_b  = 4.00,
                    m_init = 0.25,
                    xtol_bisec = 2e-12,
                    rtol_bisec = 8.881784197001252e-16,
                    max_iter_bisec = 100,
                    tol  = def_tol,
                    max_iter = def_max_iter,
                    disp = False
                    ):
        args = (p_a, p_b, m_init, xtol_bisec, rtol_bisec,
                max_iter_bisec, tol, max_iter)
        if disp:
            stopwatch = StopWatch()
            print('Starting bisection...')
        w_star, stats = bisect(self.excess_labor_demand,
                               a = w_a,
                               b = w_b,
                               args = args,
                               xtol = xtol_bisec,
                               rtol = rtol_bisec,
                               maxiter = max_iter_bisec,
                               full_output = True
                               )
        if disp:
            stopwatch.stop()
            print(f'Bisection took {stats.iterations} iterations.')
            print(f'w_star = {w_star:.4f}')
        # calculate the stationary equilibrium
        diff = self.excess_labor_demand(w_star)
        self.w_star = w_star

def solve_question_2a(trans_mat, ce_values):
    result_mat = np.zeros((len(ce_values), 6))
    if type(ce_values) is list:
        ce_values = np.array(ce_values)
    # iteratively caluculare the statistics
    stopwatch = StopWatch()
    for i, ce_i in enumerate(ce_values):
        print(f'Solving the Hopenhayn model with ce = {ce_i:.2f}...')
        model_i = Hopenhayn1992(trans_mat = trans_mat,
                                ce = ce_i)
        model_i()
        result_mat[i, :] = np.array([model_i.p_star,
                                     model_i.entry_rate,
                                     model_i.exit_rate,
                                     model_i.N_star,
                                     model_i.ave_size,
                                     model_i.A_star])
    stopwatch.stop()
    # Graphics
    result_mat = result_mat.T
    titles = ('$p^{*}$', 'Entry rate', 'Exit rate',
              'Total employment', 'Average firm size',
              'Total output per worker')
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    for row in range(3):
        for col in range(2):
            idx = 2*row + col
            ax[row, col].plot(ce_values, result_mat[idx, :],
                              c = '#7BA23F', marker = 'o')
            ax[row, col].set_title(titles[idx])
            if idx >= 4:
                ax[row, col].set_xlabel('$c_e$')
    plt.savefig('figQ2a.png', dpi = 100, bbox_inches='tight', pad_inches=0)

def solve_question_2c(trans_mat, ce_values, N_star):
    result_mat = np.zeros((len(ce_values), 7))
    if type(ce_values) is list:
        ce_values = np.array(ce_values)
    # iteratively caluculare the statistics
    stopwatch = StopWatch()
    for i, ce_i in enumerate(ce_values):
        print(f'Solving the Moscoso and Mukoyama model with ce = {ce_i:.2f}...')
        model_i = MoscosoMukoyama2012(trans_mat = trans_mat,
                                      ce = ce_i,
                                      L = N_star)
        model_i.solve_for_w()
        result_mat[i, :] = np.array([model_i.p_star,
                                     model_i.entry_rate,
                                     model_i.exit_rate,
                                     model_i.N_star,
                                     model_i.ave_size,
                                     model_i.A_star,
                                     model_i.w_star,])
    stopwatch.stop()
    # Graphics
    result_mat = result_mat.T
    titles = ('$p^{*}$', 'Entry rate', 'Exit rate',
              'Total employment','Average firm size',
              'Total output per worker', '$w^{*}$')
    fig, ax = plt.subplots(4, 2, figsize=(10, 16))
    for row in range(4):
        for col in range(2):
            idx = 2*row + col
            if idx == 7:
                ax[row, col].axis('off')
            else:
                ax[row, col].plot(ce_values, result_mat[idx, :],
                                  c = '#7BA23F', marker = 'o')
                ax[row, col].set_title(titles[idx])
                if idx >= 5:
                    ax[row, col].set_xlabel('$c_e$')
    plt.savefig('figQ2c.png', dpi = 100, bbox_inches='tight', pad_inches=0)
                              