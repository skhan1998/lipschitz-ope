from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp
from sklearn.metrics import pairwise_distances
import logging
from tqdm.notebook import tqdm
from plotnine import *
import pandas as pd
import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()


class EstimatorBase(ABC):
    def __init__(self, X, Y, A, pi_b, pi_e, unsafe = False) -> None:
        self.X = X
        self.Y = Y
        self.A = A
        self.pi_b = np.array(pi_b)
        self.pi_e = np.array(pi_e)
        if X.shape[1] == 1:
            self.D = pairwise_distances(X.reshape(-1, 1))
        else:
            self.D = pairwise_distances(X)

        self.n = X.shape[0]
        # self.d = X.shape[1]
        self.no_overlap = (self.pi_b == 0).astype(int)
        self.unsafe = unsafe

    def compute_psi1_hat(self):
        # logger.info("Running IPW in overlap region")

        reweighted_data = [(1-self.no_overlap[i]) * self.Y[i] * self.A[i] * self.pi_e[i] /
                           self.pi_b[i] if self.no_overlap[i] == 0 else 0 for i in range(self.n)]
        self.psi1_hat = np.mean(reweighted_data)
        return self.psi1_hat

    @abstractmethod
    def check_feasibility(self):
        self.feasible = None
        pass

    @abstractmethod
    def mu_hat_bounds(self, i):
        pass

    def compute_psi2_bounds(self):
        # logger.info("Bounding contribution from non-overlap region")
        self.check_feasibility()

        if not self.feasible and not self.unsafe:
            self.psi2_inf = np.nan
            self.psi2_sup = np.nan

            return [self.psi2_inf, self.psi2_sup]

        all_bounds = np.array(
            [self.mu_hat_bounds(i) if self.no_overlap[i] == 1 else [0, 0] for i in range(self.n)])

        self.psi2_inf = np.mean(all_bounds[:, 0]*self.no_overlap*self.pi_e)
        self.psi2_sup = np.mean(all_bounds[:, 1]*self.no_overlap*self.pi_e)

        return [self.psi2_inf, self.psi2_sup]

    def psi_hat(self):
        # self.check_feasibility()
        self.compute_psi1_hat()
        self.compute_psi2_bounds()
            
        if self.unsafe:
            return [self.psi1_hat + self.psi2_inf, self.psi1_hat + self.psi2_sup]
        
        else:
            if self.feasible:
                return [self.psi1_hat + self.psi2_inf, self.psi1_hat + self.psi2_sup]
            return [np.nan, np.nan]

class ManskiEstimator(EstimatorBase):
    def __init__(self, X, Y, A, pi_b, pi_e, M_lower, M_upper) -> None:
        super().__init__(X, Y, A, pi_b, pi_e)
        self.M_lower = M_lower
        self.M_upper = M_upper

    def check_feasibility(self):
        self.feasible = True
        return self.feasible

    def mu_hat_bounds(self, i):
        return [self.M_lower, self.M_upper]

class LipBddNoiseEstimator(EstimatorBase):
    def __init__(self, X, Y, A, pi_b, pi_e, L, eps) -> None:
        super().__init__(X, Y, A, pi_b, pi_e)
        self.L = L
        self.eps = eps

    def check_feasibility(self):
        # logger.info(f"Checking feasibility with L={self.L}")
        # this needs to be fixed to handle the case wehre A_i = 1 but A_j = 0
        response_diffs = np.abs(
            np.subtract.outer(self.A*self.Y, self.A*self.Y))
        lipschitz_bounds = self.L*self.D + 2*self.eps

        self.feasible = np.all(response_diffs <= lipschitz_bounds)

        return self.feasible

    def mu_hat_bounds(self, i):

        distances = self.D[i, :]
        distances[i] = np.inf
        distances[self.A == 0] = np.inf

        j_star = np.argmin(distances)

        return [self.Y[j_star] - self.L*self.D[i, j_star] - self.eps, self.Y[j_star] + self.L*self.D[i, j_star] + self.eps]


class LipImputeEstimator(EstimatorBase):
    def __init__(self, X, Y, A, pi_b, pi_e, L, model, unsafe = False) -> None:
        super().__init__(X, Y, A, pi_b, pi_e, unsafe)
        self.L = L
        self.model = model

        # Fit model where A==1 and make predictions
        obs = self.A == 1
        X_obs = X[obs, :]
        Y_obs = Y[obs]

        self.model.fit(X_obs, Y_obs)
        if hasattr(model, "predict_proba"):
            Y_hat = model.predict_proba(self.X)[:, 1]
        else:
            Y_hat = model.predict(self.X)

        self.Y_hat = np.round(Y_hat, 10)


        self.imputation_value = np.mean(self.Y_hat*self.pi_e)

    def check_feasibility(self):
        # logger.info(f"Checking feasibility with L={self.L}")
        response_diffs = np.abs(
            np.subtract.outer(self.Y_hat, self.Y_hat))
        lipschitz_bounds = self.L*self.D

        self.feasible = np.all(response_diffs <= lipschitz_bounds)

        return self.feasible

    def mu_hat_bounds(self, i):
        lower_bounds = self.Y_hat - self.L*self.D[i,:]
        upper_bounds = self.Y_hat + self.L*self.D[i,:]
        
        lower_bounds[self.no_overlap == 1] = -np.inf
        upper_bounds[self.no_overlap == 1] = np.inf 

        return [max(lower_bounds), min(upper_bounds)]

class LipImputeBddRespEstimator(LipImputeEstimator):
    def __init__(self, X, Y, A, pi_b, pi_e, L, model, M_lower, M_upper) -> None:
        super().__init__(X, Y, A, pi_b, pi_e, L, model)
        self.M_lower = M_lower
        self.M_upper = M_upper

    def mu_hat_bounds(self, i):
        lower_bounds = self.Y_hat - self.L*self.D[i,:]
        lower_bounds[self.no_overlap == 1] = -np.inf

        lip_lower_bound = max(lower_bounds)

        upper_bounds = self.Y_hat + self.L*self.D[i,:]
        upper_bounds[self.no_overlap == 1] = np.inf 
        lip_upper_bound = min(upper_bounds)

        return [max(lip_lower_bound, self.M_lower), min(lip_upper_bound, self.M_upper)]

class LipImputeApproxEstimator(LipImputeEstimator):
    def __init__(self, X, Y, A, pi_b, pi_e, L, model) -> None:
        super().__init__(X, Y, A, pi_b, pi_e, L, model)

    def mu_hat_bounds(self, i):
        #Compute bounds using nearest neighbors rather than finding the sharpest bound
        distances = self.D[i, :]
        distances[i] = np.inf
        distances[self.no_overlap == 1] = np.inf

        j_star = np.argmin(distances)

        lower_bound_lip = self.Y_hat[j_star] - self.L*self.D[i, j_star]
        upper_bound_lip = self.Y_hat[j_star] + self.L*self.D[i, j_star]

        return [max(lower_bound_lip, self.M_lower), min(upper_bound_lip, self.M_upper)]

class MultiActionOPEEstimator:
    def __init__(self, X, Y, A, pi_b, pi_e, L, actions, binary_estimator, **kwargs) -> None:
        # pi_b and pi_e are dictionaries indexed by actions
        self.X = X
        self.Y = Y
        self.A = A
        self.pi_b = pi_b
        self.pi_e = pi_e
        self.L = L
        self.actions = actions
        self.kwargs = kwargs
        self.binary_estimator = binary_estimator
        self.ope_values = {} 
        self.imputation_values = {} 
        self.manski_upper = {} 
        self.manski_lower = {} 

    def estimate_action(self, action):
        estimator_class = getattr(sys.modules[__name__], self.binary_estimator)

        binary_actions = (self.A == int(action)).astype(int)

        action_estimator = estimator_class(self.X, self.Y, binary_actions,
                                       self.pi_b[action], self.pi_e[action], self.L, 
                                       **self.kwargs)

        self.ope_values[action] = action_estimator.psi_hat()

        self.imputation_values[action] = action_estimator.imputation_value
        
        if "M_lower" in self.kwargs.keys() and "M_upper" in self.kwargs.keys():
            manski_estimator = ManskiEstimator(self.X, self.Y, binary_actions, 
            self.pi_b[action], self.pi_e[action], self.kwargs["M_lower"], self.kwargs["M_upper"])

            self.manski_lower[action], self.manski_upper[action] = manski_estimator.psi_hat()

        return self.ope_values[action]

    def psi_hat(self):
        action_values = np.zeros((len(self.actions), 2))
        for idx, action in enumerate(self.actions):
            # logger.info(f"Running A={action}")
            action_values[idx,:] = self.estimate_action(action)

        return np.sum(action_values, axis = 0)            

class SensitivityAnalysis:
    def __init__(self, estimator, L_grid) -> None:
        self.estimator = estimator
        self.L_grid = L_grid

    def compute_pid_intervals(self):
        results = []

        for l in tqdm(self.L_grid, total=len(self.L_grid)):
            logger.info(f"Running L={l}")
            setattr(self.estimator, "L", l)

            psi_minus, psi_plus = self.estimator.psi_hat()
            results.append([l, psi_minus, psi_plus, psi_plus - psi_minus])

        result_df = pd.DataFrame(results)
        result_df.columns = ["L", "psi_minus", "psi_plus", "width"]
        self.result_df = result_df

        return result_df.head()

    def plot_results(self):
        p = (ggplot(self.result_df, aes(x="L"))
             + geom_line(aes(y="psi_plus"))
             + geom_line(aes(y="psi_minus"))
             + labs(x="Lipschitz constant, L", y="Off-policy value")
             + theme_bw())

        return p
