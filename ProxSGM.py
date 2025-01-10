import numpy as np
from abc import ABC
from copy import deepcopy as copy

from utils import herman_meyer_order

ADMISSIBLE_SVRG_TYPES = ["standard", "lsq_datafit"]
ADMISSIBLE_SAGA_TYPES = ["standard", "linear_model"]


"""
Basic class for stochastic gradient-based methods
"""


class ProxSGD(ABC):
    def __init__(
        self,
        smooth_fs,
        proxfriendly_g,
        Lmax=None,
        sampling=None,
        sampling_weights=None,
        stepsizes=None,
        seed=42,
    ):
        """
        Inputs
        :param list smooth_fs: list of smooth functions
        :param cil.function proxfriendly_g: prox-friendly function
        :param np.float Lmax: maximum smoothness constant
        :param str sampling: sampling method
        :param str sampling_weights: weights for weighted sampling
        :param func stepsizes: stepsize function
        :param int seed: random seed
        """
        self.num_subsets = len(smooth_fs)
        self.Lmax = Lmax

        self.smooth_fs = smooth_fs
        self.proxfriendly_g = proxfriendly_g

        self.set_sampling(
            sampling=sampling, sampling_weights=sampling_weights, seed=seed
        )

        self.set_stepsize_func(stepsizes)

        self.data_passes = []
        self.used_indices = []
        self.stored_iterates = []

        self.metrics = None
        self.metrics_functions = None
        self._algo_name = "ProxSGD"

    def set_sampling(self, sampling=None, sampling_weights=None, seed=42):

        if sampling is None:
            sampling = "uniform"
        if sampling in ["uniform", "weighted", "adaptive", "without_replacement"]:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
        if sampling == "without_replacement":
            self.idx_list = []
        elif sampling == "hermanmeyer":
            self.perm_idx_list = herman_meyer_order(self.num_subsets)
            self.idx_list = copy(self.perm_idx_list)
        if sampling in ["weighted", "adaptive"]:
            self.sampling_weights = sampling_weights
            if self.sampling_weights is not None:
                self.sampling_weights = np.asarray(self.sampling_weights).astype(
                    "float64"
                )
        self.sampling = sampling

    def set_stepsize_func(self, stepsizes=None):

        if (
            stepsizes is None or stepsizes == "constant"
        ):  
            self.stepsize_func = lambda x: 1.0 / 2.0 / self.Lmax
        else:
            self.stepsize_func = stepsizes

    def get_nextsubset(self, iterate=None):

        if self.sampling == "uniform":
            return self.rng.choice(self.num_subsets, replace=True)
        elif self.sampling == "without_replacement":
            if not self.idx_list:
                self.idx_list = self.rng.permutation(self.num_subsets).tolist()
            return self.idx_list.pop(0)
        elif self.sampling == "weighted":
            return self.rng.choice(
                self.num_subsets, replace=True, p=self.sampling_weights
            )
        elif self.sampling == "hermanmeyer":
            if not self.idx_list:
                self.idx_list = copy(self.perm_idx_list)
            return self.idx_list.pop(0)
        elif self.sampling == "cyclic":
            return iterate % self.num_subsets
        else:
            raise ValueError("In get_nextsubset: unidentified sampling method")

    def get_updatedirection(self, x, subset_num):
        return self.smooth_fs[subset_num].gradient(x)

    def get_stepsize(self, iteration=None):
        return self.stepsize_func(iteration)

    def objective_function(self, x):
        return sum([smooth_f(x) for smooth_f in self.smooth_fs]) + self.proxfriendly_g(
            x
        )

    def compute_metrics(self, x, iter):
        if self.metrics is None:
            raise TypeError("In compute_metrics: metrics are not defined")
        else:
            for key, func in self.metrics_functions.items():
                self.metrics[key].append(func(x))
        self.metrics["iter"].append(iter)

    def prerun_things(self, x):
        self.data_passes.append(0.0)

    def post_iterate_updates(self, subset_num, iterate):
        self.used_indices.append(subset_num)
        self.data_passes.append(self.data_passes[-1] + 1.0 / self.num_subsets)

    def run(
        self,
        initial,
        num_epochs=10,
        verbose=False,
        metrics=None,
        metrics_interval=1,
        store_iterate_interval=np.inf,
    ):
        """
        :param cil_data initial: initial iterate
        :param int num_epochs: number of epochs
        :param bool verbose: print metrics
        :param list metrics: metrics to compute
        :param int metrics_interval: interval to compute metrics
        :param np.int store_iterate_interval: interval to store iterates
        """

        print("Running {} with {} sampling".format(self._algo_name, self.sampling))
        max_iter = 50000
        self.x = initial

        if np.isfinite(store_iterate_interval):
            self.stored_iterates.append(self.x.array)

        self.prerun_things(initial)

        if metrics is not None:
            compute_metrics = True
            self.metrics = {key: [] for key in metrics.keys()}
            self.metrics["iter"] = []
            self.metrics_functions = metrics
            self.compute_metrics(self.x, 0)
        else:
            compute_metrics = False

        for iter in range(1, max_iter + 1):
            # Get the subset, stepsize, and the update direction
            self.x_old = copy(self.x)
            subset_num = self.get_nextsubset()
            step_size = self.get_stepsize(iteration=(iter - 1))

            update_direction = self.get_updatedirection(self.x, subset_num, iter)

            # Apply update
            self.x = self.proxfriendly_g.proximal(
                self.x - step_size * update_direction, step_size / self.num_subsets
            )

            # Metrics and other tracking
            self.post_iterate_updates(subset_num, iter)
            if compute_metrics:
                if iter % metrics_interval == 0:
                    self.compute_metrics(self.x, iter)  #
                    if verbose is True:
                        print(
                            " ".join(
                                f"{key}: {value[-1]:.4f}\t\t"
                                for key, value in self.metrics.items()
                            )
                        )

            if np.isfinite(store_iterate_interval) and (
                iter % store_iterate_interval == 0
            ):
                self.stored_iterates.append(self.x.array)

            if (num_epochs - self.data_passes[-1]) < 1e-4:
                break


"""
Inherited class, defining prox-SAGA
"""

class ProxSAGA(ProxSGD):
    def __init__(
        self,
        smooth_fs,
        proxfriendly_g,
        Lmax,
        sampling=None,
        stepsizes=None,
        sampling_weights=None,
        saga_type="standard",
        operators=None,
        seed=42,
    ):  
        """
        Inputs
        :param list smooth_fs: list of smooth functions
        :param cil.function proxfriendly_g: prox-friendly function
        :param np.float Lmax: maximum smoothness constant
        :param str sampling: sampling method
        :param str sampling_weights: weights for weighted sampling
        :param func stepsizes: stepsize function
        :param str saga_type: type of SAGA. Options: 'standard', 'linear_model'
        :param list operators: list of operators, required for linear model SAGA
        :param int seed: random seed
        """
        super(ProxSAGA, self).__init__(
            smooth_fs,
            proxfriendly_g,
            Lmax,
            sampling=sampling,
            stepsizes=stepsizes,
            sampling_weights=sampling_weights,
            seed=seed,
        )
        self.gradient_table = []
        self.full_gradient = None

        self.saga_type = saga_type
        if saga_type == "linear_model" and operators is None:
            raise TypeError("Unidentified saga type")
        else:
            self.operators = operators

        self._algo_name = self.saga_type + " ProxSAGA"

    def set_stepsize_func(self, stepsizes=None):
        if stepsizes is None or stepsizes == "constant":
            self.stepsize_func = lambda x: 1.0 / 3.0 / self.Lmax
        else:
            self.stepsize_func = stepsizes

    def prerun_things(self, x):
        if self.saga_type == "standard":
            self.gradient_table = [smooth_f.gradient(x) for smooth_f in self.smooth_fs]
            self.full_gradient = sum(self.gradient_table)
        elif self.saga_type == "linear_model":
            self.gradient_table = [
                smooth_f.gradient(operator.direct(x))
                for smooth_f, operator in zip(self.smooth_fs, self.operators)
            ]
            self.full_gradient = sum(
                [
                    operator.adjoint(stored_gradient)
                    for operator, stored_gradient in zip(
                        self.operators, self.gradient_table
                    )
                ]
            )
        self.data_passes = [1.0]

    def get_updatedirection(self, x, subset_num, iterate):
        if self.saga_type == "standard":
            tmp = self.smooth_fs[subset_num].gradient(x)
            direction = (
                tmp
                - self.gradient_table[subset_num]
                + (1.0 / self.num_subsets) * self.full_gradient
            )
            self.full_gradient += tmp - self.gradient_table[subset_num]
            self.gradient_table[subset_num] = tmp
        elif self.saga_type == "linear_model":
            tmp = self.smooth_fs[subset_num].gradient(
                self.operators[subset_num].direct(x)
            )
            grad_change = self.operators[subset_num].adjoint(
                tmp - self.gradient_table[subset_num]
            )
            direction = grad_change + (1.0 / self.num_subsets) * self.full_gradient
            self.full_gradient += grad_change
            self.gradient_table[subset_num] = tmp
        else:
            raise ValueError("Inadmissible saga type")
        return direction

    def post_iterate_updates(self, subset_num, iterate):
        self.used_indices.append(subset_num)
        if iterate == 0:
            self.data_passes.append(1.0)
        else:
            self.data_passes.append(self.data_passes[-1] + 1.0 / self.num_subsets)

"""
Inherited class, defining prox-SVRG
"""

class ProxSVRG(ProxSGD):
    def __init__(
        self,
        smooth_fs,
        proxfriendly_g,
        Lmax,
        sampling=None,
        sampling_weights=None,
        stepsizes=None,
        svrg_type="standard",
        operators=None,
        update_frequency=2,
        seed=42,
    ):  
        """
        Inputs
        :param list smooth_fs: list of smooth functions
        :param cil.function proxfriendly_g: prox-friendly function
        :param np.float Lmax: maximum smoothness constant
        :param str sampling: sampling method
        :param str sampling_weights: weights for weighted sampling
        :param func stepsizes: stepsize function
        :param str svrg_type: type of SVRG. Options: 'standard', 'lsq_datafit'
        :param list operators: list of operators, required for linear model SAGA
        :param int update_frequency: frequency of full gradient update
        :param int seed: random seed
        """

        super(ProxSVRG, self).__init__(
            smooth_fs,
            proxfriendly_g,
            Lmax,
            sampling=sampling,
            stepsizes=stepsizes,
            sampling_weights=sampling_weights,
            seed=seed,
        )

        self.svrg_type = svrg_type
        if svrg_type == "lsg_datafit" and operators is None:
            raise ValueError("For lsq_datafit svrg, operators must be provided")
        else:
            self.operators = operators

        self.full_gradient = None
        self.reference_iterate = None

        self.update_frequency = update_frequency
        self.full_update_flag = False

        self._algo_name = self.svrg_type + " ProxSVRG"

    def set_stepsize_func(self, stepsizes=None):
        if stepsizes is None or stepsizes == "constant":
            self.stepsize_func = lambda x: 1.0 / 3.0 / self.Lmax
        else:
            self.stepsize_func = stepsizes

    def prerun_things(self, x):
        self.data_passes = [0]
        self.reference_iterate = x.copy()
        if self.sampling == "adaptive":
            tmp = [smooth_f.gradient(x) for smooth_f in self.smooth_fs]
            self.full_gradient = sum(tmp)
            self.sampling_weights = np.array(
                [subgrad.norm() for subgrad in tmp]
            ).astype("float64")
            self.sampling_weights /= np.sum(self.sampling_weights)
        else:
            self.full_gradient = sum(
                [smooth_f.gradient(x) for smooth_f in self.smooth_fs]
            )
        self.full_update_flag = True

    def get_updatedirection(self, x, subset_num, iterate):
        if iterate % (self.update_frequency * self.num_subsets) == 0:
            self.reference_iterate = x.copy()
            if self.sampling == "adaptive":
                tmp = [smooth_f.gradient(x) for smooth_f in self.smooth_fs]
                self.full_gradient = sum(tmp)
                self.sampling_weights = np.array(
                    [subgrad.norm() for subgrad in tmp]
                ).astype("float64")
                self.sampling_weights /= np.sum(self.sampling_weights)
            else:
                self.full_gradient = sum(
                    [smooth_f.gradient(x) for smooth_f in self.smooth_fs]
                )

            self.full_update_flag = True
            direction = (1.0 / self.num_subsets) * self.full_gradient
        else:
            if self.svrg_type == "standard":
                direction = (
                    self.smooth_fs[subset_num].gradient(x)
                    - self.smooth_fs[subset_num].gradient(self.reference_iterate)
                    + (1.0 / self.num_subsets) * self.full_gradient
                )
            elif self.svrg_type == "lsq_datafit":
                tmp = self.operators[subset_num].direct(x - self.reference_iterate)
                direction = (
                    self.operators[subset_num].adjoint(tmp)
                    + (1.0 / self.num_subsets) * self.full_gradient
                )
            else:
                raise ValueError("Inadmissible svrg type")
        return direction

    def post_iterate_updates(self, subset_num, iterate):
        self.used_indices.append(subset_num)
        if self.full_update_flag is True:
            self.data_passes.append(self.data_passes[-1] + 1)
            self.full_update_flag = False
        else:
            self.data_passes.append(self.data_passes[-1] + 1.0 / self.num_subsets)

"""
A proximal version of ADAM
"""


class ProxADAM(ABC):
    def __init__(
        self,
        smooth_fs,
        proxfriendly_g,
        Lmax=None,
        sampling=None,
        sampling_weights=None,
        stepsizes=None,
        seed=42,
    ):  
        """
        Inputs
        :param list smooth_fs: list of smooth functions
        :param cil.function proxfriendly_g: prox-friendly function
        :param np.float Lmax: maximum smoothness constant
        :param str sampling: sampling method
        :param str sampling_weights: weights for weighted sampling
        :param func stepsizes: stepsize function
        :param int seed: random seed
        """
        self.num_subsets = len(smooth_fs)
        self.Lmax = Lmax

        self.smooth_fs = smooth_fs
        self.proxfriendly_g = proxfriendly_g

        self.set_sampling(
            sampling=sampling, sampling_weights=sampling_weights, seed=seed
        )

        self.set_stepsize_func(stepsizes)

        self.data_passes = []
        self.used_indices = []
        self.stored_iterates = []

        # ADAM parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsi = 1e-8


        self.metrics = None
        self.metrics_functions = None
        self._algo_name = "ProxADAM"

    def set_sampling(self, sampling=None, sampling_weights=None, seed=42):
        if sampling is None:
            sampling = "uniform"
        if sampling in ["uniform", "weighted", "adaptive", "without_replacement"]:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
        if sampling == "without_replacement":
            self.idx_list = []
        elif sampling == "hermanmeyer":
            self.perm_idx_list = herman_meyer_order(self.num_subsets)
            self.idx_list = copy(self.perm_idx_list)
        if sampling in ["weighted", "adaptive"]:
            self.sampling_weights = sampling_weights
            if self.sampling_weights is not None:
                self.sampling_weights = np.asarray(self.sampling_weights).astype(
                    "float64"
                )
        self.sampling = sampling

    def set_stepsize_func(self, stepsizes=None):
        if stepsizes is None or stepsizes == "constant":
            self.stepsize_func = lambda x: 1.0 / 2.0 / self.Lmax
        else:
            self.stepsize_func = stepsizes

    def get_nextsubset(self, iterate=None):
        if self.sampling == "uniform":
            return self.rng.choice(self.num_subsets, replace=True)
        elif self.sampling == "without_replacement":
            if not self.idx_list:
                self.idx_list = self.rng.permutation(self.num_subsets).tolist()
            return self.idx_list.pop(0)
        if self.sampling in ["weighted", "adaptive"]:
            return self.rng.choice(
                self.num_subsets, replace=True, p=self.sampling_weights
            )
        elif self.sampling == "hermanmeyer":
            if not self.idx_list:
                self.idx_list = copy(self.perm_idx_list)
            return self.idx_list.pop(0)
        elif self.sampling == "cyclic":
            return iterate % self.num_subsets
        else:
            pass

    def get_updatedirection(self, x, subset_num, iteration):
        return self.smooth_fs[subset_num].gradient(x)

    def get_stepsize(self, iteration=None):
        return self.stepsize_func(iteration)

    def objective_function(self, x):
        return sum([smooth_f(x) for smooth_f in self.smooth_fs]) + self.proxfriendly_g(
            x
        )

    def compute_metrics(self, x, iter):
        if self.metrics is None:
            raise TypeError("Your metrics dont exist")
        else:
            for key, func in self.metrics_functions.items():
                self.metrics[key].append(func(x))
        self.metrics["iter"].append(iter)

    def prerun_things(self, x):
        self.data_passes.append(0.0)

    def post_iterate_updates(self, subset_num, iterate):
        self.used_indices.append(subset_num)
        self.data_passes.append(self.data_passes[-1] + 1.0 / self.num_subsets)

    def run(
        self,
        initial,
        num_epochs=10,
        verbose=False,
        metrics=None,
        metrics_interval=1,
        store_iterate_interval=np.inf,
        measure_time=False,
    ):
        """
        :param cil_data initial: initial iterate
        :param int num_epochs: number of epochs
        :param bool verbose: print metrics
        :param list metrics: metrics to compute
        :param int metrics_interval: interval to compute metrics
        :param np.int store_iterate_interval: interval to store iterates
        """

        print("Running {} with {} sampling".format(self._algo_name, self.sampling))
        max_iter = 50000
        self.x = initial

        self.first_mom = 0.0 * initial
        self.second_mom = 0.0 * initial
        self.first_mom_hat = 0.0 * initial
        self.second_mom_hat = 0.0 * initial

        if np.isfinite(store_iterate_interval):
            self.stored_iterates.append(self.x.array)

        if measure_time is True:
            import time

            start_time = time.time()
        self.prerun_things(initial)

        if measure_time is True:
            self.algo_time = [time.time() - start_time]
        else:
            self.algo_time = []

        if metrics is not None:
            compute_metrics = True
            self.metrics = {key: [] for key in metrics.keys()}
            self.metrics["iter"] = []
            self.metrics_functions = metrics
            self.compute_metrics(self.x, 0)
        else:
            compute_metrics = False

        for iter in range(1, max_iter + 1):
            # Get subset, stepsize, and direction
            self.x_old = copy(self.x)
            subset_num = self.get_nextsubset()
            step_size = self.get_stepsize(iteration=(iter - 1))

            if measure_time is True:
                start_time = time.time()
            update_direction = self.get_updatedirection(self.x, subset_num, iter)

            self.first_mom = (
                self.beta1 * self.first_mom + (1 - self.beta1) * update_direction
            )
            self.second_mom = (
                self.beta2 * self.second_mom + (1 - self.beta2) * update_direction ** 2
            )
            self.first_mom_hat = self.first_mom / (1 - self.beta1 ** iter)
            self.second_mom_hat = self.second_mom / (1 - self.beta2 ** iter)
            precond = self.first_mom_hat / (self.second_mom_hat ** 0.5 + self.epsi)

            # Apply update
            self.x = self.proxfriendly_g.proximal(
                self.x - step_size * precond, step_size / self.num_subsets
            )

            # Tracking
            self.post_iterate_updates(subset_num, iter)
            if measure_time is True:
                self.algo_time.append(self.algo_time[-1] + time.time() - start_time)
            if compute_metrics:
                if iter % metrics_interval == 0:
                    self.compute_metrics(self.x, iter)
                    if verbose is True:
                        print(
                            " ".join(
                                f"{key}: {value[-1]:.4f}\t\t"
                                for key, value in self.metrics.items()
                            )
                        )

            if np.isfinite(store_iterate_interval) and (
                iter % store_iterate_interval == 0
            ):
                self.stored_iterates.append(self.x.array)

            if (num_epochs - self.data_passes[-1]) < 1e-4:
                break
