import numpy as np
import sys
import pickle

# sys.path.append("/media/chen/Zeljko/stochastic/StochasticCIL/")

from cil.optimisation.functions import LeastSquares
from cil.plugins.ccpi_regularisation.functions import FGP_TV

from cil.framework import AcquisitionGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.processors import Slicer

from ProxSGM import ProxSAGA, ProxSGD

# Set up the forward operator and the data
num_angles = 240
num_detectors = 128
device = "gpu"
angles = np.linspace(0, 180, num_angles, dtype="float32")
ag = AcquisitionGeometry.create_Parallel2D().set_angles(angles).set_panel(num_detectors)
ig = ag.get_ImageGeometry()
A = ProjectionOperator(ig, ag, device=device)

# Load the reference solution and objective value for performance metrics
with open("SL_Ref_nparray.pkl", "rb") as f:
    ref_dict = pickle.load(f)
ref_soln = ref_dict["ref_soln"]
ref_value = ref_dict["value"]

# Ground truth
phantom_np = ref_dict["phantom"]
phantom = ig.allocate(0)
phantom.fill(phantom_np)

# Sinogram
sinogram_np = ref_dict["sinogram"]
noisy_data = ag.allocate(0)
noisy_data.fill(sinogram_np)

# Clean data
clean_data = A.direct(phantom)

# Objective and optimisation settings
smooth_f = LeastSquares(A, b=noisy_data, c=0.5)
alpha = 7
proxfriendly_g = (alpha / ig.voxel_size_x) * FGP_TV(
    max_iteration=100, device="gpu", nonnegativity=True
)
metrics = {"objective": lambda x: smooth_f(x) + proxfriendly_g(x)}

subsets = [30, 240]
num_epochs = 50
eta = 0.01  # step size parameter

# Storing the results
saga_with_replacement_dict = {}
saga_without_replacement_dict = {}
sgd_with_replacement_dict = {}
sgd_without_replacement_dict = {}
igd_dict = {}
iaga_dict = {}

for l, num_subsets in enumerate(subsets):
    if num_subsets > num_angles:
        num_subsets = num_angles
    print("\t working on {} subsets".format(num_subsets))

    # Partition the data into subsets and create the functions
    datas = [
        Slicer(roi={"angle": (i, num_angles, num_subsets)})(noisy_data)
        for i in range(num_subsets)
    ]
    Ais = [
        ProjectionOperator(ig, data_batch.geometry, device=device)
        for data_batch in datas
    ]
    smooth_fs = [LeastSquares(Ai, b=datai, c=0.5) for Ai, datai in zip(Ais, datas)]

    metrics = {
        "objective": lambda x: sum([smoothf(x) for smoothf in smooth_fs])
        + proxfriendly_g(x)
    }
    if ref_soln is not None:
        metrics["2_error"] = lambda x: (x - ref_soln).norm()
        metrics["relative obj"] = (
            lambda x: sum([smoothf(x) for smoothf in smooth_fs])
            + proxfriendly_g(x)
            - ref_value
        )

    Lmax = max([Ai.norm() ** 2 for Ai in Ais])

    # SGD
    sgd = ProxSGD(
        smooth_fs=smooth_fs,
        proxfriendly_g=proxfriendly_g,
        sampling="uniform",
        stepsizes=lambda iter: 1.0 / (2.0 * Lmax * (1 + eta * iter / num_subsets)),
    )
    sgd.run(
        initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics
    )
    sgd_with_replacement_dict[l] = {
        "data_passes": sgd.data_passes,
        "metrics": sgd.metrics,
    }

    sgd = ProxSGD(
        smooth_fs=smooth_fs,
        proxfriendly_g=proxfriendly_g,
        sampling="without_replacement",
        stepsizes=lambda iter: 1.0 / (2.0 * Lmax * (1 + eta * iter / num_subsets)),
    )
    sgd.run(
        initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics
    )
    sgd_without_replacement_dict[l] = {
        "data_passes": sgd.data_passes,
        "metrics": sgd.metrics,
    }

    # IGD
    sgd = ProxSGD(
        smooth_fs=smooth_fs,
        proxfriendly_g=proxfriendly_g,
        sampling="hermanmeyer",
        stepsizes=lambda iter: 1.0 / (2.0 * Lmax * (1 + eta * iter / num_subsets)),
    )
    sgd.run(
        initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics
    )
    igd_dict[l] = {"data_passes": sgd.data_passes, "metrics": sgd.metrics}

    # SAGA
    saga = ProxSAGA(
        smooth_fs=smooth_fs,
        proxfriendly_g=proxfriendly_g,
        sampling="uniform",
        Lmax=Lmax,
    )
    saga.run(
        initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics
    )
    saga_with_replacement_dict[l] = {
        "data_passes": saga.data_passes,
        "metrics": saga.metrics,
    }

    saga = ProxSAGA(
        smooth_fs=smooth_fs,
        proxfriendly_g=proxfriendly_g,
        sampling="without_replacement",
        Lmax=Lmax,
    )
    saga.run(
        initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics
    )
    saga_without_replacement_dict[l] = {
        "data_passes": saga.data_passes,
        "metrics": saga.metrics,
    }

    # IAGA
    saga = ProxSAGA(
        smooth_fs=smooth_fs,
        proxfriendly_g=proxfriendly_g,
        sampling="hermanmeyer",
        Lmax=Lmax,
    )
    saga.run(
        initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics
    )
    iaga_dict[l] = {"data_passes": saga.data_passes, "metrics": saga.metrics}

total_results_dict = {
    "sgd with replacement": sgd_with_replacement_dict,
    "sgd without replacement": sgd_without_replacement_dict,
    "saga with replacement": saga_with_replacement_dict,
    "saga without replacement": saga_without_replacement_dict,
    "igd": igd_dict,
    "iaga": iaga_dict,
}
with open("SheppLogan_Sampling_Results.pkl", "wb") as fp:
    pickle.dump(total_results_dict, fp)
    print("Results saved")
