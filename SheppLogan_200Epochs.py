import numpy as np
import pickle

from cil.optimisation.functions import LeastSquares, L2NormSquared
from cil.optimisation.algorithms import FISTA
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
num_epochs = 50
smooth_f = LeastSquares(A, b=noisy_data, c=0.5)
alpha = 7
proxfriendly_g = (alpha / ig.voxel_size_x) * FGP_TV(
    max_iteration=100, device="gpu", nonnegativity=True
)
metrics = {"objective": lambda x: smooth_f(x) + proxfriendly_g(x)}

subsets = [10, 60, 240]
num_epochs = 200
eta = 0.01  # step size parameter for SGD

# Storing the results
sgd_dict = {}
saga_dict = {}
pgd_dict = {}

# PGD
Lipsch = A.norm() ** 2
pgd = ProxSGD(
    smooth_fs=[smooth_f],
    proxfriendly_g=proxfriendly_g,
    stepsizes=lambda iter: 1.0 / Lipsch,
)
pgd.run(
    initial=ig.allocate(0),
    verbose=False,
    num_epochs=num_epochs,
    metrics=None,
    store_iterate_interval=1,
)
pgd_dict = {"stored_x": pgd.stored_iterates}

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
        stepsizes=lambda iter: 1.0 / (2.0 * Lmax * (1 + eta * iter / num_subsets)),
    )
    sgd.run(
        initial=ig.allocate(0),
        verbose=False,
        num_epochs=num_epochs,
        metrics=None,
        store_iterate_interval=num_subsets,
    )
    sgd_dict[l] = {"data_passes": sgd.data_passes, "metrics": sgd.metrics}

    # SAGA
    saga = ProxSAGA(smooth_fs=smooth_fs, proxfriendly_g=proxfriendly_g, Lmax=Lmax)
    saga.run(
        initial=ig.allocate(0),
        verbose=False,
        num_epochs=num_epochs,
        metrics=None,
        store_iterate_interval=num_subsets,
    )
    saga_dict[l] = {"data_passes": saga.data_passes, "metrics": saga.metrics}

total_results_dict = {"sgd": sgd_dict, "saga": saga_dict, "pgd": pgd_dict}

import pickle

with open("SheppLogan_240Epochs_Results.pkl", "wb") as fp:
    pickle.dump(total_results_dict, fp)
    print("Results saved")
