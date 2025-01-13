import numpy as np
import sys

# sys.path.append("/media/chen/Zeljko/stochastic/StochasticCIL/")

from cil.optimisation.functions import LeastSquares
from cil.plugins.ccpi_regularisation.functions import FGP_TV

from cil.framework import AcquisitionGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.processors import Slicer

from ProxSGM import ProxADAM
import pickle


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
num_epochs = 50

# Storing the results
adam_small_stepsize_dict = {}
adam_medium_stepsize_dict = {}
adam_large_stepsize_dict = {}

# ADAM with 10 subsets
num_subsets = 10
print("\t working on {} subsets".format(num_subsets))

# Partition the data into subsets and create the functions
datas = [
    Slicer(roi={"angle": (i, num_angles, num_subsets)})(noisy_data)
    for i in range(num_subsets)
]
Ais = [
    ProjectionOperator(ig, data_batch.geometry, device=device) for data_batch in datas
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

adam = ProxADAM(
    smooth_fs=smooth_fs,
    proxfriendly_g=proxfriendly_g,
    sampling="uniform",
    stepsizes=lambda iter: 0.01,
)
adam.run(initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics)
adam_small_stepsize_dict[0] = {"data_passes": adam.data_passes, "metrics": adam.metrics}

adam = ProxADAM(
    smooth_fs=smooth_fs,
    proxfriendly_g=proxfriendly_g,
    sampling="uniform",
    stepsizes=lambda iter: 0.1,
)
adam.run(initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics)
adam_medium_stepsize_dict[0] = {
    "data_passes": adam.data_passes,
    "metrics": adam.metrics,
}

adam = ProxADAM(
    smooth_fs=smooth_fs,
    proxfriendly_g=proxfriendly_g,
    sampling="uniform",
    stepsizes=lambda iter: 0.5,
)
adam.run(initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics)
adam_large_stepsize_dict[0] = {"data_passes": adam.data_passes, "metrics": adam.metrics}


# ADAM with 60 subsets
num_subsets = 60
print("\t working on {} subsets".format(num_subsets))

# Partition the data into subsets and create the functions
datas = [
    Slicer(roi={"angle": (i, num_angles, num_subsets)})(noisy_data)
    for i in range(num_subsets)
]
Ais = [
    ProjectionOperator(ig, data_batch.geometry, device=device) for data_batch in datas
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

adam = ProxADAM(
    smooth_fs=smooth_fs,
    proxfriendly_g=proxfriendly_g,
    sampling="uniform",
    stepsizes=lambda iter: 0.001,
)
adam.run(initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics)
adam_small_stepsize_dict[1] = {"data_passes": adam.data_passes, "metrics": adam.metrics}

adam = ProxADAM(
    smooth_fs=smooth_fs,
    proxfriendly_g=proxfriendly_g,
    sampling="uniform",
    stepsizes=lambda iter: 0.005,
)
adam.run(initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics)
adam_medium_stepsize_dict[1] = {
    "data_passes": adam.data_passes,
    "metrics": adam.metrics,
}

adam = ProxADAM(
    smooth_fs=smooth_fs,
    proxfriendly_g=proxfriendly_g,
    sampling="uniform",
    stepsizes=lambda iter: 0.05,
)
adam.run(initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics)
adam_large_stepsize_dict[1] = {"data_passes": adam.data_passes, "metrics": adam.metrics}

total_results_dict = {
    "adam small stepsize": adam_small_stepsize_dict,
    "adam medium stepsize": adam_medium_stepsize_dict,
    "adam large stepsize": adam_large_stepsize_dict,
}

with open("SheppLogan_ADAM_Results.pkl", "wb") as fp:
    pickle.dump(total_results_dict, fp)
    print("Results saved")
