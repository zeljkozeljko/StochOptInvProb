import numpy as np
from pathlib import Path
from scipy.io import loadmat
import pickle

from cil.optimisation.functions import LeastSquares, L2NormSquared
from cil.plugins.ccpi_regularisation.functions import FGP_TV

from cil.framework import  AcquisitionGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.processors import Slicer

from ProxSGM import ProxSVRG


# Load the data
path = "20201111_walnut_sinogram_data_res_280.mat"
loaded_walnut = loadmat(Path(path))

# Set up the forward operator and the data
source_center = loaded_walnut["distanceSourceOrigin"][0][0]
source_detector = loaded_walnut["distanceSourceDetector"][0][0]
pixel_size = loaded_walnut["pixelSize"][0][0]
angles = loaded_walnut["angles"][0][:-1:2]
num_angles = len(angles)
device = "gpu"

# Ground Truth
full_sinogram = np.array(loaded_walnut["sinogram"], dtype=np.float64)
reduced_sinogram = full_sinogram[:-1:2, :]
num_detectors = reduced_sinogram.shape[1]

ag = (
    AcquisitionGeometry.create_Cone2D(
        source_position=[0, -source_center],
        detector_position=[0, source_detector - source_center],
    )
    .set_panel(num_pixels=num_detectors, pixel_size=pixel_size)
    .set_angles(angles=-angles, angle_unit="degree")
)
ig = ag.get_ImageGeometry()
A = ProjectionOperator(ig, ag, device=device)

# Clean data
clean_data = ag.allocate()
clean_data.fill(reduced_sinogram)

# Load the noisy sinograms, reference solution (and hyper parameters) and objective value for performance metrics
with open("Walnut_NP_Reference.pkl", "rb") as f:
    ref_dict = pickle.load(f)

sinograms = ref_dict["sinograms"]
intensities = ref_dict["intensities"]
ref_solns = ref_dict["solutions"]
ref_vals = ref_dict["values"]

# Recover the correct reference terms. 1 corresponds to medium intensity
ref_soln_np = ref_solns[1]
ref_value = ref_vals[1]
sinogram_np = sinograms[1]

ref_soln = ig.allocate()
ref_soln.fill(ref_soln_np)
sinogram = ag.allocate()
sinogram.fill(sinogram_np)

# Objective and optimisation settings
alphas = ref_dict["alphas"]
proxfriendly_g = (alphas[1]/ig.voxel_size_x) * FGP_TV(max_iteration = 100, device=device)
num_epochs = 100
subsets = [30, 120]

# Storing the results
svrg_small_stepsize_dict = {}
svrg_medium_stepsize_dict = {}
svrg_large_stepsize_dict = {}



for l, num_subsets in enumerate(subsets):
    print("\t working on {} subsets".format(num_subsets))

    # Partition the data into subsets and create the functions
    datas = [
        Slicer(roi={"angle": (i, num_angles, num_subsets)})(sinogram)
        for i in range(num_subsets)
    ]
    Ais = [
        ProjectionOperator(ig, data_batch.geometry, device=device)
        for data_batch in datas
    ]

    smooth_fs = [LeastSquares(Ai, b=datai, c=0.5) for Ai, datai in zip(Ais, datas)]
    smooth_preop_fs = [0.5 * L2NormSquared(b=datai) for datai in datas]
    
    # Set up the metrics
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
    Lmax = max([Ai.norm()**2 for Ai in Ais])

    # SVRG with stepsize 1/3Lmax
    svrg = ProxSVRG(smooth_fs=smooth_fs, proxfriendly_g=proxfriendly_g, Lmax=Lmax, svrg_type='lsq_datafit', operators=Ais, stepsizes = lambda iter: 1./3./Lmax)
    svrg.run(initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics)
    svrg_small_stepsize_dict[l] = {'data_passes': svrg.data_passes, 'metrics': svrg.metrics}

    # SVRG with stepsize 1/Lmax
    svrg = ProxSVRG(smooth_fs=smooth_fs, proxfriendly_g=proxfriendly_g, Lmax=Lmax, svrg_type='lsq_datafit', operators=Ais, stepsizes = lambda iter: 1./Lmax)
    svrg.run(initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics)
    svrg_medium_stepsize_dict[l] = {'data_passes': svrg.data_passes, 'metrics': svrg.metrics}

    # SVRG with stepsize 2/Lmax
    svrg = ProxSVRG(smooth_fs=smooth_fs, proxfriendly_g=proxfriendly_g, Lmax=Lmax, svrg_type='lsq_datafit', operators=Ais, stepsizes = lambda iter: 2./Lmax)
    svrg.run(initial=ig.allocate(0), verbose=False, num_epochs=num_epochs, metrics=metrics)
    svrg_large_stepsize_dict[l] = {'data_passes': svrg.data_passes, 'metrics': svrg.metrics}

        
total_results_dict = {
    'svrg small stepsize' : svrg_small_stepsize_dict,
    'svrg_medium_stepsize' : svrg_medium_stepsize_dict,
    'svrg_large_stepsize' : svrg_large_stepsize_dict
}

import pickle
with open('Walnut_SVRGStepsize_Results.pkl', 'wb') as fp:
    pickle.dump(total_results_dict, fp)
    print('Results saved')