import numpy as np
from pathlib import Path
from scipy.io import loadmat
import pickle


from cil.optimisation.functions import LeastSquares, L2NormSquared, BlockFunction
from cil.optimisation.operators import BlockOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV

from cil.framework import AcquisitionGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.processors import Slicer

from ProxSGM import ProxSAGA, ProxSGD, ProxSVRG
from cil.optimisation.algorithms import PDHG, SPDHG

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
Lipsch = A.norm() ** 2

# Clean data
clean_data = ag.allocate()
clean_data.fill(reduced_sinogram)

num_epochs = 40
num_subsets = 60

# Load the noisy sinograms, reference solution (and hyper parameters) and objective value for performance metrics
with open("Walnut_NP_Reference.pkl", "rb") as f:
    ref_dict = pickle.load(f)

sinograms = ref_dict["sinograms"]
intensities = ref_dict["intensities"]
alphas = ref_dict["alphas"]
ref_solns = ref_dict["solutions"]
ref_vals = ref_dict["values"]

# Storing the results
pgd_dict = {}
sgd_dict = {}
saga_dict = {}
svrg_dict = {}
spdhg_dict = {}
pdhg_dict = {}

print("\t working on {} subsets".format(num_subsets))

# Reconstruction for each intensity (noise level)
for k, (key, intensity) in enumerate(intensities.items()):
    print("=" * 30)
    print(
        "Working on intensity={} with alpha={} and k={}".format(
            intensity, alphas[key], k
        )
    )
    print("=" * 30)

    # Recover the correct reference terms
    ref_soln_np = ref_solns[k]
    ref_value = ref_vals[k]
    sinogram_np = sinograms[k]

    ref_soln = ig.allocate()
    ref_soln.fill(ref_soln_np)
    sinogram = ag.allocate()
    sinogram.fill(sinogram_np)

    proxfriendly_g = (alphas[k] / ig.voxel_size_x) * FGP_TV(
        max_iteration=100, device=device
    )

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
    
    # PGD
    smooth_f = LeastSquares(A, b = sinogram, c = 0.5)
    pgd = ProxSGD(
        smooth_fs=[smooth_f],
        proxfriendly_g=proxfriendly_g,
        stepsizes=lambda iter: 1.0 / Lipsch,
    )
    pgd.run(
        initial=ig.allocate(0),
        verbose=False,
        num_epochs=num_epochs,
        metrics=metrics
    )
    pgd_dict[k] = {"data_passes": pgd.data_passes, "metrics": pgd.metrics}



    Lmax = max([Ai.norm() ** 2 for Ai in Ais])
    # SGD
    sgd = ProxSGD(smooth_fs=smooth_fs, proxfriendly_g=proxfriendly_g, Lmax=Lmax)
    sgd.run(
        initial=ig.allocate(0),
        verbose=False,
        num_epochs=num_epochs,
        metrics=metrics,
    )
    
    # SAGA
    saga = ProxSAGA(smooth_fs=smooth_fs, proxfriendly_g=proxfriendly_g, Lmax=Lmax)
    saga.run(
        initial=ig.allocate(0),
        verbose=False,
        num_epochs=num_epochs,
        metrics=metrics,
    )
    saga_dict[k] = {"data_passes": saga.data_passes, "metrics": saga.metrics}

    # SVRG
    svrg = ProxSVRG(
        smooth_fs=smooth_fs,
        proxfriendly_g=proxfriendly_g,
        Lmax=Lmax,
        svrg_type="lsq_datafit",
        operators=Ais,
    )
    svrg.run(
        initial=ig.allocate(0),
        verbose=False,
        num_epochs=num_epochs,
        metrics=metrics,
    )
    svrg_dict[k] = {"data_passes": svrg.data_passes, "metrics": svrg.metrics}

    # PDHG
    smooth_preop_f = 0.5 * L2NormSquared(b=sinogram)
    pdhg = PDHG(f=smooth_preop_f, g=proxfriendly_g, operator=A, max_iteration=num_epochs, update_objective_interval=1)
    pdhg.run(verbose=0, iterations = num_epochs)
    if ref_soln is not None:  
        pdhg_dict[k] = {"data_passes": pgd.data_passes} 
        pdhg_dict[k]["metrics"] = {}
        pdhg_dict[k]["metrics"]["relative objective"] = [obj - ref_value for obj in pdhg.objective]
        pdhg_dict[k]["metrics"]["objective"] = pdhg.objective

    # SPDHG
    Aiblock = BlockOperator(*Ais)
    smooth_preopfs_block = BlockFunction(*smooth_preop_fs)

    spdhg = SPDHG(f=smooth_preopfs_block, g=proxfriendly_g, operator=Aiblock, max_iteration=num_epochs*num_subsets, gamma=1, update_objective_interval=1)
    spdhg.run(verbose=0, iterations = num_epochs*num_subsets)

    if ref_soln is not None:  
        spdhg_dict[k] = {"data_passes": sgd.data_passes} 
        spdhg_dict[k]["metrics"] = {}
        spdhg_dict[k]["metrics"]["relative objective"] = [obj - ref_value for obj in spdhg.objective]
        spdhg_dict[k]["metrics"]["objective"] = spdhg.objective

total_results_dict = {
    "pgd" : pgd_dict,
    "sgd": sgd_dict,
    "svrg": svrg_dict,
    "saga": saga_dict,
    "pdhg" : pdhg_dict,
    "spdhg" : spdhg_dict
}

with open("Walnut_Results.pkl", "wb") as fp:
    pickle.dump(total_results_dict, fp)
    print("Results saved")
