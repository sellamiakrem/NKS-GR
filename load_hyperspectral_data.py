import os
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from utils.plots import plot_rgb_image
import nibabel as nib
import numpy as np
import h5py
try:
    import spectral
except ImportError:
    spectral = None

def load_data(path, dataset_name, rgb=False, r_idx=30, g_idx=20, b_idx=10):
    """
    Load hyperspectral data and labels from various formats.

    Parameters:
        path (str): Directory containing the dataset files.
        dataset_name (str): Dataset identifier (e.g. 'IP', 'PU', 'Salinas', ...)
        rgb (bool): whether to plot RGB composite
        r_idx (int), g_idx (int), b_idx (int): indices for RGB composite

    Returns:
        X (ndarray): Spectral data as (n_samples, n_bands).
        y (ndarray): Class labels as 1D array.
    """
    dataset_name = dataset_name.lower()

    config = {
        "ip": {
            "type": "mat",
            "data_file": "Indian_pines_corrected.mat",
            "label_file": "Indian_pines_gt.mat",
            "data_key": "indian_pines_corrected",
            "label_key": "indian_pines_gt",
            "label_names": ['Unlabeled',
                'Alfalfa (C1)', 'Corn-notill (C2)', 'Corn-mintill (C3)', 'Corn (C4)',
                'Grass-pasture (C5)', 'Grass-trees (C6)', 'Grass-pasture-mowed (C7)',
                'Hay-windrowed (C8)', 'Oats (C9)', 'Soybean-notill (C10)', 'Soybean-mintill (C11)',
                'Soybean-clean (C12)', 'Wheat (C13)', 'Woods (C14)',
                'Buildings-Grass-Trees-Drives (C15)', 'Stone-Steel-Towers (C16)'
            ]
        },
        "pu": {
            "type": "mat",
            "data_file": "PaviaU.mat",
            "label_file": "PaviaU_gt.mat",
            "data_key": "paviaU",
            "label_key": "paviaU_gt",
            "label_names": [
                'Asphalt', 'Meadows', 'Gravel', 'Trees',
                'Painted metal sheets', 'Bare Soil',
                'Bitumen', 'Self-Blocking Bricks', 'Shadows'
            ]
        },
        "salinas": {
            "type": "mat",
            "data_file": "Salinas_corrected.mat",
            "label_file": "Salinas_gt.mat",
            "data_key": "salinas_corrected",
            "label_key": "salinas_gt",
            "label_names": ['Unlabeled',
                'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
                'Fallow_rough_plow', 'Fallow_smooth', 'Stubble',
                'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                'Lettuce_romaine_7wk', 'Vinyard_untrained',
                'Vinyard_vertical_trellis'
            ]
        },
        "botswana": {
        "type": "mat",
        "data_file": "Botswana.mat",
        "label_file": "Botswana_gt.mat",
        "data_key": "Botswana",
        "label_key": "Botswana_gt",
        "label_names": ['Unlabeled',
            'Water', 'Hippo grass', 'Floodplain grasses 1', 'Floodplain grasses 2',
            'Reeds', 'Riparian', 'Fire break', 'Island interior',
            'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands',
            'Short mopane', 'Mixed mopane', 'Exposed soils'
        ]
    },
        "houston13": {
            "type": "mat-single",  # single file
            "data_file": "Houston2013.mat",  # 
            "data_key": "data_hs",
            "label_key": "all_labels",
            "label_names_key": "class_names"
        },
           "brain": {
            "type": "npy", 
            "data_file": "medical_image.nii.gz",  
            "label_file": "labels.npy",
            "label_names": ['Unlabeled',
            'NT', 'TT', 'HT', 'B'
        ]
        },
        "custom_envi": {
            "type": "envi",
            "data_file": "data.hdr",
            "label_file": "labels.npy"
            # Optional: "label_names": [...]
        }
    }


    if dataset_name not in config:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(config.keys())}")

    info = config[dataset_name]
    dtype = info["type"]
    if dtype == "mat":
        data_file = os.path.join(path, info["data_file"])
        label_file = os.path.join(path, info["label_file"])
        l_names = info.get("label_names", None)

        try:
            data = sio.loadmat(data_file)[info["data_key"]]
            labels = sio.loadmat(label_file)[info["label_key"]]
        except NotImplementedError:
            print("read via h5py (MATLAB v7.3 format)...")

            # Lecture des clés disponibles pour debug
            with h5py.File(data_file, 'r') as f:
                print(f"keys available in {info['data_file']} : {list(f.keys())}")
                if info["data_key"] in f:
                    data = f[info["data_key"]][()]
                else:
                    raise KeyError(f"Key '{info['data_key']}' not found in the file {info['data_file']}")
                data = data.transpose()

            with h5py.File(label_file, 'r') as f:
                print(f"Keys available in {info['label_file']} : {list(f.keys())}")
                if info["label_key"] in f:
                    labels = f[info["label_key"]][()]
                else:
                    raise KeyError(f"Key '{info['label_key']}' not found in the file {info['label_file']}")
                labels = labels.transpose()

        cluster_names = [f"C{i+1}" for i in range(len(l_names))] if l_names else None
        
    elif dtype == "mat-single":
        full = sio.loadmat(os.path.join(path, info["data_file"]))
        data = full[info["data_key"]]
        labels = full[info["label_key"]]
        l_names = [str(c[0]) for c in full[info["label_names_key"]][0]]
        cluster_names = [f"C{i+1}" for i in range(len(l_names))] if l_names else None

    elif dtype == "npy":
        nifti_img= nib.load(os.path.join(path, info["data_file"]))
        data=nifti_img.get_fdata()
        labels = np.load(os.path.join(path, info["label_file"]))
        l_names = info.get("label_names", None)
        cluster_names = [f"C{i+1}" for i in range(len(l_names))] if l_names else None

    elif dtype == "envi":
        if spectral is None:
            raise ImportError("Please install `spectral` to read ENVI format: pip install spectral")

        data = spectral.envi.open(os.path.join(path, info["data_file"])).load()
        labels = np.load(os.path.join(path, info["label_file"]))
        l_names = info.get("label_names", None)
        cluster_names = [f"C{i+1}" for i in range(len(l_names))] if l_names else None

    else:
        raise ValueError(f"Unknown data type: {dtype}")

    X = data.reshape(-1, data.shape[-1])
    y = labels.flatten()
    valid_idx = y > 0
    X = X[valid_idx]
    y = y[valid_idx] - 1
    X_scaled = StandardScaler().fit_transform(X)
    if rgb:
        plot_rgb_image(data, r_idx, g_idx, b_idx, title=f"RGB composite of {dataset_name}")

    return X_scaled, y, l_names, cluster_names, labels