import numpy as np
from lv_set.find_lsf_3d import find_lsf
from lv_set.potential_func import DOUBLE_WELL
from lv_set.seg_method import *

import nibabel as nib

def dump_image_to_nii(image_data: np.ndarray, file_name: str, affine=None):
    """
    Dump 3D image data to a NIfTI (.nii) file.
    
    :param image_data: 3D NumPy array representing the image data.
    :param file_name: The file name to save the NIfTI file (e.g., 'output.nii' or 'output.nii.gz').
    :param affine: Affine transformation matrix (optional). Default is identity matrix.
    """
    # If no affine transformation is provided, use identity matrix
    if affine is None:
        affine = np.eye(4)
    
    # Create a NIfTI image object from the image data and affine matrix
    nifti_image = nib.Nifti1Image(image_data, affine)
    
    # Save the NIfTI image to the specified file
    nib.save(nifti_image, file_name)
    print(f"NIfTI image saved to {file_name}")

def create_3d_sphere_image(size=50, intensity_range=(0, 255)):
    """
    Create a 3D sphere image where the sphere is black (gradually towards the center) 
    and the most outer voxel is white (intensity 255).
    
    :param size: Size of the 3D image (assumed to be cubic: size x size x size)
    :param radius: Radius of the sphere
    :param intensity_range: Tuple indicating the range of intensity values (center intensity, boundary intensity)
    
    :return: A 3D numpy array representing the sphere image.
    """
    # Create a 3D grid of coordinates
    radius = size/2*np.sqrt(2)
    x = np.linspace(-radius, radius, size)
    y = np.linspace(-radius, radius, size)
    z = np.linspace(-radius, radius, size)
    X, Y, Z = np.meshgrid(x, y, z)

    # Calculate the distance from the center for each voxel
    distance_to_center = np.sqrt(X**2 + Y**2 + Z**2)

    # Calculate the intensity based on the distance to the center, 
    # mapping distances from [0, radius] to [min_intensity, max_intensity]
    min_intensity, max_intensity = intensity_range
    img = distance_to_center / radius
    # img = np.clip(((distance_to_center / radius) * (max_intensity - min_intensity)) + min_intensity, min_intensity, max_intensity)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])
    return img


def sphere_params():
    # Create the 3D sphere image
    img = create_3d_sphere_image(size=50, intensity_range=(0, 255))
    dump_image_to_nii(img,"sphere.nii.gz")

    print(np.min(img), np.max(img))
    
    # Initialize the level set function (LSF) as a binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    
    # Create an initial region inside the sphere (this can be any arbitrary region)
    initial_lsf[20:30, 20:30, 20:30] = -c0  # A small cuboid region inside the sphere

    # Parameters for the level set segmentation
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 10,
        'iter_outer': 30,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -3,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 0.8,  # scale parameter in Gaussian kernel
        'upper': 127.5,  # upper intensity for thresholding
        'lower': -1,    # lower intensity for thresholding
        'potential_function': DOUBLE_WELL,
        'seg_method': THRESHOLD
    }

# Get the parameters for the sphere image
params = sphere_params()

# Run the level set segmentation on the 3D sphere image
phi = find_lsf(**params)
