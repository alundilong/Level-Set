import numpy as np
from lv_set.find_lsf_3d import find_lsf
from lv_set.potential_func import DOUBLE_WELL, SINGLE_WELL
from lv_set.seg_method import *
from lv_set.save_image import dump_image_to_vtk

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
    dump_image_to_vtk(img,"sphere.vti")

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


def create_3d_sharp_sphere_image(size=50, intensity_range=(0, 255)):
    """
    Create a 3D sphere image where the inside of the sphere has intensity 0 (black) 
    and the outside of the sphere has intensity 255 (white).
    
    :param size: Size of the 3D image (assumed to be cubic: size x size x size)
    :param intensity_range: Tuple indicating the range of intensity values (inside intensity, outside intensity)
    
    :return: A 3D numpy array representing the sphere image.
    """
    # Create a 3D grid of coordinates
    radius = size / 2  # Radius of the sphere (half the size)
    x = np.linspace(-radius, radius, size)
    y = np.linspace(-radius, radius, size)
    z = np.linspace(-radius, radius, size)
    X, Y, Z = np.meshgrid(x, y, z)

    # Calculate the distance from the center for each voxel
    distance_to_center = np.sqrt(X**2 + Y**2 + Z**2)

    # Create a binary image where inside the sphere is 0 and outside the sphere is 255
    img = np.where(distance_to_center <= radius/1.5, float(intensity_range[0]), float(intensity_range[1]))
    
    return img


def sharp_sphere_params():
    # Create the 3D sphere image
    img = create_3d_sharp_sphere_image(size=50, intensity_range=(0, 255))
    dump_image_to_vtk(img,"sharp_sphere.vti")

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
        'timestep': 5,  # time step
        'iter_inner': 10,
        'iter_outer': 30,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -3,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 0.8,  # scale parameter in Gaussian kernel
        'upper': 128,  # upper intensity for thresholding
        'lower': -1,    # lower intensity for thresholding
        'potential_function': DOUBLE_WELL,
        'seg_method': EDGE
    }

# Get the parameters for the sphere image
# params = sphere_params()
params = sharp_sphere_params()

# Run the level set segmentation on the 3D sphere image
phi = find_lsf(**params)
