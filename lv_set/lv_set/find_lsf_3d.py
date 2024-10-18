import numpy as np
from scipy.ndimage import gaussian_filter
from lv_set.drlse_algo_3d import drlse_edge, drlse_threshold
from lv_set.potential_func import DOUBLE_WELL, SINGLE_WELL
from lv_set.seg_method import EDGE, THRESHOLD
from visualize_3d import visualize_3d_image_and_phi_dynamic  # Updated dynamic PyVista visualization function
from lv_set.save_image import dump_image_to_nii

def find_lsf(img: np.ndarray, initial_lsf: np.ndarray, timestep=1, iter_inner=10, iter_outer=30, lmda=5,
             alfa=-3, epsilon=1.5, sigma=0.8, upper=2, lower=-2, potential_function=DOUBLE_WELL, seg_method=EDGE):
    """
    :param img: Input 3D image as a grayscale uint8 array (0-255)
    :param initial_lsf: Array of the same size as img that contains the seed points for the LSF.
    :param timestep: Time Step
    :param iter_inner: How many iterations to run DRLSE before showing the output
    :param iter_outer: How many iterations to run the iter_inner loop
    :param lmda: coefficient of the weighted length term L(phi)
    :param alfa: coefficient of the weighted area term A(phi)
    :param epsilon: parameter that specifies the width of the DiracDelta function
    :param sigma: scale parameter in Gaussian kernel
    :param potential_function: Potential function to use in DRLSE algorithm. Should be SINGLE_WELL or DOUBLE_WELL
    """
    if len(img.shape) != 3:
        raise Exception("Input image should be a 3D grayscale image")

    if len(img.shape) != len(initial_lsf.shape):
        raise Exception("Input image and the initial LSF should be the same shape")

    if np.max(img) <= 1:
        raise Exception("Please make sure the image data is in the range [0, 255]")

    # parameters
    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)

    # Convert image to float32 for computation
    img = np.array(img, dtype='float32')

    # Smooth the image using a 3D Gaussian filter
    img_smooth = gaussian_filter(img, sigma)

    # Compute the 3D gradient of the smoothed image
    [Iz, Iy, Ix] = np.gradient(img_smooth)

    # Compute the edge indicator function
    f = np.square(Ix) + np.square(Iy) + np.square(Iz)
    g = 1 / (1 + f)
    dump_image_to_nii(g, "gradient.nii.gz")

    # Initialize the level set function (LSF)
    phi = initial_lsf.copy()

    if potential_function != SINGLE_WELL:
        potential_function = DOUBLE_WELL  # default potential function

    # Create a PyVista plotter for dynamic updates
    plotter = None

    # Start level set evolution
    for n in range(iter_outer):
        # Perform the segmentation based on the chosen method
        if seg_method == EDGE:
            phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potential_function)
        elif seg_method == THRESHOLD:
            phi = drlse_threshold(phi, img, lmda, mu, alfa, epsilon, upper, lower, timestep, iter_inner, potential_function)
        else:
            raise Exception("Only support edge or threshold segmentation method!")

        # Visualize the 3D image and segmentation result using dynamic PyVista after each iteration
        print(f"Visualizing after {n + 1} iterations")
        plotter = visualize_3d_image_and_phi_dynamic(img, phi, plotter)

    # Refine the zero-level contour by running additional iterations with alfa=0
    alfa = 0
    iter_refine = 10

    # Final refinement using the chosen segmentation method
    if seg_method == EDGE:
        phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potential_function)
    elif seg_method == THRESHOLD:
        phi = drlse_threshold(phi, img, lmda, mu, alfa, epsilon, upper, lower, timestep, iter_inner, potential_function)
    else:
        raise Exception("Only support edge or threshold segmentation method!")

    # Final visualization of the refined result
    visualize_3d_image_and_phi_dynamic(img, phi, plotter)

    return phi
