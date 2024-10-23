"""
This python code demonstrates an edge-based active contour model as an application of the
Distance Regularized Level Set Evolution (DRLSE) formulation in the following paper:

  C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation",
     IEEE Trans. Image Processing, vol. 19 (12), pp. 3243-3254, 2010.

Author: Ramesh Pramuditha Rathnayake
E-mail: rsoft.ramesh@gmail.com

Released Under MIT License
"""

import numpy as np
from scipy.ndimage import laplace

from lv_set.potential_func import SINGLE_WELL, DOUBLE_WELL
import torch


def drlse_edge(phi_0, g, lmda, mu, alfa, epsilon, timestep, iters, potential_function):  # Updated Level Set Function
    """

    :param phi_0: level set function to be updated by level set evolution
    :param g: edge indicator function
    :param lmda: weight of the weighted length term
    :param mu: weight of distance regularization term
    :param alfa: weight of the weighted area term
    :param epsilon: width of Dirac Delta function
    :param timestep: time step
    :param iters: number of iterations
    :param potential_function: choice of potential function in distance regularization term.
%              As mentioned in the above paper, two choices are provided: potentialFunction='single-well' or
%              potentialFunction='double-well', which correspond to the potential functions p1 (single-well)
%              and p2 (double-well), respectively.
    """
    phi = phi_0.copy()
    [vy, vx] = np.gradient(g)
    for k in range(iters):
        phi = neumann_bound_cond(phi)
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        delta = 1e-10
        n_x = phi_x / (s + delta)  # add a small positive number to avoid division by zero
        n_y = phi_y / (s + delta)
        curvature = div(n_x, n_y)

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace(phi, mode='nearest') - curvature  # compute distance regularization term in equation (13) with the single-well potential p1.
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2(phi)  # compute the distance regularization term in eqaution (13) with the double-well potential p2.
        else:
            raise Exception('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.')
        dirac_phi = dirac(phi, epsilon)
        area_term = dirac_phi * g  # balloon/pressure force
        edge_term = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * g * curvature
        phi += timestep * (mu * dist_reg_term + lmda * edge_term + alfa * area_term)
    return phi

def drlse_threshold(phi_0, img, lmda, mu, alfa, epsilon,upper,lower, timestep, iters, potential_function):  # Updated Level Set Function
    """

    :param phi_0: level set function to be updated by level set evolution
    :param g: edge indicator function
    :param lmda: weight of the weighted length term
    :param mu: weight of distance regularization term
    :param alfa: weight of the weighted area term
    :param epsilon: width of Dirac Delta function
    :param upper: upper threshold
    :param lower: lower threshold 
    :param timestep: time step
    :param iters: number of iterations
    :param potential_function: choice of potential function in distance regularization term.
%              As mentioned in the above paper, two choices are provided: potentialFunction='single-well' or
%              potentialFunction='double-well', which correspond to the potential functions p1 (single-well)
%              and p2 (double-well), respectively.
    """
    phi = phi_0.copy()
    eps = 0.5*(upper-lower)
    T = 0.5*(upper+lower)
    # [vy, vx] = np.gradient(g)
    for k in range(iters):
        phi = neumann_bound_cond(phi)
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        delta = 1e-10
        n_x = phi_x / (s + delta)  # add a small positive number to avoid division by zero
        n_y = phi_y / (s + delta)
        curvature = div(n_x, n_y)

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace(phi, mode='nearest') - curvature  # compute distance regularization term in equation (13) with the single-well potential p1.
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2(phi)  # compute the distance regularization term in eqaution (13) with the double-well potential p2.
        else:
            raise Exception('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.')
        dirac_phi = dirac(phi, epsilon)
        # print(eps,T,np.min(phi),np.max(phi),np.min(img),np.max(img))
        area_term =  (eps-np.fabs(img-T))/eps*dirac_phi*80.0
        # print(np.min(area_term),np.max(area_term))
        edge_term = curvature*dirac_phi # curvature term as edge term
        phi += timestep *0.2* (mu * dist_reg_term + lmda * edge_term + alfa * area_term)
    return phi

def dist_reg_p2(phi):
    """
        compute the distance regularization term with the double-well potential p2 in equation (16)
    """
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)  # compute first order derivative of the double-well potential p2 in equation (16)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))  # compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + laplace(phi, mode='nearest')


def div(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy


def dirac(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def neumann_bound_cond(f):
    """
        Make a function satisfy Neumann boundary condition
    """
    g = f.copy()

    g[np.ix_([0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1] = g[np.ix_([2, -3]), 1:-1]
    g[1:-1, np.ix_([0, -1])] = g[1:-1, np.ix_([2, -3])]
    return g

# New methods for narrow band implementation

def find_zero_crossings(phi):
    """
    Find the zero-crossing points of the level set function phi.
    Zero-crossings are where the sign of phi changes.
    """
    # Calculate differences along each axis and convert to boolean
    diff_0 = np.diff(np.sign(phi), axis=0).astype(bool)
    diff_1 = np.diff(np.sign(phi), axis=1).astype(bool)

    # Pad the differences with an extra row/column of False to match the original shape of `phi`
    padded_diff_0 = np.pad(diff_0, ((0, 1), (0, 0)), mode='constant', constant_values=False)
    padded_diff_1 = np.pad(diff_1, ((0, 0), (0, 1)), mode='constant', constant_values=False)

    # Combine the padded differences using bitwise OR and create the zero-crossings array
    zero_crossings = np.where(padded_diff_0 | padded_diff_1, 1, 0)
    return zero_crossings

def initialize_narrow_band(phi, r=3):
    """
    Initialize the narrow band based on zero-crossing points of phi and a neighborhood radius r.
    """
    zero_crossings = find_zero_crossings(phi)
    narrow_band = np.zeros_like(phi, dtype=bool)

    # Mark points in a neighborhood of radius r around zero-crossing points
    indices = np.argwhere(zero_crossings > 0)
    for index in indices:
        i, j = index
        narrow_band[max(0, i-r):min(i+r+1, phi.shape[0]), max(0, j-r):min(j+r+1, phi.shape[1])] = True

    return narrow_band

def drlse_edge_narrow_band(phi_0, g, lmda, mu, alfa, epsilon, timestep, iters, potential_function, r=3, h=4):
    """
    Refined narrow band implementation of the edge-based DRLSE evolution based on provided steps.
    """
    phi = phi_0.copy()
    [vy, vx] = np.gradient(g)
    
    # Initialize narrow band
    narrow_band = initialize_narrow_band(phi, r)
    
    for k in range(iters):
        # Update the LSF only within the narrow band
        phi = neumann_bound_cond(phi)
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        curvature = div(n_x, n_y)

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace(phi, mode='nearest') - curvature
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2(phi)
        else:
            raise Exception('Error: Wrong choice of potential function.')

        dirac_phi = dirac(phi, epsilon)
        area_term = dirac_phi * g
        edge_term = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * g * curvature

        # Apply updates only in the narrow band
        phi[narrow_band] += timestep * (mu * dist_reg_term[narrow_band] + lmda * edge_term[narrow_band] + alfa * area_term[narrow_band])

        # Step 3: Update the narrow band by finding zero-crossing points and extending the band
        new_zero_crossings = find_zero_crossings(phi)
        new_indices = np.argwhere(new_zero_crossings > 0)

        # Update the narrow band by adding neighborhoods around new zero-crossing points
        new_narrow_band = np.zeros_like(phi, dtype=bool)
        for index in new_indices:
            i, j = index
            new_narrow_band[max(0, i-r):min(i+r+1, phi.shape[0]), max(0, j-r):min(j+r+1, phi.shape[1])] = True
        
        # Assign values to new pixels in the narrow band based on step 4
        newly_added_points = new_narrow_band & ~narrow_band
        phi[newly_added_points] = np.where(phi[newly_added_points] > 0, h, -h)

        # Update narrow band for the next iteration
        narrow_band = new_narrow_band.copy()

        # Termination condition based on zero-crossing changes (optional for early stop)

    return phi

def drlse_threshold_narrow_band(phi_0, img, lmda, mu, alfa, epsilon, upper, lower, timestep, iters, potential_function, r=3, h=4):
    """
    Refined narrow band implementation of the threshold-based DRLSE evolution based on provided steps.
    """
    phi = phi_0.copy()
    eps = 0.5 * (upper - lower)
    T = 0.5 * (upper + lower)

    # Step 1: Initialize narrow band
    narrow_band = initialize_narrow_band(phi, r)

    for k in range(iters):
        # Update the LSF only within the narrow band
        phi = neumann_bound_cond(phi)
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        curvature = div(n_x, n_y)

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace(phi, mode='nearest') - curvature
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2(phi)
        else:
            raise Exception('Error: Wrong choice of potential function.')

        dirac_phi = dirac(phi, epsilon)
        area_term = (eps - np.abs(img - T)) / eps * dirac_phi * 80.0
        edge_term = curvature * dirac_phi

        # Apply updates only in the narrow band
        phi[narrow_band] += timestep * 0.2 * (mu * dist_reg_term[narrow_band] + lmda * edge_term[narrow_band] + alfa * area_term[narrow_band])

        # Step 3: Update the narrow band by finding zero-crossing points and extending the band
        new_zero_crossings = find_zero_crossings(phi)
        new_indices = np.argwhere(new_zero_crossings > 0)

        # Update the narrow band by adding neighborhoods around new zero-crossing points
        new_narrow_band = np.zeros_like(phi, dtype=bool)
        for index in new_indices:
            i, j = index
            new_narrow_band[max(0, i-r):min(i+r+1, phi.shape[0]), max(0, j-r):min(j+r+1, phi.shape[1])] = True

        # Step 4: Assign values to new pixels in the narrow band based on step 4
        newly_added_points = new_narrow_band & ~narrow_band
        phi[newly_added_points] = np.where(phi[newly_added_points] > 0, h, -h)

        # Update narrow band for the next iteration
        narrow_band = new_narrow_band.copy()

        # Step 5: Optional termination condition based on zero-crossing changes

    return phi

### GPU-BASED IMPLEMENTATION FOR 2D USING PYTORCH ###

def drlse_edge_gpu(phi_0, g, lmda, mu, alfa, epsilon, timestep, iters, potential_function, device="cuda"):
    """
    GPU-accelerated implementation of edge-based DRLSE using PyTorch for 2D images.
    """
    phi = torch.tensor(phi_0, dtype=torch.float32, device=device).clone()
    g = torch.tensor(g, dtype=torch.float32, device=device)
    [vy, vx] = torch.gradient(g)  # Compute gradients on the GPU

    for k in range(iters):
        phi = neumann_bound_cond_gpu(phi)  # Neumann boundary condition for 2D on GPU
        [phi_y, phi_x] = torch.gradient(phi)  # 2D gradient on GPU

        s = torch.sqrt(phi_x**2 + phi_y**2)  # 2D norm of gradients
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        curvature = div_gpu(n_x, n_y)  # 2D divergence on GPU

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace_gpu(phi) - curvature
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2_gpu_2d(phi)
        else:
            raise Exception('Error: Wrong choice of potential function.')

        dirac_phi = dirac_gpu(phi, epsilon)
        area_term = dirac_phi * g
        edge_term = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * g * curvature
        phi += timestep * (mu * dist_reg_term + lmda * edge_term + alfa * area_term)

    return phi.cpu().numpy()  # Move result back to CPU for further processing if needed

def drlse_threshold_gpu(phi_0, img, lmda, mu, alfa, epsilon, upper, lower, timestep, iters, potential_function, device="cuda"):
    """
    GPU-accelerated implementation of threshold-based DRLSE using PyTorch for 2D images.
    """
    phi = torch.tensor(phi_0, dtype=torch.float32, device=device).clone()
    img = torch.tensor(img, dtype=torch.float32, device=device)
    eps = 0.5 * (upper - lower)
    T = 0.5 * (upper + lower)

    for k in range(iters):
        phi = neumann_bound_cond_gpu(phi)
        [phi_y, phi_x] = torch.gradient(phi)
        s = torch.sqrt(phi_x**2 + phi_y**2)
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        curvature = div_gpu(n_x, n_y)

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace_gpu(phi) - curvature
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2_gpu(phi)
        else:
            raise Exception('Error: Wrong choice of potential function.')

        dirac_phi = dirac_gpu(phi, epsilon)
        area_term = (eps - torch.abs(img - T)) / eps * dirac_phi * 80.0
        edge_term = curvature * dirac_phi
        phi += timestep * 0.2 * (mu * dist_reg_term + lmda * edge_term + alfa * area_term)

    return phi.cpu().numpy()

def laplace_gpu(input):
    """
    Compute the Laplacian on GPU using a correct 2D convolution kernel.
    """
    # Create a 2D Laplacian kernel with shape (1, 1, 3, 3)
    kernel = torch.tensor([[[[0, 1, 0], 
                             [1, -4, 1], 
                             [0, 1, 0]]]], 
                          device=input.device, dtype=torch.float32)
    
    # Reshape kernel to the correct shape for conv2d: (out_channels, in_channels, height, width)
    kernel = kernel.view(1, 1, 3, 3)  # (1, 1, H, W)
    
    # Add batch and channel dimensions to the input tensor (N, C, H, W)
    input = input.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Perform 2D convolution with appropriate padding
    lap = torch.nn.functional.conv2d(input, kernel, stride=1, padding=1)
    
    return lap.squeeze()  # Remove the added dimensions

def dirac_gpu(x, sigma):
    """
    Compute the Dirac delta function on GPU for 2D images.
    """
    f = (1 / (2 * sigma)) * (1 + torch.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b

def neumann_bound_cond_gpu(f):
    """
    Apply Neumann boundary conditions on GPU for 2D images.
    """
    g = f.clone()
    g[0, :], g[-1, :], g[:, 0], g[:, -1] = g[1, :], g[-2, :], g[:, 1], g[:, -2]
    return g

def div_gpu(nx, ny):
    """
    Compute 2D divergence on GPU.
    """
    _, nxx = torch.gradient(nx)
    nyy, _ = torch.gradient(ny)
    return nxx + nyy

def dist_reg_p2_gpu(phi):
    """
    Compute the distance regularization term with double-well potential p2 on GPU for 2D images.
    """
    [phi_y, phi_x] = torch.gradient(phi)
    s = torch.sqrt(phi_x**2 + phi_y**2)
    
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * torch.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))

    return div_gpu(dps * phi_x - phi_x, dps * phi_y - phi_y) + laplace_gpu(phi)
