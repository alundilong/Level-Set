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
import torch
from scipy.ndimage import laplace

from lv_set.potential_func import SINGLE_WELL, DOUBLE_WELL
from lv_set.save_image import dump_image_to_nii, dump_image_to_vtk

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
    if not hasattr(drlse_edge, "call_count"):
        drlse_edge.call_count = 0
    # drlse_edge.call_count += 1

    phi = phi_0.copy()
    [vz, vy, vx] = np.gradient(g)  # 3D gradient
    for k in range(iters):
        drlse_edge.call_count += 1
        phi = neumann_bound_cond(phi)
        [phi_z, phi_y, phi_x] = np.gradient(phi)  # 3D gradient
        s = np.sqrt(np.square(phi_x) + np.square(phi_y) + np.square(phi_z))  # 3D norm
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        n_z = phi_z / (s + delta)
        curvature = div(n_x, n_y, n_z)  # 3D divergence

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace(phi, mode='nearest') - curvature
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2(phi)
        else:
            raise Exception('Error: Wrong choice of potential function.')

        dirac_phi = dirac(phi, epsilon)
        area_term = dirac_phi * g
        edge_term = dirac_phi * (vx * n_x + vy * n_y + vz * n_z) + dirac_phi * g * curvature
        phi += timestep * (mu * dist_reg_term + lmda * edge_term + alfa * area_term)
        dump_image_to_vtk(phi,f"edge_innerloop_{drlse_edge.call_count}.vti")
        dump_image_to_vtk(area_term,f"edge_area_term_{drlse_edge.call_count}.vti")
        dump_image_to_vtk(edge_term,f"edge_edge_term_{drlse_edge.call_count}.vti")
        dump_image_to_vtk(dist_reg_term,f"edge_dist_reg_term_{drlse_edge.call_count}.vti")
    return phi

def drlse_threshold(phi_0, img, lmda, mu, alfa, epsilon, upper, lower, timestep, iters, potential_function):
    """
    :param phi_0: level set function to be updated by level set evolution
    :param img: the 3D input image
    :param lmda: weight of the weighted length term
    :param mu: weight of distance regularization term
    :param alfa: weight of the weighted area term
    :param epsilon: width of Dirac Delta function
    :param upper: upper threshold
    :param lower: lower threshold
    :param timestep: time step
    :param iters: number of iterations
    :param potential_function: choice of potential function (SINGLE_WELL or DOUBLE_WELL)
    """
    if not hasattr(drlse_edge, "call_count"):
        drlse_edge.call_count = 0

    phi = phi_0.copy()
    eps = 0.5 * (upper - lower)
    T = 0.5 * (upper + lower)

    for k in range(iters):
        drlse_edge.call_count += 1
        phi = neumann_bound_cond(phi)  # Neumann boundary condition for 3D
        [phi_z, phi_y, phi_x] = np.gradient(phi)  # 3D gradient
        s = np.sqrt(np.square(phi_x) + np.square(phi_y) + np.square(phi_z))  # 3D norm of gradients
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        n_z = phi_z / (s + delta)
        curvature = div(n_x, n_y, n_z)  # 3D divergence

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace(phi, mode='nearest') - curvature  # distance regularization with single-well potential
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2(phi)  # distance regularization with double-well potential
        else:
            raise Exception('Error: Wrong choice of potential function.')

        dirac_phi = dirac(phi, epsilon)

        # Threshold-based area term
        area_term = (eps - np.abs(img - T)) / eps * dirac_phi * 80.0  # balloon/pressure force term

        # Edge term (curvature term)
        edge_term = curvature * dirac_phi  # curvature term as edge term

        # Update phi using the distance regularization, edge, and area terms
        phi += timestep * 0.2 * (mu * dist_reg_term + lmda * edge_term + alfa * area_term)
        dump_image_to_vtk(phi,f"threshold_innerloop_{drlse_edge.call_count}.vti")
        dump_image_to_vtk(area_term,f"threshold_area_term_{drlse_edge.call_count}.vti")
        dump_image_to_vtk(edge_term,f"threshold_edge_term_{drlse_edge.call_count}.vti")
        dump_image_to_vtk(dist_reg_term,f"threshold_dist_reg_term_{drlse_edge.call_count}.vti")
    
    return phi

def dist_reg_p2(phi):
    """
    Compute the distance regularization term with the double-well potential p2 in equation (16)
    for a 3D image.
    """
    # Compute the gradient in 3D
    [phi_z, phi_y, phi_x] = np.gradient(phi)

    # Compute the gradient magnitude (3D norm)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y) + np.square(phi_z))

    # Double-well potential p2 (as per equation 16 in the paper)
    a = (s >= 0) & (s <= 1)
    b = (s > 1)

    # Compute p2's derivative
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)

    # Compute d_p(s) = p'(s) / s in equation (10)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))

    # Compute the 3D divergence of the double-well potential
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y, dps * phi_z - phi_z) + laplace(phi, mode='nearest')



def div(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray) -> np.ndarray:
    [nzz, _, _] = np.gradient(nz)
    [_, nyy, _] = np.gradient(ny)
    [_, _, nxx] = np.gradient(nx)
    return nxx + nyy + nzz


def dirac(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def neumann_bound_cond(f):
    g = f.copy()

    # Neumann boundary conditions for 3D
    g[np.ix_([0, -1], [0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1, 1:-1] = g[np.ix_([2, -3]), 1:-1, 1:-1]
    g[1:-1, np.ix_([0, -1]), 1:-1] = g[1:-1, np.ix_([2, -3]), 1:-1]
    g[1:-1, 1:-1, np.ix_([0, -1])] = g[1:-1, 1:-1, np.ix_([2, -3])]
    return g

def find_zero_crossings(phi):
    """
    Find the zero-crossing points of the level set function phi in 3D.
    Zero-crossings are where the sign of phi changes.
    """
    diff_0 = np.diff(np.sign(phi), axis=0).astype(bool)
    diff_1 = np.diff(np.sign(phi), axis=1).astype(bool)
    diff_2 = np.diff(np.sign(phi), axis=2).astype(bool)

    # Pad the differences with an extra layer to match the original shape of `phi`
    padded_diff_0 = np.pad(diff_0, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=False)
    padded_diff_1 = np.pad(diff_1, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=False)
    padded_diff_2 = np.pad(diff_2, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=False)

    zero_crossings = np.where(padded_diff_0 | padded_diff_1 | padded_diff_2, 1, 0)
    return zero_crossings

def initialize_narrow_band(phi, r=3):
    """
    Initialize the narrow band based on zero-crossing points of phi and a neighborhood radius r in 3D.
    """
    zero_crossings = find_zero_crossings(phi)
    narrow_band = np.zeros_like(phi, dtype=bool)

    # Mark points in a neighborhood of radius r around zero-crossing points
    indices = np.argwhere(zero_crossings > 0)
    for index in indices:
        i, j, k = index
        narrow_band[max(0, i-r):min(i+r+1, phi.shape[0]), 
                    max(0, j-r):min(j+r+1, phi.shape[1]), 
                    max(0, k-r):min(k+r+1, phi.shape[2])] = True

    return narrow_band

def drlse_edge_narrow_band(phi_0, g, lmda, mu, alfa, epsilon, timestep, iters, potential_function, r=3, h=4):
    """
    Refined narrow band implementation of the edge-based DRLSE evolution in 3D based on provided steps.
    """
    if not hasattr(drlse_edge, "call_count"):
        drlse_edge.call_count = 0

    phi = phi_0.copy()
    [vz, vy, vx] = np.gradient(g)  # 3D gradient

    # Step 1: Initialize narrow band
    narrow_band = initialize_narrow_band(phi, r)

    for k in range(iters):
        drlse_edge.call_count += 1
        # Step 2: Update the LSF only within the narrow band
        phi = neumann_bound_cond(phi)
        [phi_z, phi_y, phi_x] = np.gradient(phi)  # 3D gradient
        s = np.sqrt(np.square(phi_x) + np.square(phi_y) + np.square(phi_z))  # 3D norm
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        n_z = phi_z / (s + delta)
        curvature = div(n_x, n_y, n_z)  # 3D divergence

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace(phi, mode='nearest') - curvature
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2(phi)
        else:
            raise Exception('Error: Wrong choice of potential function.')

        dirac_phi = dirac(phi, epsilon)
        area_term = dirac_phi * g
        edge_term = dirac_phi * (vx * n_x + vy * n_y + vz * n_z) + dirac_phi * g * curvature

        # Apply updates only in the narrow band
        phi[narrow_band] += timestep * (mu * dist_reg_term[narrow_band] + lmda * edge_term[narrow_band] + alfa * area_term[narrow_band])

        # Step 3: Update the narrow band by finding zero-crossing points and extending the band
        new_zero_crossings = find_zero_crossings(phi)
        new_indices = np.argwhere(new_zero_crossings > 0)

        # Update the narrow band by adding neighborhoods around new zero-crossing points
        new_narrow_band = np.zeros_like(phi, dtype=bool)
        for index in new_indices:
            i, j, k = index
            new_narrow_band[max(0, i-r):min(i+r+1, phi.shape[0]), 
                            max(0, j-r):min(j+r+1, phi.shape[1]), 
                            max(0, k-r):min(k+r+1, phi.shape[2])] = True

        # Step 4: Assign values to new pixels in the narrow band based on step 4
        newly_added_points = new_narrow_band & ~narrow_band
        phi[newly_added_points] = np.where(phi[newly_added_points] > 0, h, -h)

        # Step 5: Update narrow band for the next iteration
        narrow_band = new_narrow_band.copy()

        # Optional: Add termination condition based on zero-crossing changes
        dump_image_to_vtk(phi,f"edge_innerloop_{drlse_edge.call_count}.vti")
    return phi

# Similarly, we can implement the threshold-based narrow band method for 3D:

def drlse_threshold_narrow_band(phi_0, img, lmda, mu, alfa, epsilon, upper, lower, timestep, iters, potential_function, r=3, h=4):
    """
    Refined narrow band implementation of the threshold-based DRLSE evolution in 3D.
    """
    if not hasattr(drlse_edge, "call_count"):
        drlse_edge.call_count = 0

    phi = phi_0.copy()
    eps = 0.5 * (upper - lower)
    T = 0.5 * (upper + lower)

    # Step 1: Initialize narrow band
    narrow_band = initialize_narrow_band(phi, r)

    for k in range(iters):
        drlse_edge.call_count += 1
        # Step 2: Update the LSF only within the narrow band
        phi = neumann_bound_cond(phi)
        [phi_z, phi_y, phi_x] = np.gradient(phi)  # 3D gradient
        s = np.sqrt(np.square(phi_x) + np.square(phi_y) + np.square(phi_z))  # 3D norm
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        n_z = phi_z / (s + delta)
        curvature = div(n_x, n_y, n_z)  # 3D divergence

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace(phi, mode='nearest') - curvature
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2(phi)
        else:
            raise Exception('Error: Wrong choice of potential function.')

        dirac_phi = dirac(phi, epsilon)

        # Threshold-based area term
        area_term = (eps - np.abs(img - T)) / eps * dirac_phi * 80.0  # balloon/pressure force term

        # Edge term (curvature term)
        edge_term = curvature * dirac_phi  # curvature term as edge term

        # Apply updates only in the narrow band
        phi[narrow_band] += timestep * 0.2 * (mu * dist_reg_term[narrow_band] + lmda * edge_term[narrow_band] + alfa * area_term[narrow_band])

        # Step 3: Update the narrow band by finding zero-crossing points and extending the band
        new_zero_crossings = find_zero_crossings(phi)
        new_indices = np.argwhere(new_zero_crossings > 0)

        # Update the narrow band by adding neighborhoods around new zero-crossing points
        new_narrow_band = np.zeros_like(phi, dtype=bool)
        for index in new_indices:
            i, j, k = index
            new_narrow_band[max(0, i-r):min(i+r+1, phi.shape[0]), 
                            max(0, j-r):min(j+r+1, phi.shape[1]), 
                            max(0, k-r):min(k+r+1, phi.shape[2])] = True

        # Step 4: Assign values to new pixels in the narrow band based on step 4
        newly_added_points = new_narrow_band & ~narrow_band
        phi[newly_added_points] = np.where(phi[newly_added_points] > 0, h, -h)

        # Step 5: Update narrow band for the next iteration
        narrow_band = new_narrow_band.copy()

        # Optional: Add termination condition based on zero-crossing changes
        dump_image_to_vtk(phi,f"threshold_innerloop_{drlse_edge.call_count}.vti")

    return phi

### GPU-BASED IMPLEMENTATION USING PYTORCH ###
def drlse_edge_gpu(phi_0, g, lmda, mu, alfa, epsilon, timestep, iters, potential_function, device="cuda"):
    """
    GPU-accelerated implementation of edge-based DRLSE using PyTorch.
    """
    phi = torch.tensor(phi_0, dtype=torch.float32, device=device).clone()
    g = torch.tensor(g, dtype=torch.float32, device=device)
    [vz, vy, vx] = torch.gradient(g)  # Compute gradients on the GPU

    for k in range(iters):
        phi = neumann_bound_cond_gpu(phi)  # Neumann boundary condition for 3D on GPU
        [phi_z, phi_y, phi_x] = torch.gradient(phi)  # 3D gradient on GPU

        s = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2)  # 3D norm of gradients
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        n_z = phi_z / (s + delta)
        curvature = div_gpu(n_x, n_y, n_z)  # 3D divergence on GPU

        if potential_function == SINGLE_WELL:
            dist_reg_term = laplace_gpu(phi) - curvature
        elif potential_function == DOUBLE_WELL:
            dist_reg_term = dist_reg_p2_gpu(phi)
        else:
            raise Exception('Error: Wrong choice of potential function.')

        dirac_phi = dirac_gpu(phi, epsilon)
        area_term = dirac_phi * g
        edge_term = dirac_phi * (vx * n_x + vy * n_y + vz * n_z) + dirac_phi * g * curvature
        phi += timestep * (mu * dist_reg_term + lmda * edge_term + alfa * area_term)

    return phi.cpu().numpy()  # Move result back to CPU for further processing if needed

def drlse_threshold_gpu(phi_0, img, lmda, mu, alfa, epsilon, upper, lower, timestep, iters, potential_function, device="cuda"):
    """
    GPU-accelerated implementation of threshold-based DRLSE using PyTorch.
    """
    phi = torch.tensor(phi_0, dtype=torch.float32, device=device).clone()
    img = torch.tensor(img, dtype=torch.float32, device=device)
    eps = 0.5 * (upper - lower)
    T = 0.5 * (upper + lower)

    for k in range(iters):
        phi = neumann_bound_cond_gpu(phi)
        [phi_z, phi_y, phi_x] = torch.gradient(phi)
        s = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2)
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        n_z = phi_z / (s + delta)
        curvature = div_gpu(n_x, n_y, n_z)

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
    Compute the Laplacian on GPU using a convolution kernel.
    """
    kernel = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], 
                            [[0, 1, 0], [1, -6, 1], [0, 1, 0]], 
                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], 
                           device=input.device, dtype=torch.float32)
    kernel = kernel / 6.0
    input = input.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    lap = torch.nn.functional.conv3d(input, kernel, padding=1)
    return lap.squeeze()

def dirac_gpu(x, sigma):
    """
    Compute the Dirac delta function on GPU.
    """
    f = (1 / (2 * sigma)) * (1 + torch.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b

def neumann_bound_cond_gpu(f):
    """
    Apply Neumann boundary conditions on GPU.
    """
    g = f.clone()
    g[0, :, :], g[-1, :, :], g[:, 0, :], g[:, -1, :], g[:, :, 0], g[:, :, -1] = \
        g[1, :, :], g[-2, :, :], g[:, 1, :], g[:, -2, :], g[:, :, 1], g[:, :, -2]
    return g

def div_gpu(nx, ny, nz):
    """
    Compute divergence on GPU.
    """
    nzz, _, _ = torch.gradient(nz)
    _, nyy, _ = torch.gradient(ny)
    _, _, nxx = torch.gradient(nx)
    return nxx + nyy + nzz

def dist_reg_p2_gpu(phi):
    """
    Compute the distance regularization term with double-well potential p2 on GPU.
    """
    [phi_z, phi_y, phi_x] = torch.gradient(phi)
    s = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2)
    
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * torch.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))

    return div_gpu(dps * phi_x - phi_x, dps * phi_y - phi_y, dps * phi_z - phi_z) + laplace_gpu(phi)