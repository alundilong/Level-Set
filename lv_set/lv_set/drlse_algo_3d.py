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
        dump_image_to_vtk(phi,f"innerloop_{drlse_edge.call_count}.vti")
        dump_image_to_vtk(area_term,f"area_term_{drlse_edge.call_count}.vti")
        dump_image_to_vtk(edge_term,f"edge_term_{drlse_edge.call_count}.vti")
        dump_image_to_vtk(dist_reg_term,f"dist_reg_term_{drlse_edge.call_count}.vti")
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
    phi = phi_0.copy()
    eps = 0.5 * (upper - lower)
    T = 0.5 * (upper + lower)

    for k in range(iters):
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