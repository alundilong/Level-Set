import pyvista as pv
import numpy as np

def visualize_3d_image_and_phi(img, phi, contour_value=0.0):
    """
    Visualize 3D image and level set function using PyVista.
    
    :param img: 3D grayscale image (e.g., medical image)
    :param phi: 3D level set function (segmentation result)
    :param contour_value: Value for extracting the contour (e.g., 0 for the zero level set)
    """
    # Normalize the image for visualization (scaling between 0 and 255)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

    # Create a PyVista UniformGrid for the 3D image data
    grid = pv.ImageData()
    grid.dimensions = img.shape  # Shape of the 3D image
    grid.origin = (0, 0, 0)  # Origin of the dataset
    grid.spacing = (1, 1, 1)  # Grid spacing, can adjust for real-world units
    grid.point_data["values"] = img.flatten(order="F")  # Add the image data to the grid

    # Create a PyVista plotter for 3D visualization
    plotter = pv.Plotter()

    # Add the image volume to the plotter
    plotter.add_volume(grid, cmap="gray", opacity="sigmoid", opacity_unit_distance=5)

    # Add the contour of the level set function (phi) to visualize the segmentation boundary
    contour = grid.contour(isosurfaces=[contour_value], scalars=phi.flatten(order="F"))
    plotter.add_mesh(contour, color="red", label="Level Set Contour")

    # Add labels, axes, and display the scene
    plotter.add_axes()
    plotter.add_legend()
    plotter.show()
