import numpy as np
import nibabel as nib
import vtk
from vtk.util import numpy_support

def dump_image_to_nii(image_data: np.ndarray, file_name: str, affine=None, voxel_spacing=(1, 1, 1)):
    """
    Dump 3D image data to a NIfTI (.nii) file, with optional control over voxel spacing.
    
    :param image_data: 3D NumPy array representing the image data.
    :param file_name: The file name to save the NIfTI file (e.g., 'output.nii' or 'output.nii.gz').
    :param affine: Affine transformation matrix (optional). If not provided, an identity matrix scaled by voxel_spacing is used.
    :param voxel_spacing: Tuple of three numbers representing the voxel size in x, y, and z dimensions.
    """
    # If no affine transformation is provided, create an identity matrix scaled by voxel_spacing
    if affine is None:
        affine = np.eye(4)
        affine[0, 0] = voxel_spacing[0]
        affine[1, 1] = voxel_spacing[1]
        affine[2, 2] = voxel_spacing[2]
    
    # Create a NIfTI image object from the image data and affine matrix
    nifti_image = nib.Nifti1Image(image_data, affine)
    
    # Save the NIfTI image to the specified file
    nib.save(nifti_image, file_name)
    print(f"NIfTI image saved to {file_name}")

def dump_image_to_vtk(phi, filename,voxel_spacing=(1, 1, 1)):
    # Get the dimensions of the 3D array (phi)
    dims = phi.shape

    # Convert the numpy array (phi) to a VTK array
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=phi.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

    # Create a VTK ImageData object
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.SetSpacing(voxel_spacing[0],voxel_spacing[1],voxel_spacing[2])  # Adjust spacing as needed
    image_data.SetOrigin(0.0, 0.0, 0.0)   # Adjust origin as needed
    image_data.GetPointData().SetScalars(vtk_data_array)

    # Write the VTK ImageData to a file (.vti format)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()

    print(f"Dumped 3D phi data to '{filename}'.")