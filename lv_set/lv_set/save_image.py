import numpy as np
import nibabel as nib
import vtk
from vtk.util import numpy_support

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


def dump_image_to_vtk(phi, filename):
    # Get the dimensions of the 3D array (phi)
    dims = phi.shape

    # Convert the numpy array (phi) to a VTK array
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=phi.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    # Create a VTK ImageData object
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.SetSpacing(1.0, 1.0, 1.0)  # Adjust spacing as needed
    image_data.SetOrigin(0.0, 0.0, 0.0)   # Adjust origin as needed
    image_data.GetPointData().SetScalars(vtk_data_array)

    # Write the VTK ImageData to a file (.vti format)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()

    print(f"Dumped 3D phi data to '{filename}'.")