import ndreg
import csv
import numpy as np
import SimpleITK as sitk
import skimage
import matplotlib
from matplotlib import pyplot as plt

def register_affine(atlas, img, learning_rate=1e-2, iters=200, min_step=1e-10, shrink_factors=[1],
            sigmas=[.150], use_mi=False, grad_tol=1e-6, verbose=False):
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
#     registration_method.SetMetricAsMeanSquares()
    if use_mi: registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=128)
    else: registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkBSpline)

    # Optimizer settings.
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=learning_rate,
                                                                 minStep=min_step,
    #                                                              estimateLearningRate=registration_method.EachIteration,
                                                                 gradientMagnitudeTolerance=grad_tol,
                                                                 numberOfIterations=iters)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # initial transform
    initial_transform = sitk.AffineTransform(atlas.GetDimension())
    length = np.array(atlas.GetSize())*np.array(atlas.GetSpacing())
    initial_transform.SetCenter(length/2.0)

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform)

    # Connect all of the observers so that we can perform plotting during registration.
    if verbose:
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(img, sitk.sitkFloat32),
                                                  sitk.Cast(atlas, sitk.sitkFloat32))
    return final_transform

def register_lddmm(affine_img, target_img, alpha_list=0.02, scale_list=[0.5, 1.0], 
                   epsilon_list=1e-4, min_epsilon_list=1e-10, sigma=None, use_mi=False, iterations=200, inMask=None,
                   verbose=True, out_dir=''):
    if sigma == None:
        sigma = (0.1/target_img.GetNumberOfPixels())

    (field, invField) = ndreg.imgMetamorphosisComposite(affine_img, target_img,
                                                        alphaList=alpha_list,
                                                        scaleList=scale_list,
                                                        epsilonList=epsilon_list,
                                                        minEpsilonList=min_epsilon_list,
                                                        sigma=sigma,
                                                        useMI=use_mi,
                                                        inMask=inMask,
                                                        iterations=iterations, 
                                                        verbose=verbose,
                                                        outDirPath=out_dir)
#     affineField = ndreg.affineToField(affine, field.GetSize(), field.GetSpacing())
#     fieldComposite = ndreg.fieldApplyField(field, affineField)

#     invAffineField = ndreg.affineToField(ndreg.affineInverse(affine), invField.GetSize(),
#                                          invField.GetSpacing())
#     invFieldComposite = ndreg.fieldApplyField(invAffineField, invField)

    source_lddmm = ndreg.imgApplyField(affine_img, field, 
                                            size=target_img.GetSize(), 
                                            spacing=target_img.GetSpacing())
    return source_lddmm, field, invField

def checkerboard_image(vmax=None):
    if vmax == None:
        vmax = ndreg.imgPercentile(target_ds, 0.99)
    if source_affine is None:
        raise Exception("Perform a registration first. This can be just affine or affine + LDDMM")
    elif source_lddmm is None:
        ndreg.imgShow(ndreg.imgChecker(_normalize_image(target_ds), 
                                       _normalize_image(source_affine)),
                      vmax=1.0)
    else:
        ndreg.imgShow(ndreg.imgChecker(_normalize_image(target_ds), 
                                       _normalize_image(source_lddmm)), 
                      vmax=1.0)

def evaluate_affine_registration(source_fiducial_file, target_fiducial_file, scale_source, scale_target,
                                 orientation_source_fid, orientation_target_fid):
    # load the landmarks for each brain
    landmarks_source = _parse_fiducial_file(source_fiducial_file, scale_source)
    landmarks_target = _parse_fiducial_file(target_fiducial_file, scale_target)
    # reorient the source fiducials so they match the orientation of
    # source we calculated transformation on. then we can apply our transformations.
    landmarks_source_r = _reorient_landmarks(landmarks_source, orientation_source_fid, 
                                                           source_orientation, source)
    landmarks_source_a = _apply_affine(landmarks_source_r)
    landmarks_source_ar = _reorient_landmarks(landmarks_source_a, source_orientation, 
                                                   orientation_target_fid, target)
    mse = _compute_error(landmarks_source_ar, landmarks_target)
    return mse

def evaluate_lddmm_registration(source_fiducial_file, target_fiducial_file, scale_source, scale_target,
                               orientation_source_fid, orientation_target_fid):
    if fieldComposite is None:
        raise Exception("Perform LDDMM registration first")
    landmarks_source = _parse_fiducial_file(source_fiducial_file, scale_source)
    landmarks_target = _parse_fiducial_file(target_fiducial_file, scale_target)
    landmarks_source_r = _reorient_landmarks(landmarks_source, orientation_source_fid, 
                                                  source_orientation, source)
    landmarks_source_a = _apply_affine(landmarks_source_r)
    landmarks_target_r = _reorient_landmarks(landmarks_target, orientation_target_fid,
                                                  source_orientation, target)
    landmarks_target_lddmm = _lmk_apply_field(landmarks_target_r, field)
    mse = _compute_error(np.array(landmarks_target_lddmm), np.array(landmarks_source_a))
    return mse

#     def create_channel_resource( rmt, chan_name, coll_name, exp_name, type='image', base_resolution=0, sources=[], 
#                                 datatype='uint16', new_channel=True):
#         channel_resource = ChannelResource(chan_name, coll_name, exp_name, type=type, base_resolution=base_resolution,
#                                            sources=sources, datatype=datatype)
#         if new_channel: 
#             new_rsc = rmt.create_project(channel_resource)
#             return new_rsc

#         return channel_resource

#     def upload_to_boss( rmt, data, channel_resource, resolution=0):
#         Z_LOC = 0
#         size = data.shape
#         for i in range(0, data.shape[Z_LOC], 16):
#             last_z = i+16
#             if last_z > data.shape[Z_LOC]:
#                 last_z = data.shape[Z_LOC]
#             print(resolution, [0, size[2]], [0, size[1]], [i, last_z])
#             rmt.create_cutout(channel_resource, resolution, [0, size[2]], [0, size[1]], [i, last_z],
#                               np.asarray(data[i:last_z,:,:], order='C'))

#     def download_ara( rmt, resolution, type='average'):
#         if resolution not in [10, 25, 50, 100]:
#             print('Please provide a resolution that is among the following: 10, 25, 50, 100')
#             return
#         REFERENCE_COLLECTION = 'ara_2016'
#         REFERENCE_EXPERIMENT = 'sagittal_{}um'.format(resolution)
#         REFERENCE_COORDINATE_FRAME = 'ara_2016_{}um'.format(resolution) 
#         REFERENCE_CHANNEL = '{}_{}um'.format(type, resolution)

#         refImg = download_image(rmt, REFERENCE_COLLECTION, REFERENCE_EXPERIMENT, REFERENCE_CHANNEL, ara_res=resolution)

#         return refImg

#     def download_image( rmt, collection, experiment, channel, res=0, isotropic=True, ara_res=None):
#         (exp_resource, coord_resource, channel_resource) = setup_channel_boss(rmt, collection, experiment, channel)
#         img = ndreg.imgDownload_boss(rmt, channel_resource, coord_resource, resolution=res, isotropic=isotropic)
#         return img

def reorient_landmarks(landmarks, in_orient, out_orient):
    """
    Takes in centered landmarks and orients them correctly
    """
    orient = {'l': 'lr', 'a': 'ap', 's': 'si','r': 'rl', 'p':'pa', 'i': 'is'}
    order_in = []
    order_out = []
    for i in range(len(in_orient)):
        # create strings representing input and output order
        order_in.append(orient['{}'.format(in_orient[i].lower())])
        order_out.append(orient['{}'.format(out_orient[i].lower())])
    locs = []
    swap = []
    reorient_mat = np.zeros((3,3))
    for i in range(len(order_in)):
        try:
            # find the axis of the input
            # that matches the output for the ith axis
            idx = order_in.index(order_out[i])
            reorient_mat[i,idx] = 1.0
        # if you can't find a match, check for
        # the reverse orientation on the ith axis
        except Exception as e:
            idx = order_in.index(order_out[i][::-1])
            reorient_mat[i,idx] = -1.0
    landmarks_reoriented = np.dot(reorient_mat, np.array(landmarks).T)
    return landmarks_reoriented.T

#     def _flip_fiducials_along_axis( fiducials, center, axis=2):
#         if type(fiducials) == list: fiducials_new = np.array(fiducials).copy()
#         else: fiducials_new = fiducials.copy() 
#         fiducials_new[:,axis] = (2.0*center[axis]) - fiducials_new[:,axis] 
#         return fiducials_new

def compute_error(ref_landmarks, img_landmarks):
    if ref_landmarks.shape[0] != img_landmarks.shape[0]:
        raise Exception("pass in the same number of fiducials")
    distances = np.zeros([len(ref_landmarks), 1])
    for i in range(len(ref_landmarks)):
#             print(ref_landmarks[i,:] - img_landmarks[i,:])
        distances[i] = np.linalg.norm(ref_landmarks[i,:] - img_landmarks[i,:])
    return distances

def parse_fiducial_file(f, scale):
    reader = csv.reader(file(f))
    # skip first line, dont need it
    reader.next()
    # get the coordinate system
    coordinateSystem = reader.next()[0].split()[3]
    columns = reader.next()[1:]
    landmarks = []
    for i in reader:
        landmarks.append(np.array(i[1:4]).astype(np.float) * scale)
    return np.array(landmarks)

def _lmk_apply_field(fids, field):
    dim = 3
    # Create transform
    field_copy = sitk.GetImageFromArray(sitk.GetArrayFromImage(field))
    field_copy.CopyInformation(field)
    transform = sitk.DisplacementFieldTransform(dim)
    transform.SetInterpolator(sitk.sitkLinear)
    transform.SetDisplacementField(sitk.Cast(field_copy, sitk.sitkVectorFloat64))
    # test inv vs not inv
    t = np.array([transform.TransformPoint(i) for i in fids])
    return t

def _apply_affine(fids):
    p = elastix_img_filt.GetTransformParameterMap()[0]
    center = p['CenterOfRotationPoint']
    center_f = np.array([float(i) for i in center])
    at = sitk.AffineTransform(3)
    at.SetCenter(center_f)
    at.SetMatrix(affine[:9])
    at.SetTranslation(affine[9:])
    # testing inverse
#         a_t = np.array([at.TransformPoint(i) for i in fids])
    at.SetInverse()
    a_tinv = np.array([at.TransformPoint(i) for i in fids])
    return a_tinv


def _normalize_image(img, low_bound=None, up_bound=0.999):
    min_val = 0.0
    if low_bound is not None:
        min_val = ndreg.imgPercentile(img, low_bound)
    max_val = ndreg.imgPercentile(img, up_bound)

    return (img - min_val)/(max_val - min_val)
    

# Utility functions for plotting

# import matplotlib.pyplot as plt
# %matplotlib inline

from ipywidgets import interact, fixed
from IPython.display import clear_output

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    
    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
def display_slices_with_alpha(fixed, moving, alpha, vmax):
    img = (1.0 - alpha)*fixed + alpha*moving
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r, vmax=vmax);
    plt.axis('off')
    plt.show()
def display_images_with_alpha(slice_num, alpha, fixed, moving, axis=2, vmax=1000):
    if axis == 0: display_slices_with_alpha(fixed[slice_num,:,:], moving[slice_num,:,:], alpha, vmax)
    elif axis == 1: display_slices_with_alpha(fixed[:,slice_num,:], moving[:,slice_num,:], alpha, vmax)
    else: display_slices_with_alpha(fixed[:,:,slice_num], moving[:,:,slice_num], alpha, vmax)
    
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))  
    
def resample(image, transform, ref_img, default_value=0.0):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = ref_img
    interpolator = sitk.sitkBSpline
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)
def view_overlay(img1, img2, axis=2, vmax=1000):
    interact(display_images_with_alpha, slice_num=(0,img1.GetSize()[axis]), alpha=(0.0,1.0,0.05), 
             fixed = fixed(img2), moving=fixed(img1), axis=(0,2,1), vmax=(1,vmax));
