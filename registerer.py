import ndreg
import csv
import numpy as np
import SimpleITK as sitk
import skimage

def register_affine(spacing, iterations=2000.0, resolutions=8.0, use_mi=False, fixed_mask=None, moving_mask=None):
    # reorient the target to match the source
    target_reoriented = ndreg.imgReorient(target, targetOrient, sourceOrient)
    source_ds1 = sitk.Clamp(ndreg.imgResample(source, [spacing]*3), upperBound=ndreg.imgPercentile(source, 0.99))
    target_ds1 = sitk.Clamp(ndreg.imgResample(target_reoriented, [spacing]*3),
                           upperBound=ndreg.imgPercentile(target_reoriented, 0.99))

    # normalize
    target_ds = _normalize_image(target_ds1)
    source_ds = _normalize_image(source_ds1)

    movingImage = source_ds
    fixedImage = target_ds

    # set parameters
    affineParameterMap = sitk.GetDefaultParameterMap('affine')
#         affineParameterMap['MaximumNumberOfSamplingAttempts'] = '0'
#         affineParameterMap['Metric'] = ['AdvancedMeanSquares']
    affineParameterMap['MaximumNumberOfIterations'] = ['{}'.format(iterations)]
    affineParameterMap['Optimizer'] = ['StandardGradientDescent']
    affineParameterMap['NumberOfResolutions'] = '{}'.format(resolutions)
#         affineParameterMap['ShowExactMetricValue'] = ['true']
#         affineParameterMap['AutomaticTransformInitialization'] = ['true']
    if not use_mi:
        affineParameterMap['Metric'] = ['AdvancedMeanSquares']
#             movingImage = source_ds1
#             fixedImage = target_ds1

    # initialize registration object
    elastixImageFilter = sitk.ElastixImageFilter()
    # set source and target images
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    # set masks
    if fixed_mask is not None: 
        elastixImageFilter.SetFixedMask(fixed_mask)
    if moving_mask is not None:
        elastixImageFilter.SetMovingMask(moving_mask)
    # set parameter map
    elastixImageFilter.SetParameterMap(affineParameterMap)
    # run the registration
    elastixImageFilter.Execute()
    elastix_img_filt = elastixImageFilter
    # get the affine transformed source image
    source_affine = elastixImageFilter.GetResultImage()
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0] 
    # save the affine transformation
    affine = [float(i) for i in transformParameterMap['TransformParameters']]
    return source_affine

def register_lddmm(affine_img, target_img, alpha_list=0.05, scale_list=[0.0625, 0.125, 0.25], 
                   epsilon_list=1e-7, sigma=None, use_mi=False, iterations=200, inMask=None, verbose=True, out_dir=''):
    if sigma == None:
        sigma = (0.1/target_img.GetNumberOfPixels())

    (field, invField) = ndreg.imgMetamorphosisComposite(affine_img, target_img,
                                                        alphaList=alpha_list,
                                                        scaleList=scale_list,
                                                        epsilonList=epsilon_list,
                                                        sigma=sigma,
                                                        useMI=use_mi,
                                                        inMask=inMask,
                                                        iterations=iterations, 
                                                        verbose=verbose,
                                                        outDirPath=out_dir)
    affineField = ndreg.affineToField(affine, field.GetSize(), field.GetSpacing())
    fieldComposite = ndreg.fieldApplyField(field, affineField)

    invAffineField = ndreg.affineToField(ndreg.affineInverse(affine), invField.GetSize(),
                                         invField.GetSpacing())
    invFieldComposite = ndreg.fieldApplyField(invAffineField, invField)

    source_lddmm = ndreg.imgApplyField(affine_img, field, 
                                            size=target_img.GetSize(), 
                                            spacing=target_img.GetSpacing())
    return source_lddmm

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
                                                           sourceOrient, source)
    landmarks_source_a = _apply_affine(landmarks_source_r)
    landmarks_source_ar = _reorient_landmarks(landmarks_source_a, sourceOrient, 
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
                                                  sourceOrient, source)
    landmarks_source_a = _apply_affine(landmarks_source_r)
    landmarks_target_r = _reorient_landmarks(landmarks_target, orientation_target_fid,
                                                  sourceOrient, target)
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