import ndreg
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import filters, morphology
import sklearn
from sklearn.mixture.gaussian_mixture import GaussianMixture
import scipy.stats as st
import math

sitkToNpDataTypes = {sitk.sitkUInt8: np.uint8,
                     sitk.sitkUInt16: np.uint16,
                     sitk.sitkUInt32: np.uint32,
                     sitk.sitkInt8: np.int8,
                     sitk.sitkInt16: np.int16,
                     sitk.sitkInt32: np.int32,
                     sitk.sitkFloat32: np.float32,
                     sitk.sitkFloat64: np.float64,
                     }

def remove_streaks(img, radius=40):
    if type(img) is sitk.SimpleITK.Image:
        img = sitk.GetArrayFromImage(img)

    fft_img = np.fft.fftshift(np.fft.fft2(img, axes=(-2, -1)), axes=(-2, -1))
    center = [math.ceil((fft_img.shape[2]+1)/2), math.ceil((fft_img.shape[1]+1)/2)]
    fx, fy = np.meshgrid(np.arange(fft_img.shape[2]), np.arange(fft_img.shape[1]))
    fx = fx - center[0]
    fy = fy - center[1]
    r = np.sqrt(fx**2 + fy**2)
    theta = np.arctan2(fy, fx)

    # for debugging
#         plt.imshow(np.mean(fft_img, axis=0).astype('float'), vmax=100)
#         plt.colorbar()
#         plt.show()

    streak1 = -90.0 * (math.pi/180.0) # radians
    streak2 = 100.0 * (math.pi/180.0) # radians
    streak3 = 80.0 * (math.pi/180.0) # radians
    streak4 = 90.0 * (math.pi/180.0) # radians
    streak5 = -80.0 * (math.pi/180.0) # radians
    streak6 = -100.0 * (math.pi/180.0) # radians
    delta = 2 * (math.pi/180.0) # radians
    small_num = 0.0

    streak_filter = np.ones(fft_img.shape[1:])

    # set streak regoins to small number
    streak_filter[(theta < streak1 + delta) & (theta > streak1 - delta)] = small_num
    streak_filter[(theta < streak2 + delta) & (theta > streak2 - delta)] = small_num
    streak_filter[(theta < streak3 + delta) & (theta > streak3 - delta)] = small_num
    streak_filter[(theta < streak4 + delta) & (theta > streak4 - delta)] = small_num
    streak_filter[(theta < streak5 + delta) & (theta > streak5 - delta)] = small_num
    streak_filter[(theta < streak6 + delta) & (theta > streak6 - delta)] = small_num
    streak_filter[ np.where(r < radius) ] = 1.0
    streak_filter_blurred = skimage.filters.gaussian(streak_filter, sigma=3)

    fft_filtered = np.ones(fft_img.shape, dtype=fft_img.dtype)
    for i in range(fft_img.shape[0]):
        fft_filtered[i,:,:] = np.multiply(fft_img[i,:,:], streak_filter_blurred)

    filtered_fft = np.fft.ifftshift(fft_filtered, axes=(-2,-1))
    img_streak_free = np.fft.ifft2(filtered_fft, axes=(-2,-1))
#         img_streak_free_sitk = sitk.GetImageFromArray(img_streak_free)
#         img_streak_free_sitk.SetSpacing(img_sitk.GetSpacing())
#         img_streak_free_sitk.SetDirection(img_sitk.GetDirection())
    return img_streak_free

def create_mask(img, background_probability=0.75, use_triangle=False, use_otsu=False):
    test_mask = None
    if use_triangle:
        test_mask = sitk.GetArrayFromImage(sitk.TriangleThreshold(img, 0, 1))
    elif use_otsu:
        test_mask = sitk.GetArrayFromImage(sitk.OtsuThreshold(img, 0, 1))
    else:
        if type(img) is sitk.SimpleITK.Image:
            img = sitk.GetArrayFromImage(img)
        gmix = GaussianMixture(n_components=3, covariance_type='full', init_params='kmeans', verbose=0)
        gmix.fit(img.ravel().reshape(-1, 1))
        covariances = gmix.covariances_
        mean_background = gmix.means_.min()
        covariance_background = covariances[np.where( gmix.means_ == mean_background ) ][0][0]
        z_score = st.norm.ppf(background_probability)
        threshold = z_score * np.sqrt(covariance_background) + mean_background
        test_mask = (img > threshold)
    eroded_im = morphology.opening(test_mask, selem=morphology.ball(2))
    connected_comp = skimage.measure.label(eroded_im)
    out = skimage.measure.regionprops(connected_comp)
    area_max = 0.0
    idx_max = 0
    for i in range(len(out)):
        if out[i].area > area_max:
            area_max = out[i].area
            idx_max = i+1
    connected_comp[ connected_comp != idx_max ] = 0
    mask = connected_comp
    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.CopyInformation(img)
    return mask_sitk

def remove_circle(img, radius=170):
    origin_shift = [0, 0]
    center = [img.shape[-1]/2 + origin_shift[0], img.shape[-2]/2 + origin_shift[1]]
    x, y, z = img.shape
    mask = np.zeros((y,z))
    for i in range(y):
        for j in range(z):
            mask[i,j] = math.sqrt((i-center[1])**2 + (j-center[0])**2) 
    mask = (mask < radius).astype('uint64')
    new_mask = np.repeat(mask[None,:,:], img.shape[0], axis=0)
    img_masked = np.multiply(img, new_mask)
    return img_masked

#     def auto_preprocess(:
#         if mask is None: create_mask()
#         masked_image = img_np.copy()
#         masked_image[np.where(mask == 0)] = 0
#         out_img = sitk.GetImageFromArray(masked_image)
#         out_img.CopyInformation(img_sitk)
#         return correct_bias_field(img=out_img)


def correct_bias_field(img, mask=None, scale=0.2, numBins=128, spline_order=4, niters=[50, 50, 50, 50],
                      num_control_pts=[5, 5, 5], fwhm=0.150):
    """
    Bias corrects an image using the N4 algorithm
    """
#     if type(img) is not sitk.SimpleITK.Image:
#         raise("Input image needs to be of type SimpleITK.SimpleITK.Image")
    
    spacing = np.array(img.GetSpacing())/scale
    img_ds = ndreg.imgResample(img, spacing=spacing)

    # Calculate bias
    if mask is None:
        mask = sitk.Image(img_ds.GetSize(), sitk.sitkUInt8)+1
        mask.CopyInformation(img_ds)
    else:
        if type(mask) is not sitk.SimpleITK.Image:
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(img)
            mask = mask_sitk
        mask = ndreg.imgResample(mask, spacing=spacing)

    img_ds_bc = sitk.N4BiasFieldCorrection(sitk.Cast(img_ds, sitk.sitkFloat32), mask, 0.001, 
                                           niters, splineOrder=spline_order, 
                                           numberOfControlPoints=num_control_pts,
                                           biasFieldFullWidthAtHalfMaximum=fwhm)
    bias_ds = img_ds_bc - sitk.Cast(img_ds, img_ds_bc.GetPixelID())

    # Upsample bias
    bias = ndreg.imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())

    # Apply bias to original image and threshold to eliminate negitive values
    upper = np.iinfo(sitkToNpDataTypes[img.GetPixelID()]).max
    img_bc = sitk.Threshold(img + sitk.Cast(bias, img.GetPixelID()),
                lower=0, upper=upper)
    return img_bc 

# TODO: finish this method
def create_iterative_mask(img):
#     img_bc_np = sitk.GetArrayFromImage(img_bias_corrected)
#     img_in = img
#     img_bc_tmp = img_bias_corrected
    for i in range(7):
        out_mask = create_mask(img, use_triangle=True)
        ndreg.imgShow(sitk.GetImageFromArray(out_mask))
        
        img_bc_tmp = preprocessor.correct_bias_field(img, mask=out_mask, 
                                                     scale=0.25, spline_order=3, niters=[25, 25, 25, 25])
        img_bc_np = sitk.GetArrayFromImage(img_bc_tmp)
        plt.imshow(img_bc_np[80,:,:], vmax=10000)
        plt.colorbar()
        plt.show()
    
    mask_sitk = sitk.GetImageFromArray(out_mask)
    mask_sitk.CopyInformation(img)
    return 

# utility functions

def downsample_and_reorient(atlas, target, atlas_orient, target_orient, spacing, size=[], set_origin=True):
    """
    make sure img1 is the source and img2 is the target.
    iamges will be resampled to match the coordinate system of img2.
    """
    target_r = ndreg.imgReorient(target, target_orient, atlas_orient)
    size_atlas = atlas.GetSize()
    size_target = target_r.GetSize()
    dims_atlas = np.array(size_atlas)*np.array(atlas.GetSpacing())
    dims_target = np.array(size_target)*np.array(target_r.GetSpacing())
#     print(dims_target)
    max_size_per_dim = [max(dims_atlas[i], dims_target[i]) for i in range(len(dims_atlas))]
    print(max_size_per_dim)
    vox_sizes = [int(i) for i in (np.array(max_size_per_dim) / spacing)]
#     print(vox_sizes)
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetSize(vox_sizes)
    resampler.SetOutputSpacing([spacing]*3)
#     resampler.SetOutputOrigin((np.array(max_size_per_dim)/2.0).tolist())

    out_origin = (np.array(max_size_per_dim)/2.0).tolist()
    out_target = resampler.Execute(target_r)
#     out_target.SetOrigin(out_origin)
    out_atlas = resampler.Execute(atlas)
#     out_atlas.SetOrigin(out_origin)
    
    assert(out_target.GetOrigin() == out_atlas.GetOrigin())
    assert(out_target.GetSize() == out_atlas.GetSize())
    assert(out_target.GetSpacing() == out_atlas.GetSpacing())
    return out_atlas, out_target

def normalize(img):
    max_val = ndreg.imgPercentile(img, 0.999)
    return sitk.Clamp(img, upperBound=max_val) / max_val
    