import ndreg
import SimpleITK as sitk
import matplotlib
import numpy as np
import skimage
from skimage import filters, morphology
import sklearn
from sklearn.mixture.gaussian_mixture import GaussianMixture
import scipy.stats as st

class Preprocessor:

    def __init__(self, img):
        if type(img) == SimpleITK.SimpleITK.Image:
            self.img_sitk = img
	    self.img_np = sitk.GetArrayFromImage(img)
        else: 
            raise Exception("Please convert your image into a SimpleITK image")

    def remove_streaks(self):
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(self.img_np, axes=(-2, -1)), axes=(-2, -1)))**2
        fft_mag = np.mean(fft_mag, axis=0)
        center = [math.ceil((fft_mag_norm.shape[1]+1)/2), math.ceil((fft_mag_norm.shape[0]+1)/2)]
        fx, fy = np.meshgrid(np.arange(fft_mag_norm.shape[1]), np.arange(fft_mag_norm.shape[0]))
        fx = fx - center[0]
        fy = fy - center[1]
        r = np.sqrt(fx**2 + fy**2)
        theta = np.arctan2(fy, fx)
        
        streak1 = -90.0 * (math.pi/180.0) # radians
        streak2 = 100.0 * (math.pi/180.0) # radians
        streak3 = 80.0 * (math.pi/180.0) # radians
        streak4 = 90.0 * (math.pi/180.0) # radians
        # streak4_pos = 180.0 * (math.pi/180.0) # radians
        # streak4_neg = -180.0 * (math.pi/180.0) # radians 
        streak5 = -80.0 * (math.pi/180.0) # radians
        streak6 = -100.0 * (math.pi/180.0) # radians
        delta = 2 * (math.pi/180.0) # radians
        # streak_angles = [streak1, streak2, streak3]
        radius = 40 # pixels
        small_num = 0.0

        streak_filter = np.ones(fft_mag_norm.shape)
        # set streak regoins to small number
        streak_filter[(theta < streak1 + delta) & (theta > streak1 - delta)] = small_num
        streak_filter[(theta < streak2 + delta) & (theta > streak2 - delta)] = small_num
        streak_filter[(theta < streak3 + delta) & (theta > streak3 - delta)] = small_num
        streak_filter[(theta < streak4 + delta) & (theta > streak4 - delta)] = small_num
        # streak_filter[(theta < streak4_neg + delta) | (theta > streak4_pos - delta)] = small_num
        streak_filter[(theta < streak5 + delta) & (theta > streak5 - delta)] = small_num
        streak_filter[(theta < streak6 + delta) & (theta > streak6 - delta)] = small_num
        streak_filter[ np.where(r < radius) ] = 1.0
        streak_filter_blurred = skimage.filters.gaussian(streak_filter, sigma=3)

        fft_img = np.fft.fftshift(np.fft.fft2(img, axes=(-2, -1)), axes=(-2,-1))
        fft_filtered = np.ones(fft_img.shape, dtype=fft_img.dtype)
        for i in range(fft_img.shape[0]):
            fft_filtered[i,:,:] = np.multiply(fft_img[i,:,:], streak_filter_blurred)

        filtered_fft = np.fft.ifftshift(fft_filtered, axes=(-2,-1))
        img_streak_free = np.fft.ifft2(filtered_fft, axes=(-2,-1))
        return img_streak_free

    def create_mask(self):
        gmix = mixture.gaussian_mixture.GaussianMixture(n_components=3, covariance_type='full', init_params='kmeans', verbose=0)
        gmix.fit(self.img_np.ravel().reshape(-1, 1))
        means = gmix.means_
        covariances = gmix.covariances_
        mean_background = means.min()
        covariance_background = covariances[np.where( means == mean_background ) ][0][0]
        z_score = st.norm.ppf(0.75)
        threshold = z_score * np.sqrt(covariance_background) + mean_background
        test_mask = (inImg_orig_np > threshold)
        connected_comp = skimage.measure.label(inMask)
        out = skimage.measure.regionprops(connected_comp)
        area = 0.0
        idx = 0
        for i in range(out):
            if out[i].area > area:
                area = out[i].area
                idx = i
        connected_comp[ connected_comp != idx ] = 0 
        return connected_comp
        
    def remove_circle(self):
        radius = 170 # px
	origin_shift = [0, 0]
	center = [inImg_orig_np.shape[-1]/2 + origin_shift[0], inImg_orig_np.shape[-2]/2 + origin_shift[1]]
	x, y, z = inImg_orig_np.shape
	mask = np.zeros((y,z))
	for i in range(y):
	    for j in range(z):
		mask[i,j] = math.sqrt((i-center[1])**2 + (j-center[0])**2) 
	mask = (mask < radius).astype('uint64')
	new_mask = np.repeat(mask[None,:,:], inImg_orig_np.shape[0], axis=0)
	img_masked = np.multiply(self.img_np, new_mask)
	return img_masked

    def correct_bias_field(self, mask=None, scale=0.4, numBins=128):
	"""
	Bias corrects an image using the N4 algorithm
	"""
	spacing = np.array(self.img_sitk.GetSpacing())/scale
	img_ds = ndreg.imgResample(self.img_sitk, spacing=spacing)

	# Calculate bias
	if mask is None:
	    mask = sitk.Image(self.img_sitk.GetSize(), sitk.sitkUInt8)+1
	    mask.CopyInformation(self.img_sitk)
	else:
	    mask = imgResample(mask, spacing=spacing)

	img_ds_bc = sitk.N4BiasFieldCorrection(sitk.Cast(img_ds, sitk.sitkFloat32), mask, numberOfHistogramBins=numBins)
	bias_ds = img_ds_bc - sitk.Cast(img_ds,img_ds_bc.GetPixelID())
	
	# Upsample bias
	bias = imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())

	# Apply bias to original image and threshold to eliminate negitive values
	upper = np.iinfo(sitkToNpDataTypes[img.GetPixelID()]).max
	img_bc = sitk.Threshold(img + sitk.Cast(bias, img.GetPixelID()),
				lower=0, upper=upper)
	return img_bc 
