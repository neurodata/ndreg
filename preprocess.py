import matplotlib
matplotlib.use('Agg')

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import ndreg
from ndreg import preprocessor, registerer, util
import SimpleITK as sitk
import numpy as np

import time



def normalize(img, percentile=0.99):
    #Accept ndarray images or sitk images
    if type(img) is np.ndarray:
        sitk_img = sitk.GetImageFromArray(img)
    else:
        sitk_img = img
    max_val = ndreg.imgPercentile(sitk_img, percentile)
    return sitk.Clamp(sitk_img, upperBound=max_val) / max_val

def process(img):
    """
    Input: SITK Image
    Output: Clipped and normalized SITK Image
    """
    temp = np.clip(sitk.GetArrayFromImage(img), -3000, 12000)
    return normalize(temp)

def overlay(img1, img2, title, save=None):
    fig = plt.figure(figsize=(5, 7))
    plt.imshow(sitk.GetArrayViewFromImage(img1), cmap='Purples', alpha=0.5)
    plt.imshow(sitk.GetArrayViewFromImage(img2), cmap='Greens', alpha=0.5)
    plt.axis('off')
    plt.title(title)
    if save is not None:
        plt.savefig(save)
#     plt.show()
    plt.close()

print("Reading data")
tp1 = util.imgRead('./data/R04_tp1.tif')
tp2 = util.imgRead('./data/R04_tp2.tif')
tp3 = util.imgRead('./data/R04_tp3.tif')
tp4 = util.imgRead('./data/R04_tp4.tif')

tp1_slices = [tp1[:,:,i] for i in range(50)]
tp2_slices = [tp2[:,:,i] for i in range(50)]
tp3_slices = [tp3[:,:,i] for i in range(50)]
tp4_slices = [tp4[:,:,i] for i in range(50)]

print("Processing images")
tp1_processed = sitk.GetImageFromArray([sitk.GetArrayFromImage(process(tp1_slices[i])) for i in range(50)])
tp2_processed = sitk.GetImageFromArray([sitk.GetArrayFromImage(process(tp2_slices[i])) for i in range(50)])
tp3_processed = sitk.GetImageFromArray([sitk.GetArrayFromImage(process(tp3_slices[i])) for i in range(50)])
tp4_processed = sitk.GetImageFromArray([sitk.GetArrayFromImage(process(tp4_slices[i])) for i in range(50)])

tp1_processed1 = [sitk.GetArrayFromImage(process(tp1_slices[i])) for i in range(50)]
tp2_processed1 = [sitk.GetArrayFromImage(process(tp2_slices[i])) for i in range(50)]
tp3_processed1 = [sitk.GetArrayFromImage(process(tp3_slices[i])) for i in range(50)]
tp4_processed1 = [sitk.GetArrayFromImage(process(tp4_slices[i])) for i in range(50)]


final_transform_12 = registerer.register_rigid(tp1_processed, tp2_processed, learning_rate=1e-1, iters=25)
print("Computing corrected image (of timepoint 2)")
corrected_img_12 = registerer.resample(tp1_processed, final_transform_12, tp2_processed)

errors_12 = []
for i in range(50):
    error = registerer.imgMSE(normalize(tp1_processed[:,:,i]), normalize(corrected_img_12[:,:,i]))
    errors_12.append(error)
    
    overlay(tp1_processed[:,:,i], corrected_img_12[:,:,i], "Timepoint 1 and Timepoint 2 Registered Overlay\n(With Processing)\nSlice {}".format(i), save='output/process/tp1tp2/tp1tp2_slice{:02d}'.format(i))
    print("Slice {}: Registration error is: {} voxels^2".format(i, error))

plt.title("Square Voxel Error vs Z-Slice\nTimepoint 1 vs Timepoint 2")
plt.xlabel('Log Square Voxel Error')
plt.ylabel('Z-Slice')
plt.gca().invert_yaxis()
plt.plot(np.log(errors_12), range(50))
plt.savefig('output/voxelerror12.png')
plt.close()


final_transform_13 = registerer.register_rigid(tp1_processed, tp3_processed, learning_rate=1e-1, iters=25)
print("Computing corrected image (of timepoint 3)")
corrected_img_13 = registerer.resample(tp1_processed, final_transform_13, tp3_processed)

errors_13 = []
for i in range(50):
    error = registerer.imgMSE(normalize(tp1_processed[:,:,i]), normalize(corrected_img_13[:,:,i]))
    errors_13.append(error)
    
    overlay(tp1_processed[:,:,i], corrected_img_13[:,:,i], "Timepoint 1 and Timepoint 3 Registered Overlay\n(With Processing)\nSlice {}".format(i), save='output/process/tp1tp3/tp1tp3_slice{:02d}'.format(i))
    print("Slice {}: Registration error is: {} voxels^2".format(i, error))

plt.title("Square Voxel Error vs Z-Slice\nTimepoint 1 vs Timepoint 3")
plt.xlabel('Log Square Voxel Error')
plt.ylabel('Z-Slice')
plt.gca().invert_yaxis()
plt.plot(np.log(errors_13), range(50))
plt.savefig('output/voxelerror13.png')
plt.close()


final_transform_14 = registerer.register_rigid(tp1_processed, tp4_processed, learning_rate=1e-1, iters=25)
print("Computing corrected image (of timepoint 4)")
corrected_img_14 = registerer.resample(tp1_processed, final_transform_14, tp4_processed)

errors_14 = []
for i in range(50):
    error = registerer.imgMSE(normalize(tp1_processed[:,:,i]), normalize(corrected_img_14[:,:,i]))
    errors_14.append(error)
    
    overlay(tp1_processed[:,:,i], corrected_img_14[:,:,i], "Timepoint 1 and Timepoint 2 Registered Overlay\n(With Processing)\nSlice {}".format(i), save='output/process/tp1tp4/tp1tp4_slice{:02d}'.format(i))
    print("Slice {}: Registration error is: {} voxels^2".format(i, error))

plt.title("Square Voxel Error vs Z-Slice\nTimepoint 1 vs Timepoint 4")
plt.xlabel('Log Square Voxel Error')
plt.ylabel('Z-Slice')
plt.gca().invert_yaxis()
plt.plot(np.log(errors_14), range(50))
plt.savefig('output/voxelerror14.png')
plt.close()
