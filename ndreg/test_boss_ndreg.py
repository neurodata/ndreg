from ndreg import *

boss_config_file = 'C:\\Users\\ben\\Documents\\repos\\ndreg\\ndreg\\neurodata.cfg'
# intern
rmt = BossRemote(boss_config_file)

collection = 'ben_dev'
experiment = 'rand_dev'
channel = 'image'

x_size = 500
y_size = 200
bit_width = 16
dtype = 'uint' + str(bit_width)

exp_setup, coord_actual, chan_actual = setup_channel_boss(rmt, collection, experiment, channel, 
                                                          channel_type='image', datatype=dtype)

print('resources set up')

# img = imgDownload_boss(rmt, chan_actual, coord_actual, resolution=0, size=[500,200,1], start=[0,0,0], isotropic=False)
# print('img downloaded')

img_data = np.random.randint(1, 2**bit_width, size=(y_size, x_size), dtype=dtype)
img = sitk.GetImageFromArray(img_data)
print('created img')

imgUpload_boss(rmt, img, chan_actual, coord_actual, resolution=0, start=[0,0,0], propagate=False, isotropic=False)
print('uploaded image')

img2 = imgDownload_boss(rmt, chan_actual, coord_actual, resolution=0, size=[x_size,y_size,1], start=[0,0,0], isotropic=False)
img_data2 = sitk.GetArrayFromImage(img2)
print('downloaded copy of image')

data_equals = np.array_equal(img_data,img_data2)
print('Data downloaded == data uploaded:', data_equals)



import matplotlib.pyplot as plt
import matplotlib as mpl
# data = sitk.GetArrayFromImage(img)
plt.figure(1)
plt.imshow(img_data, cmap='gray')

plt.figure(2)
plt.imshow(img_data2, cmap='gray')

plt.show()
print('plotted')