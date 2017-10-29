import sys
sys.path.append('../')
import Preprocessor
import SimpleITK as sitk

class TestPreprocessor(BaseTestCase):
    
    @classmethod
    def setUpClass():
        img = tf.imread('../sample_data/atlas_to_control_9.tif')
        img_sitk = sitk.GetImageFromArray(img)
        img_sitk.SetSpacing((0.0005, 0.0005, 0.005))
        img_sitk.SetDirection(sitk.AffineTransform(img_sitk.GetDimension()).GetMatrix())
        img_sitk.SetOrigin([0] * img_sitk.GetDimension())

    def test_good_initialization(self):
        preprocessor_good = Preprocessor(img_sitk)

    def test_bad_initialization(self):
        preprocessor_bad = Preprocessor(sitk.GetArrayFromImage(img_sitk))

    def test_remove_streak(self):

    def test_create_mask(self):

    def test_remove_circle(self):

    def correct_bias_field(self):

