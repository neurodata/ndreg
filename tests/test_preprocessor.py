import sys
sys.path.append('../')
from Preprocessor import Preprocessor
import SimpleITK as sitk
import tifffile as tf
from testlib.testcase import BaseTestCase

class PreprocessorTestCase(BaseTestCase):
    
    @classmethod
    def setUpClass(self):
        self.img = tf.imread('../sample_data/atlas_to_control_9.tif')
        self.img_sitk = sitk.GetImageFromArray(self.img)
        self.img_sitk.SetSpacing((0.0005, 0.0005, 0.005))
        self.img_sitk.SetDirection(sitk.AffineTransform(self.img_sitk.GetDimension()).GetMatrix())
        self.img_sitk.SetOrigin([0] * self.img_sitk.GetDimension())

    def test_good_initialization(self):
        preprocessor_good = Preprocessor(self.img_sitk)

    def test_bad_initialization(self):
        try:
            preprocessor_bad = Preprocessor(self.img)
        except Exception as e:
            assert(e.message == "Please convert your image into a SimpleITK image")  

    def test_remove_streaks(self):
        preprocessor = Preprocessor(self.img_sitk)
        preprocessor.remove_streaks()

    def test_create_mask(self):
        preprocessor = Preprocessor(self.img_sitk)
        preprocessor.create_mask()

    def test_remove_circle(self):
        preprocessor = Preprocessor(self.img_sitk)
        preprocessor.remove_circle()

    def test_correct_bias_field(self):
        preprocessor = Preprocessor(self.img_sitk)
        preprocessor.correct_bias_field()

