import sys
sys.path.append('../')
import Preprocessor
import SimpleITK as sitk

class TestPreprocessor(BaseTestCase):
    
    def test_good_initialization(self):
        img = sitk.ReadImage(
        
        
