# test_live_cell_segmentation.py
import unittest
import numpy as np
from live_cell_segmentation import load_data, unet_model, train_model, evaluate_and_visualize

class TestLiveCellSegmentation(unittest.TestCase):

    def setUp(self):
        # Set up any necessary data for tests
        self.data_dir = 'path/to/test/data'
        self.img_size = (256, 256)
        self.train_images, self.val_images, self.val_masks = load_data(self.data_dir, self.img_size)
        self.model = unet_model()

    def test_load_data(self):
        self.assertEqual(len(self.train_images.shape), 4)
        self.assertEqual(len(self.val_images.shape), 4)
        self.assertEqual(len(self.val_masks.shape), 4)
        self.assertEqual(self.train_images.shape[1:], (*self.img_size, 1))
        self.assertEqual(self.val_images.shape[1:], (*self.img_size, 1))
        self.assertEqual(self.val_masks.shape[1:], (*self.img_size, 1))

    def test_unet_model(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.input_shape[1:], (*self.img_size, 1))

    def test_train_model(self):
        try:
            train_model(self.model, self.train_images, (self.val_images, self.val_masks), epochs=1)
        except Exception as e:
            self.fail(f"train_model raised Exception unexpectedly: {e}")

    def test_evaluate_and_visualize(self):
        try:
            evaluate_and_visualize(self.model, self.val_images, self.val_masks)
        except Exception as e:
            self.fail(f"evaluate_and_visualize raised Exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
