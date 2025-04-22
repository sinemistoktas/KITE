import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, morphology, restoration, transform, registration, exposure, feature, measure
from scipy.ndimage import map_coordinates, binary_fill_holes, median_filter, gaussian_laplace, uniform_filter, generic_filter, gaussian_filter, laplace
from skimage.io import imread
from scipy.stats import trim_mean
import cv2
import os
import pandas as pd
import json
from data.data_preprocessing import Preprocessor
from PIL import Image
# steps to follow
# 1. load the data
# 2. preprocess the data
# 3. find the ground truth
# 4. make annotation 
# 5. save the annotated data
# 6. start segmentation based on the annotated data
# 7. save the segmented data

# subject 09_31 , 10_33

class SegmentationModel():
    def __init__(self,preprocessor):
        self.preprocessor = preprocessor
        self.input_directory = "./duke_original/image"
        self.ground_truth_directory = "./duke_original/lesion"
        self.last_mask = None # Used to store the last output.
        self.last_predicted_points = []
    
    def grow_region(self,image, seed_mask, threshold=10):
        height, width = image.shape
        grown_mask = seed_mask.copy()
        seeds = list(zip(*np.nonzero(seed_mask)))
        visited = set(seeds)

        while seeds:
            x, y = seeds.pop()
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]: # 8-connectivity
                nx, ny = x + dx, y + dy
                if (0 <= nx < height) and (0 <= ny < width):
                    if (nx, ny) in visited:
                        continue

                    visited.add((nx, ny))
                    if grown_mask[nx, ny] == 0:
                        if abs(int(image[nx, ny]) - int(image[x, y])) < threshold:
                            grown_mask[nx, ny] = 1
                            seeds.append((nx, ny))
        return grown_mask

    def run_segmentation_from_json(self, annotation_json, filename):
        image_path = os.path.join(self.preprocessor.input_directory, filename)
        image = imread(image_path, as_gray=True)
        ground_truth = self.preprocessor.find_ground_truth(filename, image)

        image = self.preprocessor.preprocess_image(image)
        bg_removed = self.preprocessor.thresholding(image, 0.4)
        bg_removed = morphology.closing(bg_removed, morphology.square(8))
        gray_image = (image * 255).astype(np.uint8)

        points = np.array(annotation_json['shapes'][0]['points'], dtype=np.int32)
        seed_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(seed_mask, [points], 1)

        grown_mask = self.grow_region(gray_image, seed_mask, threshold=5)
        grown_mask = morphology.closing(grown_mask, morphology.square(4))

        image_rgb = np.stack([gray_image] * 3, axis=-1).astype(np.uint8)
        image_rgb[grown_mask == 1] = [0, 0, 255]

        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(bg_removed, cmap='gray')
        plt.title("Thresholded")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(image_rgb)
        plt.title("Segmented")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(gray_image, cmap='gray')
        plt.contour(ground_truth, colors='red', linewidths=1)
        plt.title("Ground Truth")
        plt.axis('off')

        plt.tight_layout()
        return fig
    
    # We should call THIS method when somebody uploads an image with no ground truth. The previous method
    # is for showcasing & comparing the ground truth image with the segmented image.
    def run_segmentation_from_json_without_ground_truth(self, image, annotation_json):
        image = self.preprocessor.preprocess_image(image)
        gray_image = (image * 255).astype(np.uint8)

        points = np.array(annotation_json['shapes'][0]['points'], dtype=np.int32)
        seed_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(seed_mask, [points], 1)

        grown_mask = self.grow_region(gray_image, seed_mask, threshold=5)
        grown_mask = morphology.closing(grown_mask, morphology.square(4))
        grown_mask = binary_fill_holes(grown_mask).astype(np.uint8)
        image_rgb = np.stack([gray_image] * 3, axis=-1).astype(np.uint8)
        image_rgb[grown_mask == 1] = [0, 0, 255]

        contours = measure.find_contours(grown_mask, 0.5)
        predicted_points = []
        for contour in contours:
            for y, x in contour:
                predicted_points.append([int(x), int(y)])

        self.last_predicted_points = predicted_points
        return Image.fromarray(image_rgb)
    
    def get_predicted_points(self):
        return self.last_predicted_points

# An example usage of the segmentation class.
if __name__ == "__main__":
    import json

    preprocessor = Preprocessor()
    segmentation_model = SegmentationModel(preprocessor)

    filename = "Subject_10_33.png"
    json_path = "Subject_10_33.json"

    with open(json_path) as f:
        annotation_data = json.load(f)

    fig = segmentation_model.run_segmentation_from_json(annotation_data, filename)

    plt.show()
