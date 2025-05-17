import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, morphology, restoration, transform, registration, exposure, feature, measure
from skimage.measure import label
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
        self.final_mask = []
    
    def grow_region(self,image, seed_mask,fluid_mask=None, threshold=5, max_area=4000):
        height, width = image.shape
        grown_mask = seed_mask.copy()
        seeds = list(zip(*np.nonzero(seed_mask)))
        visited = set(seeds)
        if not seeds:
            return grown_mask
        seed_value = np.min(image[seed_mask == 1])
        print("Using seed value:", seed_value)

        while seeds:
            x, y = seeds.pop()
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < height) and (0 <= ny < width):
                    if (nx, ny) in visited:
                        continue
                    visited.add((nx, ny))
                    if grown_mask[nx, ny] == 0:
                        intensity = image[nx, ny]
                        if intensity <= seed_value + threshold and (fluid_mask is None or fluid_mask[nx, ny]):
                            grown_mask[nx, ny] = 1
                            seeds.append((nx, ny))
            if grown_mask.sum() > max_area:
                print("Region grew too large, reverting to seed only.")
                return seed_mask

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

    #NOTE TO MİSLİNA: THE PREPROCESSED IMAGE CAN BE FOUND BY CHECKING "FLUID_MASK"

    #TODO: Currently, the image that is showing on the front end is the median filtered image. Maybe we can
    # change that.
    def run_segmentation_from_json_without_ground_truth(self, image, annotation_json):
    # Preprocess the image with median filtering
        print(annotation_json)
        image, fluid_mask = self.preprocessor.preprocess_image(image)
        # Convert to grayscale
        gray_image = (image * 255).astype(np.uint8)
        
        # Returns the preprocessed image if there are no annotations.
        if (not annotation_json.get("shapes") or not annotation_json["shapes"][0].get("points")):
            self.last_predicted_points = []
            image_rgb = np.stack([gray_image] * 3, axis=-1).astype(np.uint8)
            return Image.fromarray(image_rgb)
        
        # Extract points and colors
        points = np.array(annotation_json['shapes'][0]['points'], dtype=np.int32)
        colors = annotation_json['shapes'][0].get('color', [])
        
        # Create the RGB image from grayscale
        image_rgb = np.stack([gray_image] * 3, axis=-1).astype(np.uint8)
        
        # Group points by color
        color_groups = {}
        for i, (point, color) in enumerate(zip(points, colors)):
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(point)
        
        # Store all predicted points across all colors
        all_predicted_points = []
        
        # Process each color group separately
        for color_hex, color_points in color_groups.items():
            if not color_points:
                continue
                
            # Convert points to numpy array
            color_points_array = np.array(color_points, dtype=np.int32)
            
            # Create seed mask for this color group
            seed_mask = np.zeros(image.shape, dtype=np.uint8)
            
            # Draw points on seed mask
            for point in color_points:
                cv2.circle(seed_mask, (point[0], point[1]), 1, 1, -1)
            
            # Ensure that the annotation's resulting mask can only apply in the identified fluid region.
            seed_mask = seed_mask * fluid_mask
            
            # Apply region growing for this color group
            grown_mask = self.grow_region(gray_image, seed_mask, fluid_mask=fluid_mask, threshold=5)
            
            # Only include the fluid regions that intersect with the region grown mask
            labeled = label(fluid_mask)
            grown_labels = np.unique(labeled[grown_mask == 1])
            connected_fluid = np.isin(labeled, grown_labels).astype(np.uint8)
            
            # Merge the regions.
            final_mask = np.logical_or(grown_mask, connected_fluid).astype(np.uint8)
            
            # Post-processing, here I applied closing and the infill method.
            final_mask = morphology.closing(final_mask, morphology.disk(4))
            final_mask = binary_fill_holes(final_mask).astype(np.uint8)

            # Send final masks for Konva groups with ids, colors, and points
            region_id = f"region-{len(self.final_mask)}"

            # Find all non-zero pixel coordinates in the final mask
            ys, xs = np.where(final_mask == 1)
            pixels = [[int(y), int(x)] for y, x in zip(ys, xs)]

            
            # Add to the mask collection
            self.final_mask.append({
                "regionId": region_id,
                "pixels": pixels,
                "color": color_hex
            })
            # Apply color to segmented region
            rgb_color = self.hex_to_rgb(color_hex)
            image_rgb[final_mask == 1] = rgb_color
            
            # Extract contours for this color group
            contours = measure.find_contours(final_mask, 0.5)
            # This part extracts the contours of the final mask, which will be used in the front end step.
            for contour in contours:
                for y, x in contour:
                    all_predicted_points.append([[int(x), int(y)], color_hex])
        self.last_predicted_points = all_predicted_points
        # Image.fromarray((fluid_mask * 255).astype(np.uint8)).show() # FLUID MASK USED FOR TESTING!!!
        # Image.fromarray((seed_mask * 255).astype(np.uint8)).show() # SEED MASK USED FOR TESTING!!!
        
        return Image.fromarray(image_rgb)
    
    def get_predicted_points(self):
        return self.last_predicted_points
    
    def get_final_mask(self):
        #mask = np.column_stack(np.where(self.final_mask == 1)).tolist()
        return self.final_mask 


    def hex_to_rgb(self, hex_color):
        """Convert hex color string to RGB list"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return [r, g, b]
        else:
            # Default to blue if hex format is incorrect
            return [0, 0, 255]
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
