import base64
import io
from django.conf import settings
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
import time
import glob
import hashlib

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
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.input_directory = "./duke_original/image"
        self.ground_truth_directory = "./duke_original/lesion"
        self.last_mask = None # Used to store the last output.
        self.last_predicted_points = []
        self.final_mask = []
        self.mask_keys = set()  # For duplicate prevention

    def grow_region(self, image, seed_mask, fluid_mask=None, threshold=5, max_area=4000):
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

    def _generate_mask_key(self, pixels, color):
        """
        Generate a hash key to uniquely identify a mask by its pixels and color.
        """
        flat = ''.join(f'{y},{x}' for y, x in pixels)
        return hashlib.md5((flat + color).encode()).hexdigest()

    #TODO: Currently, the image that is showing on the front end is the median filtered image. Maybe we can
    # change that.
    def run_segmentation_from_json_without_ground_truth(self, image, annotation_json):
        # COMPLETELY reset ALL previous segmentation data
        self.final_mask = []
        self.last_predicted_points = []
        self.last_mask = None
        self.mask_keys = set()  # Reset mask keys for new segmentation

        # Preprocess the image with median filtering
        print(annotation_json)
        image, fluid_mask = self.preprocessor.preprocess_image(image)
        # Convert to grayscale
        gray_image = (image * 255).astype(np.uint8)
        image_rgb = np.stack([gray_image] * 3, axis=-1).astype(np.uint8)

        # Returns the preprocessed image if there are no annotations.
        if (not annotation_json.get("shapes") or not annotation_json["shapes"][0].get("points")):
            return Image.fromarray(image_rgb)

        # Check if we have layer-based annotation (new format) or color-based (legacy format)
        has_layer_data = any('layerId' in shape for shape in annotation_json.get('shapes', [{}]))

        if has_layer_data:
            # Use layer-based processing (enhanced functionality)
            return self._process_layer_based_annotation(image, annotation_json, gray_image, image_rgb, fluid_mask)
        else:
            # Use color-based processing (legacy compatibility)
            return self._process_color_based_annotation(image, annotation_json, gray_image, image_rgb, fluid_mask)

    def _process_layer_based_annotation(self, image, annotation_json, gray_image, image_rgb, fluid_mask):
        """Process annotations with layer information (enhanced functionality)"""
        # Extract points, colors, and layer IDs
        points = np.array(annotation_json['shapes'][0]['points'], dtype=np.int32)
        colors = annotation_json['shapes'][0].get('color', [])
        layer_ids = annotation_json['shapes'][0].get('layerId', [])

        # Group points by LAYER instead of color!!!!
        layer_groups = {}
        for i, (point, color, layer_id) in enumerate(zip(points, colors, layer_ids)):
            if layer_id not in layer_groups:
                layer_groups[layer_id] = {
                    'points': [],
                    'color': color  # Use the layer's color
                }
            layer_groups[layer_id]['points'].append(point)

        # Store all predicted points across all layers
        all_predicted_points = []
        individual_masks = []

        # Process each layer separately to get individual masks.
        for layer_id, layer_data in layer_groups.items():
            color_hex = layer_data['color']
            color_points = layer_data['points']

            if not color_points:
                continue

            # Convert points to numpy array
            color_points_array = np.array(color_points, dtype=np.int32)

            # Create seed mask for this layer.
            seed_mask = np.zeros(image.shape, dtype=np.uint8)

            # Draw points on seed mask
            for point in color_points:
                cv2.circle(seed_mask, (point[0], point[1]), 1, 1, -1)

            # Ensure that the annotation's resulting mask can only apply in the identified fluid region.
            seed_mask = seed_mask * fluid_mask

            # Apply region growing for this layer.
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

            # Store individual mask with its color and layer ID
            individual_masks.append({
                "mask": final_mask,
                "color": color_hex,
                "layer_id": layer_id
            })

        # Merge masks that overlap. What this means is, if two layers create the same mask, only choose ONE.
        resolved_masks = self.resolve_overlapping_masks(individual_masks)

        for i, mask_data in enumerate(resolved_masks):
            final_mask = mask_data["mask"]
            dominant_color = mask_data["color"]
            layer_id = mask_data["layer_id"]

            # Find all non-zero pixel coordinates in the final mask
            ys, xs = np.where(final_mask == 1)
            pixels = [[int(y), int(x)] for y, x in zip(ys, xs)]

            # Check for duplicates using hash
            mask_key = self._generate_mask_key(pixels, dominant_color)
            if mask_key not in self.mask_keys:
                self.mask_keys.add(mask_key)

                # Send final masks for Konva groups with ids, colors, and points
                region_id = f"region-{len(self.final_mask)}"

                # Add to the mask collection
                self.final_mask.append({
                    "regionId": region_id,
                    "pixels": pixels,
                    "color": dominant_color
                })

                # Apply color to segmented region
                rgb_color = self.hex_to_rgb(dominant_color)
                image_rgb[final_mask == 1] = rgb_color

                # Extract contours for this mask.
                contours = measure.find_contours(final_mask, 0.5)
                for contour in contours:
                    for y, x in contour:
                        all_predicted_points.append([[int(x), int(y)], dominant_color])

        self.last_predicted_points = all_predicted_points
        return Image.fromarray(image_rgb)

    def _process_color_based_annotation(self, image, annotation_json, gray_image, image_rgb, fluid_mask):
        """Process annotations with color-based grouping (legacy compatibility)"""
        # Extract points and colors
        points = np.array(annotation_json['shapes'][0]['points'], dtype=np.int32)
        colors = annotation_json['shapes'][0].get('color', [])

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

            # Find all non-zero pixel coordinates in the final mask
            ys, xs = np.where(final_mask == 1)
            pixels = [[int(y), int(x)] for y, x in zip(ys, xs)]

            # Check for duplicates using hash
            mask_key = self._generate_mask_key(pixels, color_hex)
            if mask_key not in self.mask_keys:
                self.mask_keys.add(mask_key)

                # Send final masks for Konva groups with ids, colors, and points
                region_id = f"region-{len(self.final_mask)}"

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
        return Image.fromarray(image_rgb)

    def get_predicted_points(self):
        return self.last_predicted_points

    def get_final_mask(self):
        return self.final_mask

        # This function keeps the LARGEST mask in the case that two masks overlap.
    def resolve_overlapping_masks(self, individual_masks):
        if not individual_masks:
            return []

        for mask_data in individual_masks:
            mask_data["size"] = np.sum(mask_data["mask"])

        # Sort by size (largest first)
        individual_masks.sort(key=lambda x: x["size"], reverse=True)

        resolved_masks = []
        used_pixels = np.zeros(individual_masks[0]["mask"].shape, dtype=bool)

        for mask_data in individual_masks:
            current_mask = mask_data["mask"].copy()

            # Remove pixels that are already used by larger masks.
            current_mask[used_pixels] = 0

            # If there are still pixels left, keep this mask.
            if np.any(current_mask):
                # Mark these pixels as used.
                used_pixels[current_mask == 1] = True

                resolved_masks.append({
                    "mask": current_mask,
                    "color": mask_data["color"],
                    "layer_id": mask_data["layer_id"],
                    "size": np.sum(current_mask)
                })

        return resolved_masks

    def get_segmentation_masks(self, filename):
        segmentation_dir = os.path.join(settings.MEDIA_ROOT, "segmentations")
        os.makedirs(segmentation_dir, exist_ok=True)

        # Clear all previous segmentation files (I think this would work better, since our project would not have as many files.)
        old_files = glob.glob(os.path.join(segmentation_dir, "*"))
        for old_file in old_files:
            try:
                os.remove(old_file)
            except OSError:
                pass  # File might be in use, skip it.

        # File naming
        filename_base = os.path.splitext(filename)[0]

        mask_urls = []

        if self.final_mask:
            all_pixels = []
            for mask_data in self.final_mask:
                all_pixels.extend(mask_data["pixels"])

            if all_pixels:
                max_y = max(pixel[0] for pixel in all_pixels) + 1
                max_x = max(pixel[1] for pixel in all_pixels) + 1
            else:
                max_y, max_x = 512, 512
        else:
            max_y, max_x = 512, 512

        # Create individual mask files for each region.
        for i, mask_data in enumerate(self.final_mask):
            region_id = mask_data["regionId"]
            color = mask_data["color"]
            pixels = mask_data["pixels"]

            # Create individual mask filename, both .png and .npy.
            npy_filename = f"{filename_base}_{region_id}_mask.npy"
            png_filename = f"{filename_base}_{region_id}_mask.png"

            npy_path = os.path.join(segmentation_dir, npy_filename)
            png_path = os.path.join(segmentation_dir, png_filename)

            individual_mask_data = {
                "regionId": region_id,
                "pixels": pixels,
                "color": color
            }
            np.save(npy_path, individual_mask_data)

            # Create visual mask image (RGBA with transparency for .png)
            mask_image = np.zeros((max_y, max_x, 4), dtype=np.uint8)  # RGBA
            rgb_color = self.hex_to_rgb(color)

            for pixel in pixels:
                y, x = pixel[0], pixel[1]
                if 0 <= y < max_y and 0 <= x < max_x:
                    mask_image[y, x] = [rgb_color[0], rgb_color[1], rgb_color[2], 180]  # Semi-transparent

            # Save PNG with transparency
            Image.fromarray(mask_image, 'RGBA').save(png_path)

            npy_url = f"{settings.MEDIA_URL}segmentations/{npy_filename}"
            png_url = f"{settings.MEDIA_URL}segmentations/{png_filename}"

            mask_urls.append({
                "regionId": region_id,
                "color": color,
                "npyUrl": npy_url,
                "pngUrl": png_url,
                "npyFilename": npy_filename,
                "pngFilename": png_filename
            })

        combined_mask_url = self.create_combined_mask_preview(filename)

        return {
            "individual_masks": mask_urls,
            "combined_mask_url": combined_mask_url
        }

    # This function is to display the combined masks right next to the segmented result.
    def create_combined_mask_preview(self, filename):
        segmentation_dir = os.path.join(settings.MEDIA_ROOT, "segmentations")
        os.makedirs(segmentation_dir, exist_ok=True)

        if not self.final_mask:
            return None

        all_pixels = []
        for mask_data in self.final_mask:
            all_pixels.extend(mask_data["pixels"])

        if all_pixels:
            max_y = max(pixel[0] for pixel in all_pixels) + 1
            max_x = max(pixel[1] for pixel in all_pixels) + 1
        else:
            return None

        combined_mask = np.zeros((max_y, max_x, 4), dtype=np.uint8)

        for mask_data in self.final_mask:
            color = mask_data["color"]
            pixels = mask_data["pixels"]
            rgb_color = self.hex_to_rgb(color)

            for pixel in pixels:
                y, x = pixel[0], pixel[1]
                if 0 <= y < max_y and 0 <= x < max_x:
                    combined_mask[y, x] = [rgb_color[0], rgb_color[1], rgb_color[2], 120]  # Semi-transparent

        filename_base = os.path.splitext(filename)[0]
        combined_filename = f"{filename_base}_combined_masks.png"
        combined_path = os.path.join(segmentation_dir, combined_filename)
        Image.fromarray(combined_mask, 'RGBA').save(combined_path)

        timestamp = str(int(time.time() * 1000))  # For some reason, I get bugs all the time if I don't use the timestamp as a unique ID??? Why??
        combined_url = f"{settings.MEDIA_URL}segmentations/{combined_filename}?v={timestamp}"
        return combined_url

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

    # To ensure that the masks are reset every time "ready to segment" is called.
    # QUESTION: Should we delete ALL the masks from the media folders too??
    def reset_masks(self):
        """Reset all stored masks for new segmentation"""
        self.final_mask = []
        self.last_predicted_points = []
        self.last_mask = None
        self.mask_keys = set()

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