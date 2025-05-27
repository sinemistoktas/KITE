# website/frontend/views.py

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from data.data_preprocessing import Preprocessor
from data.data_segmentation import SegmentationModel
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import io
import os
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage import morphology
from base64 import b64encode
from PIL import Image
import traceback
from io import BytesIO
import numpy as np
import torch
import cv2 as cv

preprocessor = Preprocessor()
segmentation_model = SegmentationModel(preprocessor)

#UNet
class UnetPredictor:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.fluid_class_ids = [1] #adjust accordingly

    def get_final_mask_for_interactivity(self, segmentation_map, fluid_only=True):
        final_mask = []

        if fluid_only:
            combined_fluid_mask = np.zeros_like(segmentation_map, dtype=np.uint8)
            for fluid_class in self.fluid_class_ids:
                combined_fluid_mask = np.logical_or(combined_fluid_mask, segmentation_map == fluid_class).astype(np.uint8)

            from skimage.measure import label
            labeled_mask = label(combined_fluid_mask)

            for region_id in range(1, labeled_mask.max() + 1):
                region_pixels = np.where(labeled_mask == region_id)
                if len(region_pixels[0]) > 10:  # Minimum pixel count
                    pixels = [[int(y), int(x)] for y, x in zip(region_pixels[0], region_pixels[1])]
                    final_mask.append({
                        "regionId": f"fluid_region_{region_id}",
                        "pixels": pixels,
                        "color": "rgba(0, 255, 255, 0.6)",  # Cyan for fluid
                        "class_id": "fluid"
                    })
        else:
            # Process each class separately
            colors = [
                [0, 0, 0],      # Class 0: Black (background)
                [255, 0, 0],    # Class 1: Red
                [0, 255, 0],    # Class 2: Green
                [0, 0, 255],    # Class 3: Blue
                [255, 255, 0],  # Class 4: Yellow
                [255, 0, 255],  # Class 5: Magenta
                [0, 255, 255],  # Class 6: Cyan
                [128, 0, 0],    # Class 7: Maroon
                [0, 128, 0],    # Class 8: Dark green
                [0, 0, 128]     # Class 9: Navy blue
            ]

            from skimage.measure import label

            for class_idx in range(1, min(10, len(colors))):
                binary_mask = (segmentation_map == class_idx).astype(np.uint8)
                if np.sum(binary_mask) == 0:
                    continue

                labeled_mask = label(binary_mask)

                for region_id in range(1, labeled_mask.max() + 1):
                    region_pixels = np.where(labeled_mask == region_id)
                    if len(region_pixels[0]) > 10:  # Minimum pixel count
                        pixels = [[int(y), int(x)] for y, x in zip(region_pixels[0], region_pixels[1])]
                        color = colors[class_idx]
                        final_mask.append({
                            "regionId": f"class_{class_idx}_region_{region_id}",
                            "pixels": pixels,
                            "color": f"rgba({color[0]}, {color[1]}, {color[2]}, 0.6)",
                            "class_id": class_idx
                        })

        return final_mask

    def predict(self, img_path, fluid_only=True):
        img = Image.open(img_path)
        original_size = img.size

        img = img.resize((512, 224))
        if img.mode != 'L':
            img = img.convert('L')

        img_array = np.array(img).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)

        probabilities = torch.softmax(output, dim=1)

        _, predicted = torch.max(output, 1)
        segmentation_map = predicted.squeeze().cpu().numpy()

        original_image = Image.open(img_path).convert('RGB')
        original_array = np.array(original_image.resize((512, 224)))

        if fluid_only:
            combined_fluid_mask = np.zeros_like(segmentation_map, dtype=np.uint8)

            for fluid_class in self.fluid_class_ids:
                fluid_prob = probabilities[0, fluid_class, :, :].cpu().numpy()

                fluid_regions = (fluid_prob > 0.5).astype(np.uint8)
                combined_fluid_mask = np.logical_or(combined_fluid_mask, fluid_regions).astype(np.uint8)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # Small circular kernel
            combined_fluid_mask = cv.morphologyEx(combined_fluid_mask, cv.MORPH_CLOSE, kernel)  # Fill gaps
            combined_fluid_mask = cv.morphologyEx(combined_fluid_mask, cv.MORPH_OPEN, kernel)   # Remove noise

            overlay = original_array.copy()
            highlight = np.zeros_like(original_array)
            highlight[combined_fluid_mask > 0] = [0, 255, 255]  # Cyan
            overlay = cv.addWeighted(overlay, 1.0, highlight, 0.6, 0)

            result_img = Image.fromarray(overlay)
            predicted_points = self.get_fluid_contours(combined_fluid_mask)

        else:
            overlay = self.create_multiclass_overlay(original_array, segmentation_map)
            result_img = Image.fromarray(overlay)
            predicted_points = self.get_all_class_contours(segmentation_map)

        return result_img, predicted_points, segmentation_map

    def create_fluid_overlay(self, original_array, fluid_mask, alpha=0.6):
        overlay = original_array.copy()  # Start with original image

        fluid_color = [0, 255, 255]  # RGB: Cyan

        highlight = np.zeros_like(original_array)  # Black image same size as original
        highlight[fluid_mask > 0] = fluid_color   # Set fluid pixels to cyan

        overlay = cv.addWeighted(overlay, 1.0, highlight, alpha, 0)

        return overlay

    def overlay_on_original(self, original_image, segmentation_mask, alpha=0.5):
        if original_image.shape[:2] != segmentation_mask.shape:
            original_image = cv.resize(original_image, (segmentation_mask.shape[1], segmentation_mask.shape[0]))

        if len(original_image.shape) == 2:  # Grayscale (1 channel)
            original_3ch = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
        elif len(original_image.shape) == 3 and original_image.shape[2] == 3:  # Already RGB/BGR (3 channels)
            original_3ch = original_image.copy()
        else:
            print(f"Unexpected image shape: {original_image.shape}")

            if len(original_image.shape) == 3 and original_image.shape[2] == 4:  # RGBA
                original_3ch = original_image[:, :, :3]  # Just take RGB channels
            else:
                original_3ch = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)
                for i in range(3):
                    original_3ch[:, :, i] = original_image[:, :, 0] if len(original_image.shape) == 3 else original_image

        highlight = np.zeros_like(original_3ch)
        highlight[segmentation_mask > 0] = [0, 0, 255]  # Red channel

        overlay = cv.addWeighted(original_3ch, 1.0, highlight, alpha, 0)

        return overlay

    def create_multiclass_overlay(self, original_array, segmentation_map):
        colors = [
            [0, 0, 0],      # Class 0: Black (background)
            [255, 0, 0],    # Class 1: Red
            [0, 255, 0],    # Class 2: Green
            [0, 0, 255],    # Class 3: Blue
            [255, 255, 0],  # Class 4: Yellow
            [255, 0, 255],  # Class 5: Magenta
            [0, 255, 255],  # Class 6: Cyan
            [128, 0, 0],    # Class 7: Maroon
            [0, 128, 0],    # Class 8: Dark green
            [0, 0, 128]     # Class 9: Navy blue
        ]

        overlay = original_array.copy()

        for class_idx in range(1, min(10, len(colors))):
            binary_mask = (segmentation_map == class_idx).astype(np.uint8)

            if np.sum(binary_mask) == 0:
                continue

            class_mask = np.zeros_like(original_array)
            class_mask[binary_mask > 0] = colors[class_idx]

            overlay = cv.addWeighted(overlay, 1.0, class_mask, 0.5, 0)

        return overlay

    def get_fluid_contours(self, fluid_mask, min_area=20):
        contours, _ = cv.findContours(fluid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        points = []

        for contour in contours:
            area = cv.contourArea(contour)
            if area < min_area:  # Skip small contours (likely noise)
                continue

            contour_points = []
            for point in contour:
                x, y = point[0]  # Extract x, y coordinates
                contour_points.append([float(x), float(y)])  # Convert to float for JSON

            if len(contour_points) > 2:
                points.append({
                    "shape_type": "polygon",
                    "points": contour_points,
                    "color": [0, 255, 255],
                    "class_id": "fluid",
                    "label": "fluid"
                })

        return points

    def get_all_class_contours(self, segmentation_map):
        colors = [
            [0, 0, 0],      # Class 0: Black (background)
            [255, 0, 0],    # Class 1: Red
            [0, 255, 0],    # Class 2: Green
            [0, 0, 255],    # Class 3: Blue
            [255, 255, 0],  # Class 4: Yellow
            [255, 0, 255],  # Class 5: Magenta
            [0, 255, 255],  # Class 6: Cyan
            [128, 0, 0],    # Class 7: Maroon
            [0, 128, 0],    # Class 8: Dark green
            [0, 0, 128]     # Class 9: Navy blue
        ]

        predicted_points = []

        for class_idx in range(1, min(10, len(colors))):
            binary_mask = (segmentation_map == class_idx).astype(np.uint8)

            if np.sum(binary_mask) == 0:
                continue

            contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv.contourArea(contour)
                if area < 10:  # Filter very small contours
                    continue

                contour_points = []
                for point in contour:
                    x, y = point[0]
                    contour_points.append([float(x), float(y)])

                if len(contour_points) > 2:
                    predicted_points.append({
                        "shape_type": "polygon",
                        "points": contour_points,
                        "color": colors[class_idx],  # Use class-specific color
                        "class_id": class_idx        # Include class ID
                    })

        return predicted_points

    def get_segmentation_map_image(self, segmentation_map):
        colors = [
            [0, 0, 0],      # Class 0: Black (background)
            [255, 0, 0],    # Class 1: Red
            [0, 255, 0],    # Class 2: Green
            [0, 0, 255],    # Class 3: Blue
            [255, 255, 0],  # Class 4: Yellow
            [255, 0, 255],  # Class 5: Magenta
            [0, 255, 255],  # Class 6: Cyan
            [128, 0, 0],    # Class 7: Maroon
            [0, 128, 0],    # Class 8: Dark green
            [0, 0, 128]     # Class 9: Navy blue
        ]
        height, width = segmentation_map.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        for class_idx in range(min(10, len(colors))):
            mask = segmentation_map == class_idx
            rgb_image[mask] = colors[class_idx]

        return Image.fromarray(rgb_image)

    def predict_fluid_with_confidence(self, img_path, confidence_threshold=0.7, min_area=50):
        img = Image.open(img_path)
        img = img.resize((512, 224))
        if img.mode != 'L':
            img = img.convert('L')

        img_array = np.array(img).astype(np.float32)

        if np.std(img_array) > 0:
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
        else:
            img_array = img_array / 255.0

        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)

        probabilities = torch.softmax(output, dim=1)

        combined_fluid_prob = torch.zeros_like(probabilities[0, 0, :, :])
        for fluid_class in self.fluid_class_ids:
            combined_fluid_prob += probabilities[0, fluid_class, :, :]

        fluid_mask = (combined_fluid_prob.cpu().numpy() > confidence_threshold).astype(np.uint8)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # Larger kernel for more aggressive cleaning
        fluid_mask = cv.morphologyEx(fluid_mask, cv.MORPH_CLOSE, kernel)  # Close gaps
        fluid_mask = cv.morphologyEx(fluid_mask, cv.MORPH_OPEN, kernel)   # Remove noise

        contours, _ = cv.findContours(fluid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv.contourArea(contour) < min_area:
                cv.fillPoly(fluid_mask, [contour], 0)

        original_image = Image.open(img_path).convert('RGB')
        original_array = np.array(original_image.resize((512, 224)))

        overlay = self.create_fluid_overlay(original_array, fluid_mask)
        result_img = Image.fromarray(overlay)
        predicted_points = self.get_fluid_contours(fluid_mask, min_area)

        return result_img, predicted_points, combined_fluid_prob.cpu().numpy()

    def segmentation_to_points(self, segmentation_map, class_index = 1):
        binary_mask = (segmentation_map == class_index).astype(np.uint8)

        contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        points = []
        for contour in contours:
            contour_points = []
            for point in contour:
                x, y = point[0]
                contour_points.append([float(x), float(y)])

            if len(contour_points) > 2:  # Only include if there are enough points
                points.append({
                    "shape_type": "polygon",
                    "points": contour_points,
                    "color": [255, 0, 0]  # Red color for the contour outline
                })

        return points

#initialize UNet
def get_unet_model_path():
    """
    Dynamically determine the UNet model path based on the current file location
    """
    # Get the current file's directory (views.py location)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from website/frontend/ to the project root, then to unet/notebooks/
    project_root = os.path.dirname(os.path.dirname(current_file_dir))
    model_path = os.path.join(project_root, 'unet', 'notebooks', 'unet_traced.pt')
    return model_path

unet_model_path = get_unet_model_path()
#unet_model_path = '/Users/durutandogan/KITE/unet/notebooks/unet_traced.pt'

print("UNet model path:", unet_model_path)
print("Exists:", os.path.exists(unet_model_path))
unet_predictor = None

try:
    if os.path.exists(unet_model_path):
        unet_predictor = UnetPredictor(unet_model_path)
        print("UNet model loaded successfully")
    else:
        print("UNet model path not found. Expected location:", unet_model_path)
        print("Please ensure the model file exists at the expected location.")
except Exception as e:
    print(f"UNet model loading failed: {e}")

def home(request):
    return render(request, 'pages/home.html')

def seg_tool(request):
    image_url = None
    image_name = None
    algorithm_name = None
    algorithm = None
    segmentation_method = None
    
    # Handle both new algorithm system and legacy segmentation method system
    if request.method == 'POST':
        segmentation_method = request.POST.get('segmentation_method', 'traditional')
        algorithm = request.POST.get('algorithm')
        algorithm_name = dict(request.POST).get('algorithm_name', [''])[0]
    else:
        segmentation_method = 'traditional'

    context = {
        'image_url': image_url,
        'image_name': image_name,
        'algorithm_name': algorithm_name,
        'segmentation_method': segmentation_method,
        'unet_available': unet_predictor is not None
    }

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_name = image.name
        fs = FileSystemStorage()
        filename = fs.save(image_name, image)
        image_url = fs.url(filename)

        context.update({
            'image_url': image_url,
            'image_name': image_name,
            'algorithm_name': algorithm_name,
            'segmentation_method': segmentation_method,
            'unet_available': unet_predictor is not None
        })

        # Handle UNet auto-processing for legacy segmentation method system
        if segmentation_method == 'unet' and unet_predictor is not None:
            try:
                image_path = os.path.join(settings.MEDIA_ROOT, filename)
                result_img, predicted_points, segmentation_map = unet_predictor.predict(image_path)
                print("UNet predicted points:", predicted_points)
                print("Image size:", result_img.size)
                result_filename = f"{os.path.splitext(filename)[0]}_unet_segmented.png"
                result_path = os.path.join(settings.MEDIA_ROOT, result_filename)
                result_img.save(result_path)

                context.update({
                    'segmented_image_url': fs.url(result_filename),
                    'show_segmentation_result': True,
                    'unet_mode': True,
                    'predicted_points': json.dumps(predicted_points)
                })
            except Exception as e:
                traceback_str = traceback.format_exc()
                print(f"UNet processing error:\n{traceback_str}")
                context['unet_error'] = str(e)
        else:
            context['unet_mode'] = False

    return render(request, 'pages/segtool.html', context)

def segment_image(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # Determine algorithm/method
            algorithm = data.get("algorithm", "kite")
            use_unet = data.get("use_unet", False)
            segmentation_method = data.get("segmentation_method", "traditional")

            # Get filename
            filename = data.get("image_name")
            if not filename:
                return JsonResponse({"error": "No image name provided."}, status=400)

            # Get file
            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(image_path):
                return JsonResponse({"error": "Image file not found."}, status=404)

            # Handle UNet processing (both new and legacy systems)
            if (use_unet or segmentation_method == 'unet') and unet_predictor is not None:
                result_img, predicted_points, segmentation_map = unet_predictor.predict(image_path, fluid_only=True)

                final_mask = unet_predictor.get_final_mask_for_interactivity(segmentation_map, fluid_only=True)

                seg_map_img = unet_predictor.get_segmentation_map_image(segmentation_map)
                seg_map_buf = io.BytesIO()
                seg_map_img.save(seg_map_buf, format="PNG")
                seg_map_buf.seek(0)
                encoded_seg_map = b64encode(seg_map_buf.getvalue()).decode("utf-8")

                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                buf.seek(0)
                encoded_image = b64encode(buf.getvalue()).decode("utf-8")

                # Add class information
                class_info = [
                    {"id": 0, "name": "Background", "color": [0, 0, 0]},
                    {"id": "fluid", "name": "Fluid", "color": [0, 255, 255]}
                ]

                return JsonResponse({
                    "segmented_image": encoded_image,
                    "predicted_annotations": predicted_points,
                    "final_mask": final_mask,
                    "segmentation_map": encoded_seg_map,
                    "class_info": class_info,
                })

            # Handle MedSAM
            elif algorithm == "medsam":
                # MedSAM uses its own API endpoint, redirect the frontend
                return JsonResponse({
                    "error": "MedSAM should use /api/medsam/segment endpoint",
                    "redirect_to": "/api/medsam/segment"
                }, status=400)

            # Handle KITE or Traditional method
            elif algorithm == "kite" or segmentation_method == "traditional":
                for shape in data["shapes"]:
                    points = shape.get("points", [])

                image = imread(image_path, as_gray=True)
                # Create a fresh segmentation model for each request to avoid state conflicts
                fresh_segmentation_model = SegmentationModel(preprocessor)
                result_img = fresh_segmentation_model.run_segmentation_from_json_without_ground_truth(image, data)
                predicted_points = fresh_segmentation_model.get_predicted_points()
                final_mask = fresh_segmentation_model.get_final_mask()

                filename_base = os.path.splitext(os.path.basename(image_path))[0]
                segmentation_masks = fresh_segmentation_model.get_segmentation_masks(filename)

                # Ensures that the original image is returned when there are no annotations.
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                buf.seek(0)
                encoded_image = b64encode(buf.getvalue()).decode("utf-8")

                return JsonResponse({
                    "segmented_image": encoded_image,
                    "predicted_annotations": predicted_points,
                    "final_mask": final_mask,
                    "segmentation_masks": segmentation_masks
                })

            else:
                return JsonResponse({"error": f"Unknown algorithm/method: {algorithm}"}, status=400)

        except Exception as e:
            import traceback
            traceback.print_exc()  # Shows full error in terminal
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'Only a POST request is allowed'}, status=405)

def preprocessed_image_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            filename = data.get("image_name")
            if not filename:
                return JsonResponse({"error": "No image name provided."}, status=400)

            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(image_path):
                return JsonResponse({"error": "Image file not found."}, status=404)

            image = imread(image_path, as_gray=True)
            result_img, fluid_mask = preprocessor.preprocess_image(image)
            result_img_scaled = (fluid_mask.astype(np.uint8)) * 255
            result_pil = Image.fromarray(result_img_scaled)
            buf = BytesIO()
            result_pil.save(buf, format="PNG")
            buf.seek(0)
            encoded_image = b64encode(buf.getvalue()).decode("utf-8")

            return JsonResponse({
                "preprocessed_image": encoded_image
            })

        except Exception as e:
            traceback_str = traceback.format_exc()
            print("Preprocessing error:\n", traceback_str)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'Only a POST request is allowed'}, status=405)

def process_with_unet(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            filename = data.get("image_name")

            if not unet_predictor:
                return JsonResponse({"error": "UNet model is not available."}, status=503)

            if not filename:
                return JsonResponse({"error": "No image name provided."}, status=400)

            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(image_path):
                return JsonResponse({"error": "Image file not found."}, status=404)

            result_img, predicted_points, segmentation_map = unet_predictor.predict(image_path, fluid_only=True)

            final_mask = unet_predictor.get_final_mask_for_interactivity(segmentation_map, fluid_only=True)

            seg_map_img = unet_predictor.get_segmentation_map_image(segmentation_map)
            seg_map_buf = io.BytesIO()
            seg_map_img.save(seg_map_buf, format="PNG")
            seg_map_buf.seek(0)
            encoded_seg_map = b64encode(seg_map_buf.getvalue()).decode("utf-8")

            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            buf.seek(0)
            encoded_image = b64encode(buf.getvalue()).decode("utf-8")

            return JsonResponse({
                "segmented_image": encoded_image,
                "predicted_annotations": predicted_points,
                "final_mask": final_mask,
                "segmentation_map": encoded_seg_map,
            })

        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"UNet processing error:\n{traceback_str}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'Only a POST request is allowed'}, status=405)

# Combines the selected masks into a single mask, preparing it in a .npy file.
def bulk_download_masks(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            selected_masks = data.get("selected_masks", [])
            filename = data.get("image_name", "")

            if not selected_masks:
                return JsonResponse({"error": "No masks selected"}, status=400)

            combined_mask_data = create_combined_mask_array(selected_masks)

            segmentation_dir = os.path.join(settings.MEDIA_ROOT, "segmentations")
            os.makedirs(segmentation_dir, exist_ok=True)

            filename_base = os.path.splitext(filename)[0] if filename else "combined"
            combined_filename = f"{filename_base}_combined_selected_masks.npy"
            combined_path = os.path.join(segmentation_dir, combined_filename)

            np.save(combined_path, combined_mask_data)

            download_url = f"{settings.MEDIA_URL}segmentations/{combined_filename}"
            return JsonResponse({
                "download_url": download_url,
                "filename": combined_filename
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

# This function is a helper function to create a combined mask array, using incremental numbering!!
# Basically, we tag the background as "0", then each region is marked with a unique number starting from 1.
def create_combined_mask_array(selected_masks):
    """Create a combined mask array with incrementally numbered regions"""
    all_pixels = []
    for mask_data in selected_masks:
        all_pixels.extend(mask_data["pixels"])

    if not all_pixels:
        return np.array([])

    max_y = max(pixel[0] for pixel in all_pixels) + 1
    max_x = max(pixel[1] for pixel in all_pixels) + 1

    combined_mask = np.zeros((max_y, max_x), dtype=np.uint8)

    # Fill with incrementally numbered regions.
    for region_number, mask_data in enumerate(selected_masks, start=1):
        pixels = mask_data["pixels"]
        for pixel in pixels:
            y, x = pixel[0], pixel[1]
            if 0 <= y < max_y and 0 <= x < max_x:
                combined_mask[y, x] = region_number

    return combined_mask


def load_annotations(request):
    if request.method == "POST":
        try:
            if 'annotation_file' in request.FILES:
                annotation_file = request.FILES['annotation_file']
                image_name = request.POST.get('image_name', '') 
                
                print(f"Loading annotation file: {annotation_file.name}")
                print(f"For image: {image_name}")
                
                # Verify it's a .npy file.
                if not annotation_file.name.endswith('.npy'):
                    return JsonResponse({"error": "Only .npy files are supported"}, status=400)
                
                # Read the .npy file.
                annotation_data = np.load(annotation_file, allow_pickle=True)
                print(f"Loaded annotation data with shape: {annotation_data.shape}")
                print(f"Data type: {annotation_data.dtype}")
                print(f"Unique values: {np.unique(annotation_data)}")
                
                # Process the annotation data and convert to colored overlay.
                colored_annotations = process_annotation_data(annotation_data)
                
                return JsonResponse({
                    "success": True,
                    "annotations": colored_annotations,
                    "message": f"Loaded {len(colored_annotations)} annotation regions"
                })
            else:
                return JsonResponse({"error": "No annotation file provided"}, status=400)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

def process_annotation_data(annotation_data):
    """
    Process .npy annotation data and convert to colored regions
    annotation_data: numpy array where 0=background, 1,2,3...=different layers
    """
    # Define colors for different annotation values
    colors = [
        "#000000",  # 0: Background (black, will be transparent)
        "#FF0000",  # 1: Red
        "#00FF00",  # 2: Green  
        "#0000FF",  # 3: Blue
        "#FFFF00",  # 4: Yellow
        "#FF00FF",  # 5: Magenta
        "#00FFFF",  # 6: Cyan
        "#FFA500",  # 7: Orange
        "#800080",  # 8: Purple
        "#FFC0CB",  # 9: Pink
        "#A52A2A",  # 10: Brown
        "#808080",  # 11: Gray
    ]
    
    annotations = []
    unique_values = np.unique(annotation_data)
    
    print(f"Processing annotation data with shape: {annotation_data.shape}")
    print(f"Unique values found: {unique_values}")
    
    # Process each unique value (skip 0 as it's background!!!)
    for value in unique_values:
        if value == 0:
            continue
            
        mask = (annotation_data == value).astype(np.uint8)
        
        # Find all pixels with this value.
        ys, xs = np.where(mask == 1)
        
        if len(ys) == 0:
            continue
            
        pixels = [[int(y), int(x)] for y, x in zip(ys, xs)]
        
        color_index = int(value) % len(colors)
        color = colors[color_index]
        
        print(f"Found {len(pixels)} pixels for value {value}, assigned color {color}")
        
        annotations.append({
            "regionId": f"loaded_region_{value}",
            "pixels": pixels,
            "color": color,
            "value": int(value)
        })
    
    return annotations

def download_annotations(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            annotations_data = data.get("annotations", [])
            image_dimensions = data.get("image_dimensions", {"width": 512, "height": 224})
            filename = data.get("image_name", "")

            if not annotations_data:
                return JsonResponse({"error": "No annotations to download"}, status=400)

            annotation_mask = create_annotation_mask_array(annotations_data, image_dimensions)

            annotations_dir = os.path.join(settings.MEDIA_ROOT, "annotations")
            os.makedirs(annotations_dir, exist_ok=True)

            filename_base = os.path.splitext(filename)[0] if filename else "annotations"
            annotation_filename = f"{filename_base}_annotations.npy"
            annotation_path = os.path.join(annotations_dir, annotation_filename)

            # Save the annotations as a .npy file (FOR NOW, maybe we can change it to .npz to add the layer name?)
            np.save(annotation_path, annotation_mask)

            download_url = f"{settings.MEDIA_URL}annotations/{annotation_filename}"
            return JsonResponse({
                "download_url": download_url,
                "filename": annotation_filename,
                "shape": annotation_mask.shape,
                "unique_values": np.unique(annotation_mask).tolist()
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)


def create_annotation_mask_array(annotations_data, image_dimensions):
    # Convert to integers to handle float values from frontend.
    width = int(round(image_dimensions["width"]))
    height = int(round(image_dimensions["height"]))
    
    print(f"Creating annotation mask with dimensions: {width} x {height}")
    print(f"Number of layers to process: {len(annotations_data)}")
    
    annotation_mask = np.zeros((height, width), dtype=np.uint8)
    
    sorted_layers = sorted(annotations_data, key=lambda x: x.get("layer_order", 0))
    
    # Process each layer and assign incremental values for different layers.
    for layer_index, layer_data in enumerate(sorted_layers, start=1):
        strokes = layer_data.get("strokes", [])
        print(f"Layer {layer_index}: {len(strokes)} strokes")
        if strokes and all(stroke.get("type") == "dot" for stroke in strokes) and len(strokes) > 10:
            # For the fill tool, render all the dots as filled area
            all_fill_points = []
            for stroke in strokes:
                all_fill_points.extend(stroke.get("points", []))
            
            if all_fill_points:
                render_fill_dots(annotation_mask, all_fill_points, layer_index, width, height)
        else:
            for stroke in strokes:
                stroke_type = stroke.get("type", "line")  # 4 options: line, box, dot, fill!
                points = stroke.get("points", [])
                
                if not points:
                    continue
                    
                if stroke_type == "dot" or len(points) == 1:
                    render_dot(annotation_mask, points[0], layer_index, width, height)
                    
                elif stroke_type == "box":
                    render_box(annotation_mask, points, layer_index, width, height)
                    
                elif stroke_type == "fill":
                    render_filled_polygon(annotation_mask, points, layer_index, width, height)
                    
                else:
                    render_line(annotation_mask, points, layer_index, width, height)
    
    print(f"Annotation mask created with unique values: {np.unique(annotation_mask)}")
    return annotation_mask

# Here I added functions to render different shapes and save them as an annotation. Nornally, each annotation is stored as separate dots and the lines are drawn 
# on the front end. This is why when we load the annotations without this processing step, we would only get dots.

def render_fill_dots(mask, points, layer_value, width, height):
    for point in points:
        x = int(round(point["x"]))
        y = int(round(point["y"]))
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                px, py = x + dx, y + dy
                if 0 <= px < width and 0 <= py < height:
                    mask[py, px] = layer_value


def render_dot(mask, point, layer_value, width, height, radius=1):
    x = int(round(point["x"]))
    y = int(round(point["y"]))

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:  # Circle equation
                px, py = x + dx, y + dy
                if 0 <= px < width and 0 <= py < height:
                    mask[py, px] = layer_value


def render_line(mask, points, layer_value, width, height, line_width=1):
    if len(points) < 2:
        return
        
    for i in range(len(points) - 1):
        x1 = int(round(points[i]["x"]))
        y1 = int(round(points[i]["y"]))
        x2 = int(round(points[i + 1]["x"]))
        y2 = int(round(points[i + 1]["y"]))
        
        draw_line(mask, x1, y1, x2, y2, layer_value, width, height, line_width)


def render_box(mask, points, layer_value, width, height, line_width=1):
    if len(points) < 4:
        return
        
    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        x1 = int(round(points[i]["x"]))
        y1 = int(round(points[i]["y"]))
        x2 = int(round(points[next_i]["x"]))
        y2 = int(round(points[next_i]["y"]))
        
        draw_line(mask, x1, y1, x2, y2, layer_value, width, height, line_width)


def render_filled_polygon(mask, points, layer_value, width, height):
    """Render a filled polygon using scanline algorithm"""
    if len(points) < 3:
        return
        
    polygon_points = [(int(round(p["x"])), int(round(p["y"]))) for p in points]
    
    # Find bounding box.
    min_y = max(0, min(p[1] for p in polygon_points))
    max_y = min(height - 1, max(p[1] for p in polygon_points))
    
    for y in range(min_y, max_y + 1):
        intersections = []
        
        # Find intersections with polygon edges.
        for i in range(len(polygon_points)):
            j = (i + 1) % len(polygon_points)
            x1, y1 = polygon_points[i]
            x2, y2 = polygon_points[j]
            
            if y1 <= y < y2 or y2 <= y < y1:
                if y2 != y1:  # Avoid division by zero here!
                    x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(x_intersect)
        
        intersections.sort()
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = max(0, int(intersections[i]))
                x_end = min(width - 1, int(intersections[i + 1]))
                for x in range(x_start, x_end + 1):
                    mask[y, x] = layer_value

# NOTE TO DURU:
# The lines are drawn a bit thicker than usual, I feel like the issue is somewhere here.
def draw_line(mask, x1, y1, x2, y2, layer_value, width, height, thickness=1):
    # Bresenham's line algorithm:
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    
    radius = thickness // 2
    
    while True:
        for dy_offset in range(-radius, radius + 1):
            for dx_offset in range(-radius, radius + 1):
                if dx_offset * dx_offset + dy_offset * dy_offset <= radius * radius:
                    px = x + dx_offset
                    py = y + dy_offset
                    if 0 <= px < width and 0 <= py < height:
                        mask[py, px] = layer_value
        
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy