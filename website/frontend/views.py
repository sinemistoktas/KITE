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

    def predict(self, img_path):
        img = Image.open(img_path)
        original_size = img.size

        img = img.resize((512, 224))
        if img.mode != 'L':
            img = img.convert('L')

        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)

        _, predicted = torch.max(output, 1)
        segmentation_map = predicted.squeeze().cpu().numpy()

        original_image = Image.open(img_path).convert('RGB')
        original_array = np.array(original_image.resize((512, 224)))

        # Define colors for each class (RGB)
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
        
        # Generate polygons for each class
        predicted_points = []
        
        # Process each class except background (class 0)
        for class_idx in range(1, min(10, len(colors))):
            binary_mask = (segmentation_map == class_idx).astype(np.uint8)
            
            if np.sum(binary_mask) == 0:
                continue
                
            class_mask = np.zeros_like(original_array)
            class_mask[binary_mask > 0] = colors[class_idx]
            overlay = cv.addWeighted(overlay, 1.0, class_mask, 0.5, 0)
            
            # Find contours for this class
            contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # Convert contours to points
            for contour in contours:
                area = cv.contourArea(contour)
                if area < 10:
                    continue
                    
                contour_points = []
                for point in contour:
                    x, y = point[0]
                    contour_points.append([float(x), float(y)])
                
                if len(contour_points) > 2:  
                    predicted_points.append({
                        "shape_type": "polygon",
                        "points": contour_points,
                        "color": colors[class_idx],  
                        "class_id": class_idx 
                    })
        
        result_img = Image.fromarray(overlay)
        
        return result_img, predicted_points

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
unet_model_path = '/Users/yamacomur/Desktop/KITE/unet/notebooks/unet_traced.pt'

print("UNet model path:", unet_model_path)
print("Exists:", os.path.exists(unet_model_path))
unet_predictor = None

try:
    if os.path.exists(unet_model_path):
        unet_predictor = UnetPredictor(unet_model_path)
        print("UNet model loaded successfully")
    else:
        print("UNet model path not found ")
except Exception as e:
    print(f"UNet model loading failed: {e}")
def home(request):
    return render(request, 'pages/home.html')

def seg_tool(request):
    image_url = None
    image_name = None
    segmentation_tool = None
    segmentation_method = request.POST.get('segmentation_method', 'traditional')  if request.method == 'POST' else 'traditional'

    context = {
        'image_url': image_url,
        'image_name': image_name,
        'segmentation_method': segmentation_method,
        'unet_available': unet_predictor is not None
    }

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_name = image.name
        fs = FileSystemStorage()
        filename = fs.save(image_name, image)
        image_url = fs.url(filename)
        segmentation_method = request.POST.get('segmentation_method', 'traditional')

        context = {
            'image_url': image_url,
            'image_name': image_name,
            'segmentation_method': segmentation_method,
            'unet_available': unet_predictor is not None
        }

        if segmentation_method == 'unet' and unet_predictor is not None:
            try:
                image_path = os.path.join(settings.MEDIA_ROOT, filename)
                result_img, predicted_points = unet_predictor.predict(image_path)
                print("UNet predicted points:", predicted_points)
                print("Image size:", result_img.size)
                result_filename = f"{os.path.splitext(filename)[0]}_unet_segmented.png"
                result_path = os.path.join(settings.MEDIA_ROOT, result_filename)
                result_img.save(result_path)

                context['segmented_image_url'] = fs.url(result_filename)
                context['show_segmentation_result'] = True
                context['unet_mode'] = True
                context['predicted_points'] = json.dumps(predicted_points)
            except Exception as e:
                traceback_str = traceback.format_exc()
                print(f"UNet processing error:\n{traceback_str}")
                context['unet_error'] = str(e)
        else:
            context['unet_mode'] = False

    return render(request, 'pages/segtool.html', context)

# Takes an HTTP request, which must include the name of the image uploaded, and the json formatted annotated file.
# The function then applies segmentation on the image, and returns the segmented image in a PNG format.
#TODO: Decide if this should be moved to the back-end or not.
def segment_image(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            use_unet = data.get("use_unet", False)

            filename = data.get("image_name")
            if not filename:
                return JsonResponse({"error": "No image name provided."}, status=400)

            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(image_path):
                return JsonResponse({"error": "Image file not found."}, status=404)

            if use_unet and unet_predictor is not None:
                result_img, predicted_points = unet_predictor.predict(image_path)

                # Ensures that the original image is returned when there are no annotations.
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                buf.seek(0)
                encoded_image = b64encode(buf.getvalue()).decode("utf-8")
                
                # Add class information
                class_info = [
                    {"id": 0, "name": "Background", "color": [0, 0, 0]},
                    {"id": 1, "name": "Layer 1", "color": [255, 0, 0]},
                    {"id": 2, "name": "Layer 2", "color": [0, 255, 0]},
                    {"id": 3, "name": "Layer 3", "color": [0, 0, 255]},
                    {"id": 4, "name": "Layer 4", "color": [255, 255, 0]},
                    {"id": 5, "name": "Layer 5", "color": [255, 0, 255]},
                    {"id": 6, "name": "Layer 6", "color": [0, 255, 255]},
                    {"id": 7, "name": "Layer 7", "color": [128, 0, 0]},
                    {"id": 8, "name": "Layer 8", "color": [0, 128, 0]},
                    {"id": 9, "name": "Layer 9", "color": [0, 0, 128]}
                ]

                return JsonResponse({
                    "segmented_image": encoded_image,
                    "predicted_annotations": predicted_points,
                    "class_info": class_info
                })


            #Traditional Method

            for shape in data["shapes"]:
                points = shape.get("points", [])
            #print("Data", data) // for debugging


            image = imread(image_path, as_gray= True)
            result_img = segmentation_model.run_segmentation_from_json_without_ground_truth(image, data)
            predicted_points = segmentation_model.get_predicted_points()

            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            buf.seek(0)
            encoded_image = b64encode(buf.getvalue()).decode("utf-8")

            return JsonResponse({
                "segmented_image": encoded_image, # will be the original image if there are no annotations
                "predicted_annotations": predicted_points,  # will be [] if no annotations
            })

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

                result_img, predicted_points = unet_predictor.predict(image_path)

                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                buf.seek(0)
                encoded_image = b64encode(buf.getvalue()).decode("utf-8")

                return JsonResponse({
                    "segmented_image": encoded_image,
                    "predicted_annotations": predicted_points,
                })

            except Exception as e:
                traceback_str = traceback.format_exc()
                print(f"UNet processing error:\n{traceback_str}")
                return JsonResponse({"error": str(e)}, status=500)

        return JsonResponse({'error': 'Only a POST request is allowed'}, status=405)
