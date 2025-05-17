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

preprocessor = Preprocessor()
segmentation_model = SegmentationModel(preprocessor)

def home(request):
    return render(request, 'pages/home.html')

def seg_tool(request):
    image_url = None
    image_name = None

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_name = image.name
        fs = FileSystemStorage()
        filename = fs.save(image_name, image)
        image_url = fs.url(filename)
    return render(request, 'pages/segtool.html', {
        'image_url': image_url,
        'image_name': image_name
    })

# Takes an HTTP request, which must include the name of the image uploaded, and the json formatted annotated file.
# The function then applies segmentation on the image, and returns the segmented image in a PNG format.
#TODO: Decide if this should be moved to the back-end or not.
def segment_image(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            for shape in data["shapes"]:
                points = shape.get("points", [])
            #print("Data", data) // for debugging
           
            filename = data.get("image_name")
            if not filename:
                return JsonResponse({"error": "No image name provided."}, status=400)
            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(image_path):
                return JsonResponse({"error": "Image file not found."}, status=404)
            image = imread(image_path, as_gray= True)
            result_img = segmentation_model.run_segmentation_from_json_without_ground_truth(image, data)
            predicted_points = segmentation_model.get_predicted_points()
            final_mask = segmentation_model.get_final_mask()

            # Ensures that the original image is returned when there are no annotations.
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            buf.seek(0)
            encoded_image = b64encode(buf.getvalue()).decode("utf-8")

            return JsonResponse({
                "segmented_image": encoded_image, # will be the original image if there are no annotations
                "predicted_annotations": predicted_points,  # will be [] if no annotations
                "final_mask": final_mask
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