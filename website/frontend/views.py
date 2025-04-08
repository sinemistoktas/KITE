from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from data.data_preprocessing import Preprocessor
from data.data_segmentation import SegmentationModel
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import json
import io
import os
from skimage.io import imread

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
            filename = data.get("image_name")

            if not filename:
                return JsonResponse({"error": "No image name provided."}, status= 400)
            
            image_path = os.path.join(settings.MEDIA_ROOT, filename)

            if not os.path.exists(image_path):
                return JsonResponse({"error": "Image file not found."}, status= 404)
            
            image = imread(image_path, as_gray= True)
            fig = segmentation_model.run_segmentation_from_json_without_ground_truth(image, data)

            buf = io.BytesIO()
            canvas = FigureCanvas(fig)
            canvas.print_png(buf)
            plt.close(fig)

            return HttpResponse(buf.getvalue(), content_type = "image/png")
        
        except Exception as e:
            return JsonResponse({"error": "Only a POST request is allowed."}, status= 405)

