from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

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

