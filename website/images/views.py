from django.shortcuts import render
from .models import Photo

def seg_tool(request):
    image_url = None
    image_name = None

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        photo = Photo.objects.create(image=image, title=image.name)
        image_url = photo.image.url
        image_name = photo.title

    return render(request, 'pages/segtool.html', {
        'image_url': image_url,
        'image_name': image_name
    })
