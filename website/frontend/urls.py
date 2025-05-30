# website/frontend/urls.py

from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('segtool/', views.seg_tool, name='seg_tool'),
    path('segment/', views.segment_image, name='segment'),
    path("preprocessed-image/", views.preprocessed_image_view, name="preprocessed_image"),
    path('process-with-unet/', views.process_with_unet, name='process_with_unet'),
    path('bulk-download-masks/', views.bulk_download_masks, name='bulk_download_masks'),
    path('load-annotations/', views.load_annotations, name='load_annotations'),
    path('download-annotations/', views.download_annotations, name='download_annotations')
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)