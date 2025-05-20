from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('segtool/', views.seg_tool, name='seg_tool'),
    path('segment/', views.segment_image, name='segment'),
    path("preprocessed-image/", views.preprocessed_image_view, name="preprocessed_image"),
    path('process-with-unet/', views.process_with_unet, name='process_with_unet'),

]
