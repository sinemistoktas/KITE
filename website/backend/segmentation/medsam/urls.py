# website/backend/segmentation/medsam/urls.py
"""
MedSAM URL Configuration
Simplified URL patterns with batch support as default.
"""

from django.urls import path
from . import views

app_name = 'medsam'

urlpatterns = [
    # Main segmentation endpoint (supports both single and batch)
    path('segment', views.segment_image, name='segment'),
    
    # Health check
    path('health', views.health_check, name='health'),

    path('download-npy/', views.download_npy_mask, name='download_npy_mask'),
]