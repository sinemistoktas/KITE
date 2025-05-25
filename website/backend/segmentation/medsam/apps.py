# website/backend/segmentation/medsam/apps.py
"""
Django app configuration for MedSAM backend.
"""

from django.apps import AppConfig

class MedsamConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backend.segmentation.medsam'
    verbose_name = 'MedSAM Segmentation'