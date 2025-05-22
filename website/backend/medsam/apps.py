# website/backend/medsam/apps.py
"""
Django app configuration for MedSAM backend.
"""

from django.apps import AppConfig

class MedsamConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backend.medsam'
    verbose_name = 'MedSAM Segmentation'