from django.urls import path
from backend import views
from images.views import seg_tool

urlpatterns = [
    path('', views.home, name='home'),
    path('seg-tool/', seg_tool, name='seg_tool'),
]
