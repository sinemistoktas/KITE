from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('segtool/', views.seg_tool, name='seg_tool'),
]