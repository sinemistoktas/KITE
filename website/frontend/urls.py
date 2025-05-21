from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('segtool/', views.seg_tool, name='seg_tool'),
    path('segment/', views.segment_image, name='segment'),
    path("preprocessed-image/", views.preprocessed_image_view, name="preprocessed_image"),

]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
