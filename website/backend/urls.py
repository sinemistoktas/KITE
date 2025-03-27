from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
import os

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('frontend.urls')),
]

if settings.DEBUG or os.environ.get('RENDER') == 'TRUE':
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)