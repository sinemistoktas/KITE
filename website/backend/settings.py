# website/backend/settings.py

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

#SECRET_KEY = 'django-insecure-your-secret-key'
#DEBUG = True

# for Render

SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    'django-insecure-please-change-this-for-local-dev'
)
DEBUG = os.environ.get('DEBUG', 'True') == 'True'


ALLOWED_HOSTS = [
    'kite-2qf9.onrender.com',
    'localhost',
    '127.0.0.1'
]
#

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'frontend.pages',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware', # render
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "frontend/"],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

# Database (default SQLite)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = []

# Language/timezone
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static & Media
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "frontend/static"
]

# MedSAM model path settings
MEDSAM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'medsam_vit_b.pth')

STATIC_ROOT = BASE_DIR / "staticfiles" # for Render

MEDIA_URL = 'backend/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'backend/media')
os.makedirs(os.path.join(MEDIA_ROOT, 'medsam_results'), exist_ok=True)

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'