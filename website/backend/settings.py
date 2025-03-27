from pathlib import Path
import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

#SECRET_KEY = 'django-insecure-your-secret-key'
#DEBUG = True

ALLOWED_HOSTS = []

### for Render
ON_RENDER = os.environ.get('RENDER') == 'TRUE'

SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    'django-insecure-please-change-this-for-local-dev'
)

DEBUG = not ON_RENDER

if ON_RENDER:
    ALLOWED_HOSTS.append('kite-2qf9.onrender.com')
else:
    ALLOWED_HOSTS += ['localhost', '127.0.0.1']
###

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'frontend.pages',
    'images',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# for Render
if not DEBUG:
    MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')

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

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "css",
    BASE_DIR / "frontend/static"
]

STATIC_ROOT = BASE_DIR / "staticfiles" # for Render

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'backend/media')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


### cloudinary
INSTALLED_APPS += [
    'cloudinary',
    'cloudinary_storage',
]

DEFAULT_FILE_STORAGE = 'cloudinary_storage.storage.MediaCloudinaryStorage'

CLOUDINARY_STORAGE = {
    'CLOUD_NAME': os.environ.get('CLOUDINARY_CLOUD_NAME'),
    'API_KEY': os.environ.get('CLOUDINARY_API_KEY'),
    'API_SECRET': os.environ.get('CLOUDINARY_API_SECRET'),
}

###