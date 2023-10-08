"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('polls.urls')),
]
# you can use this to serve media files mean /media/{file} you can load any file save in media folder specified in MEDIA_URL and root
urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
# you can use this to serve static files mean /static/{file} you can load any file save in static folder specified in MEDIA_URL and root

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
