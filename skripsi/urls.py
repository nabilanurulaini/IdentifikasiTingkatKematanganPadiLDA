"""skripsi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from django.urls import re_path, include
from . import views
from training.views import training as trainresult
from testing.views import testing as testresult

urlpatterns = [
    re_path(r'^admin/', admin.site.urls),
    re_path(r'^$', views.index, name='home'),
    re_path(r'^training/', include('training.urls')),
    re_path(r'^trainresult/$', trainresult),
    re_path(r'^testing/', include('testing.urls')),
    re_path(r'^testresult/$', testresult)
] 
#note : always use include() when u include url from another app such as training, testing etc