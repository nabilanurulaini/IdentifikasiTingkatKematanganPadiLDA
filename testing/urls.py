from django.urls import re_path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    re_path(r'', views.index, name='testing'),
    re_path(r'testresult/', views.testing, name='testing'),
]
