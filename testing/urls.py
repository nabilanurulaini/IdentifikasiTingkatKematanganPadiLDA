from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'', views.index, name='testing'),
    re_path(r'testresult/', views.testing, name='testing'),
]