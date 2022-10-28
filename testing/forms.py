from django import forms
from django.forms import ClearableFileInput
from .models import Multiple

class UploadImage(forms.ModelForm):
    class Meta:
        model = Multiple
        fields = ['image']
        widgets = {
            'image': ClearableFileInput(attrs={'multiple': True}),
        }