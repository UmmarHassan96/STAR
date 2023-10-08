# qa/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User  # Import the User model

class PromptForm(forms.Form):
    prompt = forms.CharField(label='Enter Prompt', max_length=200)


class SignupForm(UserCreationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Customize help texts or remove them as needed
        self.fields['username'].help_text = None  # Remove help text for username
        self.fields['password1'].help_text = None  # Remove help text for password1
        self.fields['password2'].help_text = None  # Remove help text for password2

    class Meta:
        model = User
        fields = ['username', 'password1', 'password2']  # Add other fields as needed


