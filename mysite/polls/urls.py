# qa/urls.py
from django.urls import path
from . import views
from django.contrib.auth.views import LoginView, LogoutView

urlpatterns = [
    path('', views.get_answer, name='get_answer'),
    path('custom_login/', views.custom_login_view, name='custom_login'),
    path('custom_logout/', views.custom_logout_view, name='custom_logout'),
    path('signup/', views.register, name='signup'),

]
