from django.urls import path
from . import views


urlpatterns = [
    path('', views.DetectionView.as_view(), name='detection'),
    path('logout/', views.LogOut.as_view(), name='logout'),
    path('imag_result/', views.ImagResult.as_view(), name='result'),
    path('video_result/', views.VideoResult.as_view(), name='result')
]