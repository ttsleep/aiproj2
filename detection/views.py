from django.shortcuts import render, redirect, HttpResponse
from django.views import View
from django.contrib.auth import logout
from django.contrib.auth.mixins import LoginRequiredMixin
import cv2
from PIL import Image
import numpy as np
import os,django
from yolov5_traffic.detect_one import detect_one,detect_vedio
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project_name.settings")
django.setup()

save_path_1 = 'static/detected_pics/output_a1_1.png'
save_videopath_1 = 'static/video_path/output_a1_v.mp4'


class DetectionView(LoginRequiredMixin, View):

    def get(self, request):
        return render(request, 'detection.html')

    def post(self, request):
        pic_object = request.FILES.get('before_process')

        if pic_object is None:
            return render(request, 'detection.html')

        if pic_object.name[-4:] == '.mp4':
            img = 'static/video_path/ori.mp4'
            f = open(img, mode='wb')
            for chunk in pic_object.chunks():
                f.write(chunk)
            f.close()
            detect_vedio(img,video_save_path=save_videopath_1)
            return redirect('/detection/video_result')

        if pic_object.name[-4:] == '.png' or '.jpg':
            img = 'static/detected_pics/a1.png'
            f = open(img, mode='wb')
            for chunk in pic_object.chunks():
                f.write(chunk)
            f.close()
            image1=detect_one(img)
            cv2.imwrite(save_path_1, image1)
            return redirect('/detection/imag_result')


class LogOut(View):
    def get(self, request):
        logout(request)

        response = redirect('/')
        response.delete_cookie('is_login')
        response.delete_cookie('username')
        return response


class ImagResult(View):
    def get(self, request):
        return render(request, 'imag_result.html')


class VideoResult(View):
    def get(self, request):
        return render(request, 'video_result.html')

