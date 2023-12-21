from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth import authenticate, login
from register.models import User


class HomeView(View):

    def get(self, request):
        return render(request, 'home.html')

    def post(self, request):
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')

        pbbool = User.objects.filter(username=username).exists()
        if not pbbool:
            msg = '用户名不存在'
            return render(request, 'home.html', locals())

        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            response = redirect('/detection')

            response.set_cookie('is_login', True)
            response.set_cookie('username', user.username, max_age=1 * 24* 3600)

            return response
        if user is None:
            msg = '用户名或密码错误'
            return render(request, 'home.html', locals())

