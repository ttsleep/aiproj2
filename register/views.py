from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from .models import User


def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        password_2 = request.POST.get('password_2', '')

        if password != password_2:
            msg = '两次输入密码不一致，请确认密码'
            return render(request, 'register.html', locals())

        elif username == '':
            msg = '用户名不能为空'
            return render(request, 'register.html', locals())

        Pbbool = User.objects.filter(username=username).exists()
        if Pbbool:
            msg = '用户名已存在'
            return render(request, 'register.html', locals())

        User.objects.create_user(username=username, password=password)
        return redirect('/')

    else:
        return render(request, 'register.html', locals())
