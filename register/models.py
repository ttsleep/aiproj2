from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    username = models.CharField(max_length=16, unique=True, blank=False)
    password = models.CharField(max_length=30, unique=False, blank=False)

    class Meta:
        # tb_table = 'users'
        verbose_name = '用户管理'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.username

