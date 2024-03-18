from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('about.html', views.about, name="about"),
    path('index.html', views.home, name="home"),
    path('blog.html', views.blog, name="blog"),
    path('contact.html', views.contact, name="contact"),
    path('watch.html', views.watch, name="watch"),
    path('track', views.track, name="func"),


]
