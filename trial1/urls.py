# This file is mainly to organise the routes between views
# this file has been included in project urls.py file
from django.urls import path
from . import views

# the names are used on href in the html very important for routing
urlpatterns = [
    path('',views.homepage, name="home"),
    path('items/', views.itemspage, name='items'),
    path('NameGenerator/',views.NameGenerator, name='NameGenerator'),
    path('login/', views.loginpage, name='login'),
    path('logout/', views.logoutpage, name='logout'),
    path('register/', views.registerpage, name='register')

]