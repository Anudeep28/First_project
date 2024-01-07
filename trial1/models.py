from django.db import models
from django.contrib.auth.models import User
# Create your models here. but they are not registered in database
# to create the table in sql we will have to do migration in django
# to add this table first run 'python manage.py makemigrations'
# then say 'python manage.py migrate' to finally create it
class Item(models.Model):
    name = models.CharField(max_length=128)
    price = models.IntegerField()
    description = models.CharField(max_length=300)
    image_url = models.CharField(max_length=512)
    # foreign key gives info about ownership who owns this product
    # default so that when you update this field into db it knows what to fill it
    # initially
    # blank tells django it is fine if field is blank
    owner = models.ForeignKey(User, default=None, blank=True,\
                              on_delete=models.SET_NULL, null=True)

    # Method to show the string within the models
    def __str__(self):
        return self.name

class appoptions(models.Model):
    # name of the apps present for use in the App
    name = models.CharField(max_length=128)
    owner = models.ForeignKey(User, default=None, blank=True,\
                              on_delete=models.SET_NULL, null=True)

    # method to show string within the models
    def __str__(self):
        return self.name
