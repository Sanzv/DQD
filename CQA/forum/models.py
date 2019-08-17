from django.db import models


# Create your models here.

class Questions(models.Model):
    question = models.CharField(max_length=150)
    answer = models.TextField(max_length=500, default="")

    def __str__(self):
        return 'Question:   ' + str(self.pk)
