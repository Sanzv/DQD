# Generated by Django 2.2.2 on 2019-07-28 07:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('forum', '0002_auto_20190728_1058'),
    ]

    operations = [
        migrations.AlterField(
            model_name='questions',
            name='answer',
            field=models.TextField(blank=True, default='', max_length=500),
        ),
    ]
