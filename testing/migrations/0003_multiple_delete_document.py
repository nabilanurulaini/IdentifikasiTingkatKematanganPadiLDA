# Generated by Django 4.1 on 2022-10-20 08:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('testing', '0002_document_delete_multipleimage'),
    ]

    operations = [
        migrations.CreateModel(
            name='Multiple',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='')),
            ],
        ),
        migrations.DeleteModel(
            name='Document',
        ),
    ]