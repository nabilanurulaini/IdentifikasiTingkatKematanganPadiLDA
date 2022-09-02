from asyncore import read
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Create your views here.


def index(request):
    context = {
        'title': 'Deteksi - LDA',
        'heading': 'Deteksi',
    }
    return render(request, 'testing/index.html', context)


def testing(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']

        fs = FileSystemStorage()
        mdl = "model.sav"
        best_model = pickle.load(open(mdl, "rb"))
        filename = fs.save(image.name, image)
        image_uploaded = fs.path(filename)

        data = []
        frame = cv2.imread(image_uploaded)
        frame = cv2.resize(frame, (336, 448))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        directory = r'C:\Django\skripsi\skripsi\static\img'
        gambarasli = "asli-{}".format(image.name)
        cv2.imwrite(os.path.join(directory, gambarasli), frame)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        H, S, V = cv2. split(hsv)
        # mendefinisikan clahe atau metode histogram ewualization yang dipakai, tile grid size 8,8 merupakan default sie dari clahe
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # menerapkan clahe pada value
        equalized_V = clahe.apply(V)
        # melakukan penggabungan anara h,s, dan value yang telah diequalized
        equalized = cv2.merge([H, S, equalized_V])
        # Mendefinisikan warna yang akan di cari pada HSV
        lower_green = np.array([16, 43, 40])
        upper_green = np.array([179, 255, 255])
        lower_brown = np.array([4, 88, 43])
        upper_brown = np.array([9, 243, 255])
        # Proses masking
        mask = cv2.inRange(equalized, lower_green, upper_green)
        mask2 = cv2.inRange(equalized, lower_brown, upper_brown)
        final_mask = mask + mask2
        # Memisahkan citra foreground dengan citra background
        # Bitwise-AND mask dan gambar asli
        image = cv2.bitwise_and(equalized, equalized, mask=final_mask)
        # Final processing dengan memberikan sedikit blur
        image = cv2.GaussianBlur(image, (3, 3), 0)

        try:
            features = cv2.Canny(image, 100, 200)
            features = np.reshape(features, (448*336))
            #result = features
            directory = r'C:\Django\skripsi\skripsi\static\img'
            preprocessing = "result-{}".format(image.name)
            cv2.imwrite(os.path.join(directory, preprocessing), features)

            data.append([features, 0])
        except Exception as e:
            print(e)
        features = []
        labels = []
        for feature, label in data:
            features.append(feature)
            labels.append(label)
        best_model = np.squeeze
        prediction = best_model.predict(features)

    context = {
        'title': 'Hasil Deteksi - LDA',
        'heading': 'Hasil Deteksi',
        'hasil': "Matang" if prediction == 0 else "Mentah",
        'preprocessing': preprocessing,
        'gambarasli': gambarasli,
    }
    return render(request, 'testing/testresult.html', context)
