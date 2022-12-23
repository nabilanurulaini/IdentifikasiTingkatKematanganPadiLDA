from asyncore import read
from cv2 import equalizeHist
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
        'title': 'LDA',
        'heading': 'Identify',
    }
    return render(request, 'testing/index.html', context)


def testing(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']

        fs = FileSystemStorage()
        directory = r'C:\Django\skripsi\skripsi\static\img'
        model = 'C:\Django\skripsi\skripsi\model.sav'
        best_model = pickle.load(open(model, "rb"))

        #prediction = best_model.predict(mdl)
        # print(prediction)
        filename = fs.save(image.name, image)
        image_uploaded = fs.path(filename)

        data = []
        frame = cv2.imread(image_uploaded)
        frame = cv2.resize(frame, (448, 336))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gambarasli = "asli-{}".format(filename)
        cv2.imwrite(os.path.join(directory, gambarasli), frame)
        array_asli = cv2.mean(frame)[:3]
        r = array_asli[0]
        g = array_asli[1]
        b = array_asli[2]

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gambar_hsv = "HSV-{}".format(filename)
        cv2.imwrite(os.path.join(directory, gambar_hsv), hsv)
        array_hsv = cv2.mean(hsv)[:3]
        h = array_hsv[0]
        s = array_hsv[1]
        v = array_hsv[2]

        H, S, V = cv2. split(hsv)
        # mendefinisikan clahe atau metode histogram ewualization yang dipakai, tile grid size 8,8 merupakan default sie dari clahe
        clahe = cv2.createCLAHE(clipLimit=1.0)
        # menerapkan clahe pada value
        equalized_V = clahe.apply(V)
        # melakukan penggabungan anara h,s, dan value yang telah diequalized
        equalized = cv2.merge([H, S, equalized_V])
        array_eq = equalized
        equalizedRGB = cv2.cvtColor(equalized, cv2.COLOR_HSV2BGR)
        equalizeHist = "equalize-{}".format(filename)
        cv2.imwrite(os.path.join(directory, equalizeHist), equalizedRGB)
        array_equalized = cv2.mean(array_eq)[:3]
        equalizedH = array_equalized[0]
        equalizedS = array_equalized[1]
        equalizedV = array_equalized[2]
        # Mendefinisikan warna yang akan di cari pada HSV
        lower_green = np.array([16, 43, 40])
        upper_green = np.array([179, 255, 255])
        lower_brown = np.array([4, 88, 43])
        upper_brown = np.array([25, 243, 255])
        # Proses masking
        mask = cv2.inRange(equalized, lower_green, upper_green)
        mask2 = cv2.inRange(equalized, lower_brown, upper_brown)
        final_mask = mask + mask2
        # Memisahkan citra foreground dengan citra background
        # Bitwise-AND mask dan gambar asli
        image = cv2.bitwise_and(equalized, equalized, mask=final_mask)
        # Final processing dengan memberikan sedikit blur
        #image = cv2.GaussianBlur(image, (3, 3), 0)

        result = image
        result = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        preprocessing = "result-{}".format(filename)
        cv2.imwrite(os.path.join(directory, preprocessing), result)
        array_after = cv2.mean(image)[:3]
        afterH = array_after[0]
        afterS = array_after[1]
        afterV = array_after[2]

        features = cv2.mean(image)[:3]
        data.append([features, 0])
        print(data)
        features = []
        labels = []
        for feature, label in data:
            features.append(feature)
            labels.append(label)

        prediction = best_model.predict(features)
        print(prediction)
        print(label)

    context = {
        'title': 'Hasil Deteksi - LDA',
        'heading': 'Result',
        'hasil': "Matang" if prediction == 0 else "Mentah",
        'preprocessing': preprocessing,
        'array_after': array_after,
        'gambarasli': gambarasli,
        'array_asli': array_asli,
        'gambar_hsv': gambar_hsv,
        'array_hsv': array_hsv,
        'r': r,
        'g': g,
        'b': b,
        'h': h,
        's': s,
        'v': v,
        'equalizedH': equalizedH,
        'equalizedS': equalizedS,
        'equalizedV': equalizedV,
        'afterH': afterH,
        'afterS': afterS,
        'afterV': afterV,
        'equalizeHist': equalizeHist,
        'array_equalized': array_equalized,
        'features': features,
        'prediction': prediction
    }
    return render(request, 'testing/testresult.html', context)
