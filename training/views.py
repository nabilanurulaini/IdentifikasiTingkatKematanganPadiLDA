from django.shortcuts import render
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import os
import numpy as np
import cv2
import pickle
import random
import numpy as np

# Create your views here..


def index(request):
    context = {
        'title': 'Training - LDA',
        'heading': 'Training',
    }
    return render(request, 'training/index.html', context)
# Fungsi Klasifikasi


def classify(trainX, trainY, testX, testY):
    # The n_components key word gives us the projection to the n most discriminative directions in the dataset. We set this parameter to two to get a transformation in two dimensional space.

    clf = LinearDiscriminantAnalysis()
    clf.fit(trainX, trainY)
    # melakukan predict terhadao test x
    # melakukan predict terhadap train x
    prediction = clf.predict(testX)

    # print(prediction)
    acc = accuracy_score(testY, prediction)
    pre = precision_score(testY, prediction)
    rec = recall_score(testY, prediction)
    cm = confusion_matrix(testY, prediction)

    tn, fp, fn, tp = confusion_matrix(
        list(testY), list(prediction), labels=[0, 1]).ravel()

    print('True Positive', tp)
    print('True Negative', tn)
    print('False Positive', fp)
    print('False Negative', fn)
    return acc, pre, rec, clf, cm
# Awal imageprocessing


def imageprocessing(frame):
    # Meresize Gambar
    frame = cv2.resize(frame, (336, 448))
    # OpenCV akan mengimport gambar kedalam format warna BGR
    # Mengubah Gambar ke format RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mengubah kedalam format HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2. split(hsv)
    # mendefinisikan clahe atau metode histogram ewualization yang dipakai, tile grid size 8,8 merupakan default sie dari clahe
    clahe = cv2.createCLAHE(clipLimit=1.0)
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
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    return image
# Akhir imageprocessing


def training(request):
    if request.method == 'POST':
        path = request.POST['path']
        # Mengimport dataset
        dir = r"{}".format(path)
        categories = ['mentah', 'matang']
        # Menampilkan data kedalam tabel
        dataImage = {}
        listed = []
        # Menyiapkan variabel penampun data
        data = []

        # Dalam setiap kategori (kelas) akan diambil gambar dan dilakukan preprocessing
        i = 1
        for category in categories:
            path = os.path.join(dir, category)
            label = categories.index(category)
            for img in os.listdir(path):
                imgpath = os.path.join(path, img)
                dataImage["No"] = i
                dataImage["Nama"] = str(img)
                dataImage["Path"] = str(imgpath)
                dataImage["Kelas"] = str(category)
                listed.append(dataImage.copy())
                i = i+1
                # Membaca gambar dari directory
                frame = cv2.imread(imgpath)
                # Memanggil fungsi imageprocessing
                image = imageprocessing(frame)

                try:
                    # ekstraksi fitur dengan menggunakan mean HSV
                    # karena cv2.mean mreturns 4 value scalar jadi digunakan :3 untuk mengambil 3 value pertama
                    # Fitur akan disimpan kedalam variabel data, Disimpan bersama dengan label dari setiap kelas
                    # Kelas matang memiliki label (1) mentah (0)
                    features = cv2.mean(image)[:3]
                    data.append([features, label])
                except Exception as e:
                    print(e)
        # Data akan diacak untuk menyebar data
        # random.shuffle(data)
        features = []
        labels = []
        # Mengambil fitur dan label dari data
        for feature, label in data:
            features.append(feature)
            labels.append(label)

        # Inisialisasi variabel penampung hasil uji
        acc = []
        pre = []
        rec = []
        model = []
        cm = []
        # split data menggunakan train test split
        trainX, testX, trainY, testY = train_test_split(
            features, labels, test_size=0.1)

        # cv = KFold(n_splits=4, shuffle=True, random_state=32)
        # # Split data dengan K fold
        # for train_index, test_index in cv2.split(features):
        #      trainX = []
        #      trainY = []
        #      testX = []
        #      testY = []
        #      # Split data menjadi data trainX dan testX untuk data fitur
        #      # Split data menjadi data trainY dan testY untuk data label

        # for i in test_index:
        #          testX.append(features[i])
        #          testY.append(labels[i])
        # print("test index: ", test_index)
        # for i in train_index:
        #          trainX.append(features[i])
        #          trainY.append(labels[i])
        # print("train index: ", train_index)
        # Memanggil fungsi classify

        accu, prec, recl, mdl, confmtrx = classify(
            trainX, trainY, testX, testY)
        print(classify)
        # Menyimpan hasil setiap fold dar pengujian
        acc.append(accu)
        pre.append(prec)
        rec.append(recl)
        model.append(mdl)
        cm.append(confmtrx)

        # Model yang dipilih adalah model yang menghasilkan akurasi tertinggi
        best_model = model[acc.index(max(acc))]
        print(best_model)
        # Menyimpan hasil
        pickle.dump(acc, open("accuracy.sav", "wb"))
        pickle.dump(pre, open("precision.sav", "wb"))
        pickle.dump(rec, open("recall.sav", "wb"))
        pickle.dump(best_model, open("model.sav", "wb"))
        pickle.dump(cm, open("confusionMtrx.sav", "wb"))

        # Menyimpan untuk grafik dalam bentuk array
        data = np.zeros((len(acc), 4))
        for i in range(len(acc)):
            data[i][0] = i+1
            data[i][1] = acc[i]
            data[i][2] = pre[i]
            data[i][3] = rec[i]
    # Mempasing seluruh data yang akan ditampilkan
    context = {
        'title': 'Hasil Training - LDA',
        'heading': ' Hasil Training',
        'hasil': data.tolist(),
        'list': listed,
    }
    return render(request, 'training/trainresult.html', context)
