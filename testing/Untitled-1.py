
from django.shortcuts import render
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import os
import numpy as np
import cv2
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

# Create your views here.


def index(request):
    context = {
        'title': 'Training - LDA',
        'heading': 'Training',
    }
    return render(request, 'training/index.html', context)
# Fungsi Klasifikasi


def classify(trainX, trainY, testX, testY):
    clf = LinearDiscriminantAnalysis()
    clf.fit(trainX, trainY)
    prediction = clf.predict(testX)
    # print(prediction)
    acc = accuracy_score(testY, prediction)
    pre = precision_score(testY, prediction)
    rec = recall_score(testY, prediction)
    confusion_matrix = confusion_matrix(testY, prediction)
    return acc, pre, rec, clf, confusion_matrix

# Awal imageprocessing


def imageprocessing(frame):
    # Meresize gambar menjadi ukuran 336 x 448
    width = 336
    height = 448
    dim = (width, height)
    frame = cv2.resize(frame, dim)
    #print('Resized Dimensions : ', frame.shape)
    # OpenCV akan mengimport gambar kedalam format warna BGR sehingga harus dikonversi menjadi format RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # melakukan konversi dari ruang warna menjadi ruang warna hsv
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # melakukan split image (memisahkan image hsv menjadi hue, saturation dan value untuk melakukan histogram qualization terhadap value)
    H, S, V = cv2. split(image_hsv)
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

    # proses masking using upper and lower bounds
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(image_hsv, lower_brown, upper_brown)
    final_mask = mask_green + mask_brown
    # Memisahkan citra foreground dengan citra background
    # Bitwise-AND mask dan gambar asli
    bitwise_and = cv2.bitwise_and(equalized, equalized, final_mask)
    image = cv2.cvtColor(bitwise_and, cv2.COLOR_HSV2RGB)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # Final processing dengan memberikan sedikit blur
    return image
# Akhir imageprocessing


def training(request):
    if request.method == 'POST':
        path = request.POST['path']
        # Mengimport dataset
        dir = r"{}".format(path)
        categories = ['matang', 'mentah']
        # Menampilkan data kedalam tabel
        dataImage = {}
        list = []
        # Menyiapkan variabel penampun data
        data = []
        #feature_matrix = np.zeros((530, 390))
        # Dalam setiap kategori (kelas) akan diambil gambar dan dilakukan preprocessing
        i = 1
        for category in categories:
            path = os.path.join(dir, category)
            label = categories.index(category)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                dataImage["No"] = i
                dataImage["Nama"] = str(img)
                dataImage["Path"] = str(img_path)
                dataImage["Kelas"] = str(category)
                list.append(dataImage.copy())
                i = i+1
                # Membaca gambar dari directory
                frame = cv2.imread(img_path)
                # Memanggil fungsi imageprocessing
                image = imageprocessing(frame)
                features = image.reshape(448, 336, -1)
                data.append([features, label])

        # Data akan diacak untuk menyebar data
        random.shuffle(data)
        features = []
        labels = []
        # Mengambil fitur dan label dari data
        for feature, label in data:
            features.append(feature)
            labels.append(label)
        # Pengujian K-Fold Cross Validation
        # Inisialisasi variabel penampung hasil uji
        akurasi = []
        presisi = []
        recall = []
        model = []
        confusion_matrix = []
        # Menggunakan K sejumalh ...
        cv = KFold(n_splits=3, shuffle=True, random_state=None)
        cv.get_n_splits(features)
        # Split data dengan K fold
        for train_index, test_index in cv.split(features):
            trainX = []
            trainY = []
            testX = []
            testY = []
            # Split data menjadi data trainX dan testX untuk data fitur
            # Split data menjadi data trainY dan testY untuk data label
            for i in test_index:
                testX.append(features[i])
                testY.append(labels[i])
            for i in train_index:
                trainX.append(features[i])
                trainY.append(labels[i])
            # Memanggil fungsi classify
            acc, pre, rec, mod, confusion_matrix = classify(
                trainX, trainY, testX, testY)
            # Menyimpan hasil setiap fold dar pengujian
            akurasi.append(acc)
            presisi.append(pre)
            recall.append(rec)
            model.append(mod)
            confusion_matrix.append(confusion_matrix)
        # Model yang dipilih adalah model yang menghasilkan akurasi tertinggi
        best_model = model[akurasi.index(max(akurasi))]
        # Menyimpan hasil
        pickle.dump(akurasi, open("accuracy.sav", "wb"))
        pickle.dump(presisi, open("precision.sav", "wb"))
        pickle.dump(recall, open("recall.sav", "wb"))
        pickle.dump(best_model, open("model.sav", "wb"))
        pickle.dump(confusion_matrix, open("confusionMtrx.sav", "wb"))
        # Menyimpan untuk grafik dalam bentuk array
        data = np.zeros((len(akurasi), 4))
        for i in range(len(akurasi)):
            data[i][0] = i+1
            data[i][1] = akurasi[i]
            data[i][2] = presisi[i]
            data[i][3] = recall[i]
    # Mempasing seluruh data yang akan ditampilkan
    context = {
        'title': 'Hasil Training - LDA',
        'heading': ' Hasil Training',
        'hasil': data.tolist(),
        'list': list,
    }
    return render(request, 'training/trainresult.html', context)
