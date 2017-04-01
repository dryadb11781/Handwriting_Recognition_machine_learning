from django.http import HttpResponse
from django.template.loader import get_template
from django import template
from django.shortcuts import render_to_response
import os
from sklearn.externals import joblib

def index(request):
    return render_to_response('index.html',locals())


def predict(request):
    import numpy as np
    data = np.array(request.GET['test'].split(','))
    X_test=data
    img_rows, img_cols = 64, 64
    batch_size = 1
    from keras import backend as K
    if K.image_dim_ordering() == 'th':
        #X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(1, 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        #X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(1, img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)


    #X_test = X_test.reshape(1, 1, img_rows, img_cols)
    #input_shape = (None, img_rows, img_cols)
    from keras.models import load_model
    model = load_model('cnn_model.h5')

    #prediction
    pred = model.predict_classes(X_test, batch_size, verbose=0)
    return HttpResponse(pred)
# Create your views here.
