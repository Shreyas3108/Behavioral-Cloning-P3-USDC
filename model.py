
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, Convolution2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.utils import np_utils
from keras.regularizers import l2
from keras.preprocessing import image
from keras.optimizers import Adam
from keras import backend as K


# In[2]:


def duplicate_data(data):
    '''
    input:
        data in dataframe format
    output:
        concatenated dataframe with flipping angles. 
    '''
    data_left = data[['left','steering']]
    data_left['steering'] = data_left['steering'] + 0.3
    data_left['image'] = data_left['left']
    data_left = data_left.drop('left',axis = 1)
    data_right = data[['right','steering']]
    data_right['steering'] = data_right['steering'] - 0.3
    data_right['image'] = data_right['right']
    data_right = data_right.drop('right',axis = 1)
    data_center = data[['center','steering']]
    data_center['image'] = data_center['center']
    data_center = data_center.drop('center',axis = 1)
    combined_data = pd.concat([data_center, data_left, data_right])
    flipped_data = combined_data.copy()
    combined_data['flip'] = False
    flipped_data['flip'] = True
    flipped_data['steering'] = flipped_data['steering'] * -1
    final_data = pd.concat([combined_data, flipped_data, combined_data.copy(), flipped_data.copy()])
    return final_data


# In[3]:


def crop_image(image, top=70, bottom=135):
    return image[top:bottom]


# In[4]:


def resize_image(image):
    return cv2.resize(image,(224, 49), interpolation= cv2.INTER_AREA)


# In[5]:


def brightness(image,angle):
    choice = np.random.randint(2)
    if choice == 1:
        c = np.random.uniform(0.3,1.2)
        hsv = cv2.cvtColor(image , cv2.COLOR_RGB2HSV)
        hsv[:,:,2] = hsv[:,:,2] * c
        brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        image = brightness
        return image , angle
    else:
        return image,angle


# In[6]:


def augmentation(entry):
    data_directory = 'data/'
    image_flip, angle = entry
    image = plt.imread(data_directory+image_flip[1]['image'].strip())
    angle = angle[1]['steering']
    image = image[70:135]
    image = cv2.resize(image,(224, 49), interpolation= cv2.INTER_AREA)
    image, angle = brightness(image, angle)
    flip = image_flip[1]['flip']
    if flip:
        image = image[:,:,::-1]
    return image, angle


# In[7]:


def generator(X, y, batch_size=128):
    N = X.shape[0]
    number_of_batches = int(np.ceil(N / batch_size))
    while True:
        X, y  = shuffle(X, y)
        for i in range(number_of_batches):
            start_index = i*batch_size
            end_index = (i+1)*(batch_size)
            if end_index <= N:
                X_batch = X[start_index:end_index]
                y_batch = y[start_index:end_index]
            else:
                X_batch = X[start_index:]
                y_batch = y[start_index:]
            X_batch, y_batch = X_batch.iterrows(), y_batch.iterrows()
            X_image_batch, y_batch = zip(*map(augmentation, zip(X_batch, y_batch)))
            X_image_batch = np.asarray(X_image_batch)
            y_batch = np.asarray(y_batch)
            yield X_image_batch, y_batch


# In[20]:


def model():
    model = Sequential()
    model.add(Lambda(lambda x: (x/ 127.5 - 1.),input_shape=(49,224,3)))
    model.add(Convolution2D(3, 1,1, border_mode='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(32, 5 , 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 5 , 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3 , 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    print(model.summary())
    return model


# In[21]:


data = pd.read_csv('data/driving_log.csv')


# In[22]:


data = duplicate_data(data)


# In[23]:


images = data[['image', 'flip']]
angles = data[['steering']]


# In[24]:


X_train, X_valid, y_train, y_valid = train_test_split(images, angles, test_size=0.1)


# In[25]:


train_gen = generator(X_train, y_train)
valid_gen = generator(X_valid, y_valid)


# In[26]:


model = model()


# In[24]:


model.fit_generator(train_gen, X_train.shape[0], nb_epoch=3, validation_data=valid_gen, nb_val_samples=X_valid.shape[0])


# In[25]:


model.save('model.h5')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




