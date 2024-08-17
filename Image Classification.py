#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


get_ipython().system('pip list')


# In[ ]:


get_ipython().system('pip install tensorflow tensorflow-gpu opencv-python matplotlib')


# In[1]:


import tensorflow as tf
import os


# In[2]:


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    
    tf.config.experimental.set_memory_growth(gpu, True)


# In[3]:


tf.config.list_physical_devices('GPU')


# In[4]:


import cv2
import imghdr


# In[5]:


data_dir='C:/Users/vires/OneDrive/Documents/VD_ML project/data'


# In[6]:


image_exts = ['jpeg','jpg', 'bmp', 'png']


# In[7]:


for image_class in os.listdir(data_dir):
    
    print(image_class)


# In[8]:


for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)


# In[9]:


get_ipython().run_line_magic('pinfo2', 'tf.data.Dataset')


# In[10]:


import numpy as np
from matplotlib import pyplot as plt


# In[11]:


data = tf.keras.utils.image_dataset_from_directory('C:/Users/vires/OneDrive/Documents/VD_ML project/data')


# In[12]:


data_iterator = data.as_numpy_iterator()


# In[13]:


batch = data_iterator.next()


# In[14]:


batch[1]


# In[15]:


fig, ax = plt.subplots(ncols=8, figsize=(20,20))
for idx, img in enumerate(batch[0][:8]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
    # class 0 = non violation
    # class 1 = violation


# In[16]:


scaled=batch[0]/255


# In[17]:


scaled.max()


# In[18]:


#allows transformations in pipeline---(map)
#x=image, y=label variable
data = data.map(lambda x,y: (x/255, y))


# In[19]:


scaled_iterator = data.as_numpy_iterator()


# In[20]:


batch = scaled_iterator.next()


# In[21]:


batch[0].max()


# In[ ]:





# In[22]:


data.as_numpy_iterator().next()


# In[23]:


len(data)


# In[24]:


5*.5


# In[25]:


train_size = int(len(data)*.5)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1


# In[26]:


train_size+val_size+test_size


# In[27]:


5*.2


# In[28]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# In[30]:


len(test)


# In[31]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[32]:


model = Sequential()


# In[33]:


model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[34]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[35]:


model.summary()


# In[36]:


logdir='logs'


# In[37]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[38]:


hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# In[39]:


train


# In[40]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[41]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[42]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[43]:


pre = Precision()
re = Recall()
acc = BinaryAccuracy()


# In[44]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[45]:


print(f'Precision : {pre.result().numpy()}, Recall : {re.result().numpy()}, Accuracy : {acc.result().numpy()}')


# In[46]:


import cv2


# In[ ]:





# In[47]:


img = cv2.imread('C:/Users/vires/OneDrive/Documents/VD_ML project/violation.jpeg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()


# In[48]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[49]:


resize.shape


# In[50]:


np.expand_dims(resize,0).shape


# In[51]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[52]:


yhat


# In[53]:


if yhat > 0.65: 
    print(f'Predicted class is Violence')
else:
    print(f'Predicted class is Non-violence')


# In[54]:


from tensorflow.keras.models import load_model


# In[55]:


model.save(os.path.join('C:/Users/vires/OneDrive/Documents/VD_ML project/models','imageclassifier.keras'))


# In[56]:


new_model = load_model('C:/Users/vires/OneDrive/Documents/VD_ML project/models/imageclassifier.keras')


# In[57]:


yhatnew = new_model.predict(np.expand_dims(resize/255, 0))


# In[58]:


if yhatnew  > 0.5: 
    print(f'Predicted class is Violence')
else:
    print(f'Predicted class is Non-violence')


# In[ ]:




