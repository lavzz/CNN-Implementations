## This script classified dogs from cats with 85% accuracy on a test set. It uses the dataset from Kaggle (called dogs vs. cats kernel edition)
## and uses a fairly small sub-set (~2000 images out of the available 24000) to train a CNN. The network which uses VGG16 to 
## transfer learn is trained on CPU for ~30 minutes only. 

import os
import shutil

original_dataset_dir = './data/train/'
base_dir = './data'
if not os.path.exists(base_dir):
	os.mkdir(base_dir)


#Creating the directories for training and test data 

train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
	os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
	os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test1')
if not os.path.exists(test_dir):
	os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
	os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
	os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_dir):
	os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dir):
	os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
	os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
	os.mkdir(test_dogs_dir)


#copy some training images to the newly created training folders 

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_cats_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_dogs_dir, fname)
	shutil.copyfile(src, dst)

#copy some test images to the newly created test folders 

fnames = ['cat.{}.jpg'.format(i) for i in range(2000, 2500)]
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_cats_dir, fname)
	shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(2000, 2500)]
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_dogs_dir, fname)
	shutil.copyfile(src, dst)

# copy some validation images to the newly created validation folders


##Feature extraction 
##import prepare_data
import os
import shutil
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator 
from keras.applications import VGG16


train_size, validation_size, test_size, = 2000, 1000, 1000 
img_width , img_height = 128, 128 


base_dir = './data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test1')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (128, 128,3))
conv_base.summary() 

datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 32 

def extract_features(directory, sample_count):
	features = np.zeros(shape = (sample_count, 4,4, 512))
	labels = np.zeros(shape = (sample_count))

	generator = datagen.flow_from_directory(directory, target_size = (img_width, img_height), batch_size = batch_size, class_mode = 'binary')

	i = 0 
	for inputs_batch , labels_batch in generator:
		features_batch = conv_base.predict(inputs_batch)
		features[i*batch_size:(i+1)*batch_size] = features_batch
		labels[i*batch_size: (i+1)*batch_size] = labels_batch
		i += 1
		if i*batch_size>= sample_count:
			break
	return features, labels 


train_features, train_labels = extract_features(train_dir, train_size)
validation_features, validation_labels = extract_features(validation_dir, validation_size)
test_features, test_labels = extract_features(test_dir, test_size)



from keras import models
from keras import layers
from keras import optimizers 
import matplotlib.pyplot as plt 
"""
from feature_extraction import train_features, train_labels, validation_features, validation_labels
from feature_extraction import test_cats_dir, test_dogs_dir, batch_size, img_width, img_heights
"""
import os 

epochs = 100 


model = models.Sequential()
model.add(layers.Flatten(input_shape = (4,4,512)))
model.add(layers.Dense(256, activation = "relu", input_dim = (4*4*512)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary() 

model.compile(optimizer = optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(train_features, train_labels, epochs = epochs, batch_size = batch_size, validation_data = (validation_features, validation_labels))

model.save('dogs_cats_fcl.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label = 'training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()


from keras.preprocessing import image
import random 
def visualize_predictions(classifier, n_cases):
    for i in range(0, n_cases):
        path = random.choice([test_cats_dir, test_dogs_dir])
        
        random_img = random.choice(os.listdir(path))
        img_path = os.path.join(path, random_img)
        img = image.load_img(img_path, target_size = (img_width, img_heights))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.
        
        features = conv_base.predict(img_tensor.reshape(1, img_width, img_heights, 3))
        
        try:
            prediction = classifier.predict(features)
        except:
            prediction = classifier.predict(features.reshape(1, 4*4*512))
        
        plt.imshow(img_tensor)
        plt.show()
        
        if prediction< 0.5:
            print('Cat')
        else:
            print('Dog')


visualize_predictions(model, 5)


##prepare for submission 
"""
from tqdm import tqdm 

pred_list = []
img_list = [] 

for img in tqdm(os.listdir('./test1')):
	img_data = img[0]
	data = img_data.reshape(-1, 128,128,3)
	predicted = model.predict([data])[0]
	img_list.append(img_)
"""


