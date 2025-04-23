import tensorflow as tf
from tensorflow.keras.untils import image_dataset_from_directory
from tensorflow import layers, models
import numpy as np 
from tensorflow.keras.preprocessing import image
import tensorflow_datasets as tfds

(train_ds, test_ds), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)
# loading data
dataset = image_dataset_from_directory(
    "data/train",
    labels = 'inferred',
    label_mode = 'binary',
    image_size = (128,128),
    batch_size = 32
)

# build model 
model = models.Sequential([
    #Block 1 
    layers.Rescaling(1./255, input_shape(128,128,3)),
    layers.Conv2D(32,(3,3),activate = 'relu'),
    layers.Maxpooling2D(2,2),

    #Block 2
    layers.Conv2D(64,(3,3), activate='relu'),
    layers.Maxpooling2D(2,2),

    #Block 3 
    layers.Flatten(),
    layers.Dense(64,activate='relu'),
    layers.Dense(1,activate= 'sigmoid') # Using for resolve problem just label 1 0
])


# Compile & Train 

model.compile(optimizer='adam',
              loss ='binary-crossentropy',
              metrics = ['accuracy']
)

img =image.load_img("data/test/cat.png",target_size(128,128))
img_arr = image.img_to_array(img)/255.0
img_arr = np.expand_dims(img_arr,axis=0)

prediction = model.predict(img_arr)
print('dog' if prediction[0][0] > 0.5 else 'cat')