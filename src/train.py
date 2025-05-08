
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from keras import Sequential 
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten , Dropout ,BatchNormalization
from keras.callbacks import EarlyStopping

train_data = keras.utils.image_dataset_from_directory(
    directory='/kaggle/working/train',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training',
    verbose=True,
)

val_data = keras.utils.image_dataset_from_directory(
    directory='/kaggle/working/train',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation',
    verbose=True,
)

test_data = keras.utils.image_dataset_from_directory(
    directory='/kaggle/input/dogs-vs-cats/test',
    labels=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    verbose=True,
)

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)
test_data = test_data.prefetch(buffer_size=AUTOTUNE)

model = Sequential()

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomContrast(0.2),
    keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
])

normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(train_data.map(lambda x, y: x))

input_layer = keras.Input(shape=(256, 256, 3))
x = normalizer(input_layer)
x = data_augmentation(input_layer)
x = Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2),strides = 2,padding='valid')(x)
x= Conv2D(64,(3,3),activation='relu')(x)
x= MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(128,activation = 'relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x) # to prevent overfitting
output_layer = Dense(1,activation='sigmoid')(x)
model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy',    metrics=['accuracy', keras.metrics.AUC(name='auc')])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=11,
    callbacks=[early_stopping,keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    ]
)

val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc:.2f}")