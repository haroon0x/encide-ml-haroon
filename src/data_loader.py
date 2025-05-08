import tensorflow as tf

def load_data(train_dir, test_dir, image_size=(256, 256), batch_size=32, seed=123):
    train_data = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=0.2,
        subset='training',
        verbose=True,
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=0.2,
        subset='validation',
        verbose=True,
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        labels=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        verbose=True,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    val_data = val_data.prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.prefetch(buffer_size=AUTOTUNE)

    return train_data, val_data, test_data