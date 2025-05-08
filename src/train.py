import argparse
import tensorflow as tf
from data_loader import load_data
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import plot_history

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', required=True)
parser.add_argument('--test_dir', required=True)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs_frozen', type=int, default=15)
parser.add_argument('--epochs_finetune', type=int, default=10)
args = parser.parse_args()

tf.config.run_functions_eagerly(True)
train_data, val_data, test_data = load_data(
    train_dir=args.train_dir,
    test_dir=args.test_dir,
    batch_size=args.batch_size
)

model = build_model()
initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr,
    decay_steps=train_data.cardinality().numpy() // 2,
    decay_rate=0.9,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint('models/best_resnet_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
]

print("Phase 1: Training with frozen base model...")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=args.epochs_frozen,
    callbacks=callbacks,
    verbose=1
)

print("Phase 2: Fine-tuning base model layers...")
base_model = model.layers[3]
base_model.trainable = True
for layer in base_model.layers[:250]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=args.epochs_finetune,
    callbacks=callbacks,
    verbose=1
)

model.save('models/resnet_finetuned_model.keras')
plot_history(history1, history2, save_path='training_history/training_history.png')
print("Training complete and Keras model saved. For TFLite conversion, use export_model.py.")
