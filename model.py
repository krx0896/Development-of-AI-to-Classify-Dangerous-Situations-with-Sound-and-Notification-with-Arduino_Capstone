import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, TimeDistributed, Permute, ReLU, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers.experimental import preprocessing

# Data augmentation for spectrograms, which can be configured in visual mode.
# To learn what these arguments mean, see the SpecAugment paper:
# https://arxiv.org/abs/1904.08779
sa = SpecAugment(spectrogram_shape=[int(input_length / 65), 65], mF_num_freq_masks=0, F_freq_mask_max_consecutive=0, mT_num_time_masks=0, T_time_mask_max_consecutive=0, enable_time_warp=False, W_time_warp_max_distance=6, mask_with_mean=False)
train_dataset = train_dataset.map(sa.mapper(), num_parallel_calls=tf.data.AUTOTUNE)

EPOCHS = 100
LEARNING_RATE = 0.005
# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

# model architecture
model = Sequential()
# Data augmentation, which can be configured in visual mode
model.add(tf.keras.layers.GaussianNoise(stddev=0.2))
model.add(Reshape((int(input_length / 65), 65, 1), input_shape=(input_length, )))
model.add(preprocessing.Resizing(24,24, interpolation= 'nearest'))

model.add(Conv2D(16, kernel_size=3))
model.add(ReLU(6.0))

model.add(Conv2D(32, kernel_size=3))
model.add(ReLU(6.0))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
# model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(64))
model.add(ReLU(6.0))

model.add(Dense(32))
model.add(ReLU(6.0))


model.add(Dense(classes, name='y_pred', activation='softmax'))


# this controls the learning rate
lr_schedule = ExponentialDecay(LEARNING_RATE, 
    decay_steps=train_sample_count//BATCH_SIZE*15,
    decay_rate=0.96, staircase=False)
opt = Adam(learning_rate= lr_schedule)
# earlystopping =EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS))

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, verbose=2, callbacks=[callbacks])

# Use this flag to disable per-channel quantization for a model.
# This can reduce RAM usage for convolutional models, but may have
# an impact on accuracy.
disable_per_channel_quantization = False