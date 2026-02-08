import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIG ---
DATA_DIR = "dataset"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30  
NUM_CLASSES = 9 

train_datagen = ImageDataGenerator (
    rescale=1./255,
    validation_split=0.2,
    brightness_range=[0.8, 1.2],
    rotation_range=20, 
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    zoom_range=0.1,         
    horizontal_flip=False,  
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed = 0
)
val_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=0
)

model = models.Sequential([

    layers.Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    

    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(), 
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    

    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    

    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'), 
    layers.BatchNormalization(),
    layers.Dropout(0.4), 
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,       
    patience=3,    
    min_lr=0.00001, 
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=8,       
    restore_best_weights=True
)

print("Starting Turbo Training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop] 
)

model.save("drone_cnn.keras")
print("Model saved as 'drone_cnn.keras'")