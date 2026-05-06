import os 
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

Dataset_path= r"C:\Users\sarvesh\Downloads\bananas_pic"

train_dir= os.path.join(Dataset_path,'train' )
valid_dir= os.path.join(Dataset_path, 'valid')
test_dir= os.path.join(Dataset_path, 'test')

img_size= (160, 160)
Batch_size= 16
Epochs= 5

os.makedirs('model', exist_ok=True)

train_data= tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size= img_size,
    batch_size= Batch_size,
    label_mode= "categorical"
    )

valid_data= tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size= img_size,
    batch_size= Batch_size,
    label_mode= 'categorical'
)

test_data= tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size= img_size,
    batch_size= Batch_size,
    label_mode='categorical',
    shuffle= False
)

class_names= train_data.class_names
print('Classes:', class_names)

with open("model/class_names.txt", 'w') as f:
    for name in class_names:
        f.write(name + '\n')
        
autotune= tf.data.AUTOTUNE

train_data= train_data.prefetch(buffer_size= autotune)
valid_data= valid_data.prefetch(buffer_size= autotune)
test_data= test_data.prefetch(buffer_size= autotune)

data_augmentation= tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

base_model= MobileNetV2(
    input_shape= (160, 160, 3),
    include_top= False,
    weights= 'imagenet'
)

base_model.trainable= False

model= models.Sequential([
    layers.Input(shape=(160,160,3)),
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
    
])

model.compile(
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss= 'categorical_crossentropy',
    metrics= ['accuracy']
)

callbacks= [
    EarlyStopping(
        monitor= 'val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "model/banana_model.keras",
        monitor= 'val_accuracy',
        save_best_only= True
    )
]

history= model.fit(
    train_data,
    validation_data= valid_data,
    epochs= Epochs,
    callbacks= callbacks
)

test_loss, test_accuracy= model.evaluate(test_data)
print("Test Accuracy:", round(test_accuracy *100, 2), '%')

model.save("model/banana_model.keras")

print("Model saved successfully")