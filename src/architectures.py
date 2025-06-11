# src/architectures.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.applications import InceptionV3, VGG19

def build_model(model_name: str, 
                num_classes: int, 
                input_shape: tuple = (224, 224, 3), 
                weights: str = 'imagenet',
                fine_tune: bool = False) -> tf.keras.Model: 

    inputs = Input(shape=input_shape)

    if model_name == 'InceptionV3':
        base_model = InceptionV3(include_top=False, weights=weights, input_tensor=inputs)
    elif model_name == 'VGG19':
        base_model = VGG19(include_top=False, weights=weights, input_tensor=inputs)
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

    base_model.trainable = fine_tune

    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(256, use_bias=False, name="dense_256")(x)
    x = BatchNormalization(name="batch_norm")(x)
    x = Activation("relu", name="relu_activation")(x)
    x = Dropout(0.5, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs)

    learning_rate = 1e-5 if fine_tune else 1e-3
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"Modelo {model_name} construido. Capas base entrenables: {base_model.trainable}")
    return model
