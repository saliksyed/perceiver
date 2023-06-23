
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt

def add_prefix(model, prefix: str, custom_objects=None):
    config = model.get_config()
    old_to_new = {}
    new_to_old = {}
    for layer in config['layers']:
        new_name = prefix + layer['name']
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]
    
    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]
    
    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]
    
    config['name'] = prefix + config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)
    
    for layer in new_model.layers:
        layer.trainable = False
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())
    
    return new_model

img_input = tf.keras.layers.Input(name="main_input", shape=(1120, 224, 3))

views = []
for i in range(0,5):
    
    crop = tf.keras.layers.Cropping2D(
        name="crop-{i}".format(i=i),
        cropping=((i * 224, (4-i) * 224), (0,0))
    )

    vgg = add_prefix(VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3)), 'model-{d}-'.format(d=i))
    block4 = vgg.get_layer('model-{i}-block4_pool'.format(i=i)).output
    x = tf.keras.layers.Flatten(name="flatten_layer-{i}".format(i=i))(block4)
    x = Dense(32, activation = "relu", name = "merge_layer-{i}".format(i=i))(x)
    new_vgg = Model(name="vgg-{d}".format(d=i), inputs=[vgg.input], outputs=[x])
    model = tf.keras.Sequential([img_input, crop, new_vgg])
    views.append(model)

merged = tf.keras.layers.Concatenate(axis=1)([v.output for v in views])
#Final Layer
combine = Dense(32, activation = "relu", name = "merge_layer")(merged)
output_layer = Dense(7, name="output_layer")(combine)

model = Model(
    inputs=[img_input], 
    outputs=[output_layer],
    name="merged"
)

model.summary()
tf.keras.utils.plot_model(model, 'model.png')

train_ds  = tf.keras.utils.image_dataset_from_directory('../datagen/extruded_polygons', seed=123, validation_split=0.2, subset="training", image_size=(1120,224))
test_ds  = tf.keras.utils.image_dataset_from_directory('../datagen/extruded_polygons', seed=123, validation_split=0.2, subset="validation", image_size=(1120,224))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.Rescaling(1./255)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()