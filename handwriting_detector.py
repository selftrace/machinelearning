import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

[x_train, y_train], [x_test, y_test] = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

tf_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

tf_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

tf_model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=1
)

num_samples = 5
sample_imgs = x_test[:num_samples]
probs = tf.nn.softmax(tf_model.predict(sample_imgs)).numpy()

plt.figure(figsize=(12, 5))
for i in range(num_samples):
  pred_y = np.argmax(probs[i])
  conf = probs[i][pred_y] 
  actual_y = y_test[i]
  correct = pred_y == actual_y

  plt.subplot(1, num_samples, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(sample_imgs[i], cmap=plt.cm.binary) 

  plt.title(
      f'predicted {pred_y} (conf: {conf:.2f})\nactual: {actual_y}', 
      color = 'green' if correct else 'red'
  )

plt.tight_layout() 
plt.show() 
