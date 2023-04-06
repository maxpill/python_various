import tensorflow as tf
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForMaskedLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-base-cased')
model = TFAutoModelForMaskedLM.from_pretrained('allegro/herbert-base-cased')

# Read data from CSV
data = pd.read_csv('drive/MyDrive/data.csv', header=None)

# Set parameters
batch_size = 8
max_length = 512
num_epochs = 10
learning_rate = 2e-5

# Set optimizer and scheduler
optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate, decay_steps=1, decay_rate=0.9, staircase=True)

# Prepare dataset
def encode_text(text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    return input_ids, attention_mask

dataset = tf.data.Dataset.from_tensor_slices(data[0].tolist())
dataset = dataset.map(lambda text: tf.py_function(encode_text, [text], (tf.int32, tf.float32)))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Train model
for epoch in range(num_epochs):
    for i, (input_ids, attention_mask) in enumerate(dataset):
        with tf.GradientTape() as tape:
            # Mask input tokens
            masked_input_ids = input_ids.numpy().copy()
            masked_input_ids[masked_input_ids == tokenizer.mask_token_id] = tokenizer.mask_token_id
            masked_input_ids = tf.convert_to_tensor(masked_input_ids, dtype=tf.int32)

            # Forward pass
            outputs = model(masked_input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=True)
            loss = outputs.loss

        # Backward pass
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update scheduler
        optimizer.learning_rate = scheduler(optimizer.iterations)

        # Print metrics
        if i % 10 == 0:
            predicted_ids = tf.argmax(outputs.logits, axis=-1)
            num_correct = tf.reduce_sum(tf.cast(predicted_ids == input_ids, tf.int32)).numpy()
            accuracy = num_correct / (batch_size * max_length)
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(data)}, Loss {loss.numpy()}, Accuracy {accuracy}')
