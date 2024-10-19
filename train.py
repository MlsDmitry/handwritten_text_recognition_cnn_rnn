#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 01:41:03 2024

@author: dima
"""

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from keras.preprocessing.sequence import pad_sequences
# from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import pad_sequences

from tqdm import tqdm

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam


from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as tf_keras_backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten

tf_keras_backend.set_image_data_format('channels_last')
tf_keras_backend.image_data_format()   

with open('./words.txt') as f:
    contents = f.readlines()

lines = [line.strip() for line in contents] 
lines = lines[18:100]
lines[0]

max_label_len = 0

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 

# string.ascii_letters + string.digits (Chars & Digits)
# or 
# "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

print(char_list, len(char_list))

def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, chara in enumerate(txt):
        dig_lst.append(char_list.index(chara))
        
    return dig_lst

images = []
labels = []

RECORDS_COUNT = 10000
train_images = []
train_labels = []
train_input_length = []
train_label_length = []
train_original_text = []

valid_images = []
valid_labels = []
valid_input_length = []
valid_label_length = []
valid_original_text = []

inputs_length = []
labels_length = []
def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    
    img = img.astype('float32')
    
    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
        
    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)
    
    img = cv2.subtract(255, img)
    
    img = np.expand_dims(img, axis=2)
    
    # Normalize 
    img = img / 255
    
    return img

for index, line in enumerate(lines):
    splits = line.split(' ')
    status = splits[1]
    
    if status == 'ok':
        word_id = splits[0]
        word = "".join(splits[8:])
        
        splits_id = word_id.split('-')
        filepath = 'words/{}/{}-{}/{}.png'.format(splits_id[0], 
                                                  splits_id[0], 
                                                  splits_id[1], 
                                                  word_id)
        
        # process image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        try:
            img = process_image(img)
        except:
            continue
            
        # process label
        try:
            label = encode_to_labels(word)
        except:
            continue
        
        if index % 10 == 0:
            valid_images.append(img)
            valid_labels.append(label)
            valid_input_length.append(31)
            valid_label_length.append(len(word))
            valid_original_text.append(word)
        else:
            train_images.append(img)
            train_labels.append(label)
            train_input_length.append(31)
            train_label_length.append(len(word))
            train_original_text.append(word)
        
        if len(word) > max_label_len:
            max_label_len = len(word)
    
    if index >= RECORDS_COUNT:
        break
train_padded_label = pad_sequences(train_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))

valid_padded_label = pad_sequences(valid_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))
train_padded_label.shape, valid_padded_label.shape


train_images = np.asarray(train_images)
train_input_length = np.asarray(train_input_length)
train_label_length = np.asarray(train_label_length)

valid_images = np.asarray(valid_images)
valid_input_length = np.asarray(valid_input_length)
valid_label_length = np.asarray(valid_label_length)
train_images.shape 


def Model1():
    # input with shape of height=32 and width=128 
    inputs = Input(shape=(32,128,1))
     
    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
     
    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
     
    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
     
    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
     
    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)
     
    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
     
    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
     
    # squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    reshape = Reshape((31, 512))(conv_7)
    
     
    # bidirectional LSTM layers with units=128
    # blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed
    blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(reshape)
    blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)
     
    outputs = Dense(len(char_list)+1, activation = 'softmax', name='predictions')(blstm_2)

    # model to be used at test time
    act_model = Model(inputs=inputs, outputs=outputs)
    
    return act_model,outputs,inputs


model,outputs,inputs=Model1()

model.summary()

batch_size = 8
epochs = 60
e = str(epochs)
optimizer_name = 'adam'


def my_ctc_loss(labels, y_pred, input_length, label_length):
    # y_pred, labels, input_length, label_length = args
    
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

# optimizer = Adam(learning_rate=1e-3)


# epochs = 10
# for epoch in range(epochs):
#     print("\nStart of epoch %d" % (epoch,))

#     # Iterate over the batches of the dataset.
#     # for step, train_batch in enumerate(train_dataset_batched):
#     for i in range(len(train_images) - 1):
#         step = i
#         # x_train = train_images_batched[i]
#         # x_train_lengths = train_input_length_batched[i]
#         # y_train = train_padded_label_batched[i]
#         # y_train_lengths = train_label_length_batched[i]
#         x_train = train_images[i]
#         x_train_lengths = train_input_length[i]
#         y_train = train_padded_label[i]
        
#         # x_train = np.insert(x_train, 0, y_train)
#         x_train = (y_train, x_train)
#         y_train_lengths = train_label_length[i]
#         # Open a GradientTape to record the operations run
#         # during the forward pass, which enables auto-differentiation.
#         with tf.GradientTape() as tape:
#             # Run the forward pass of the layer.
#             # The operations that the layer applies
#             # to its inputs are going to be recorded
#             # on the GradientTape.
#             logits = model(x_train, training=True)  # Logits for this minibatch

#             # Compute the loss value for this minibatch.
#             # loss_value = loss_fn(y_batch_train, logits)
#             loss_value = my_ctc_loss(y_train, logits, x_train_lengths, y_train_lengths)

#         # Use the gradient tape to automatically retrieve
#         # the gradients of the trainable variables with respect to the loss.
#         grads = tape.gradient(loss_value, model.trainable_weights)

#         # Run one step of gradient descent by updating
#         # the value of the variables to minimize the loss.
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

#         # Log every 200 batches.
#         # if step % 1 == 0:
#         print(
#             "Training loss (for one batch) at step %d: %.4f"
#             % (step, float(loss_value))
#         )
#         print("Seen so far: %s samples" % ((step + 1) * batch_size))

# Define batch size
batch_size = 8

# Create train dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_padded_label, train_input_length, train_label_length)
)

# Shuffle the dataset and batch it
# train_dataset = train_dataset.shuffle(buffer_size=len(train_images)) \
train_dataset = train_dataset.batch(batch_size) \
                             .prefetch(tf.data.experimental.AUTOTUNE)

# Create validation dataset similarly without shuffling
valid_dataset = tf.data.Dataset.from_tensor_slices(
    (valid_images, valid_padded_label, valid_input_length, valid_label_length)
).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

optimizer = Adam()
best_val_loss = np.inf
epochs = 1

# CTC loss function
# def ctc_loss(y_true, y_pred, input_length, label_length):
#     return tf.reduce_mean(
#         tf.nn.ctc_loss(
#             labels=y_true,
#             logits=y_pred,
#             label_length=label_length,
#             logit_length=input_length,
#             blank_index=-1,
#             logits_time_major=False  # the output has shape (batch_size, time_steps, num_classes)
#         )
#     )
def ctc_loss(y_true, y_pred, input_length, label_length):
    """
    Custom CTC loss function using tf.keras.backend.ctc_batch_cost.
    
    Args:
        y_true: The true labels (sparse) with shape (batch_size, max_label_length).
        y_pred: The predicted logits with shape (batch_size, max_time_steps, num_classes).
        input_length: The lengths of the input sequences (logits).
        label_length: The lengths of the true labels.
    
    Returns:
        Computed CTC loss for the batch.
    """
    # Using Keras backend's ctc_batch_cost
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    return tf.reduce_mean(loss)

# Get the number of time steps (sequence length) after convolutions (31 in your case)
# def get_time_steps(input_shape):
#     return (input_shape[1] // 2) // 2  # Calculate based on pooling layers

# time_steps = get_time_steps(train_images.shape)
time_steps = 31

filepath="{}o-{}r-{}e-{}t-{}v.keras".format(optimizer_name,
                                          str(RECORDS_COUNT),
                                          str(epochs),
                                          str(train_images.shape[0]),
                                          str(valid_images.shape[0]))


# Function to calculate accuracy by decoding the predicted logits
def calculate_accuracy(y_true, y_pred, input_length, label_length):
    # Decode the predictions using CTC decoder
    decoded_pred = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)[0][0]
    
    # Cast the decoded predictions to the same dtype as y_true
    decoded_pred = tf.cast(decoded_pred, dtype=tf.int32)

    # Iterate over the batch and compare the true and predicted sequences
    batch_size = tf.shape(y_true)[0]
    correct_predictions = 0
    for i in range(batch_size):
        true_label = y_true[i][:label_length[i]]  # Trim y_true to actual label length
        pred_label = decoded_pred[i][:label_length[i]]  # Trim decoded prediction to label length
        
        # Compare and count correct predictions
        if tf.reduce_all(tf.equal(true_label, pred_label)):
            correct_predictions += 1

    # Calculate accuracy (batch-wise)
    accuracy = correct_predictions / tf.cast(batch_size, tf.float32)
    
    return accuracy

def train_model(model, train_dataset, valid_dataset, epochs, checkpoint_dir, resume_training=False):
    optimizer = tf.keras.optimizers.Adam()
    
    # Define the checkpoint object
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    # Define the checkpoint manager (manages checkpoint files)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    
    # Restore latest checkpoint if resuming training
    if resume_training and checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"Restored from {checkpoint_manager.latest_checkpoint}")
    else:
        print("Starting training from scratch...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
    
        # Initialize metrics
        train_loss = 0.0
        train_accuracy = 0.0
        total_batches = len(train_dataset)
    
        # Create a progress bar for the training loop
        # train_pbar = tqdm(train_dataset, desc=f"Training Epoch {epoch+1}/{epochs}", total=len(train_images)//batch_size)
        
        with tqdm(total=total_batches, desc="Training", unit="batch") as pbar:
            # Training loop
            for batch, (batch_images, batch_padded_labels, batch_input_length, batch_label_length) in enumerate(train_dataset):
                if batch_input_length.shape[0] != 8:
                    continue
                with tf.GradientTape() as tape:
                    # Forward pass (model's call method)
                    y_pred = model([batch_images], training=True)  # Pass only images to model
                    
                    input_length_tensor = tf.reshape(batch_input_length, (8, 1))
                    label_length_tensor = tf.reshape(batch_label_length, (8, 1))
            
                    # Compute CTC loss
                    loss_value = ctc_loss(batch_padded_labels, y_pred, input_length_tensor, label_length_tensor)
                    # print('loss_value:', loss_value)
            
                # Compute gradients
                gradients = tape.gradient(loss_value, model.trainable_variables)
                
                # Apply gradients (backpropagation)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                # Update loss
                train_loss += loss_value
    
                # Calculate accuracy for this batch
                batch_accuracy = calculate_accuracy(batch_padded_labels, y_pred, batch_input_length, batch_label_length)
                train_accuracy += batch_accuracy
    
                # Update progress bar
                pbar.set_postfix({"loss": loss_value.numpy(), "accuracy": batch_accuracy.numpy()})
                pbar.update(1)
        
            # At the end of the epoch, average the loss and accuracy
            train_loss /= total_batches
            train_accuracy /= total_batches
    
            # Print final epoch metrics
            print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    
            # Save a checkpoint at the end of each epoch
            checkpoint_manager.save()
            print(f"Checkpoint saved at epoch {epoch + 1}")
            
            # Validation phase (if validation dataset is available)
            if valid_dataset is not None:
                valid_loss = 0.0
                valid_accuracy = 0.0
                total_val_batches = len(valid_dataset)
                
                with tqdm(total=total_val_batches, desc="Validating", unit="batch") as pbar:
                    for batch, (batch_images, batch_padded_labels, batch_input_length, batch_label_length) in enumerate(valid_dataset):
                        if batch_input_length.shape[0] != 8:
                            continue
                        # Forward pass (without gradient computation)
                        y_pred = model(batch_images, training=False)
    
                        
                        input_length_tensor = tf.reshape(batch_input_length, (8, 1))
                        label_length_tensor = tf.reshape(batch_label_length, (8, 1))
                
                        # Compute CTC loss
                        loss_value = ctc_loss(batch_padded_labels, y_pred, input_length_tensor, label_length_tensor)
                        
                        # Update validation loss
                        valid_loss += loss_value
                        
                        # Calculate accuracy for this batch
                        batch_accuracy = calculate_accuracy(batch_padded_labels, y_pred, batch_input_length, batch_label_length)
                        valid_accuracy += batch_accuracy
                        
                        # Update progress bar
                        pbar.set_postfix({"val_loss": loss_value.numpy(), "val_accuracy": batch_accuracy.numpy()})
                        pbar.update(1)
                
                # At the end of the validation, average the loss and accuracy
                valid_loss /= total_val_batches
                valid_accuracy /= total_val_batches
    
                # Print final validation metrics
                print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")
    

    # avg_loss = total_loss / total_batches
    # print(f"Training Loss: {avg_loss.numpy()}")
    # train_pbar.set_postfix({'training loss': avg_loss.numpy()})

    # # Validation loop
    # total_val_loss = 0.0
    # total_val_batches = 0
    
    # # Create a progress bar for the validation loop
    # valid_pbar = tqdm(valid_dataset, desc=f"Validating Epoch {epoch+1}/{epochs}", total=len(valid_images)//batch_size)
    
    # for batch, (batch_images, batch_padded_labels, batch_input_length, batch_label_length) in enumerate(valid_pbar):
    #     if batch_input_length.shape[0] != 8:
    #         continue
    #     # Forward pass for validation
    #     y_pred = model(batch_images, training=False)
        
    #     input_length_tensor = tf.reshape(batch_input_length, (8, 1))
    #     label_length_tensor = tf.reshape(batch_label_length, (8, 1))
        
    #     # Compute CTC loss
    #     val_loss_value = ctc_loss(batch_padded_labels, y_pred, input_length_tensor, label_length_tensor)
        
    #     total_val_loss += tf.reduce_sum(val_loss_value)
    #     total_val_batches += 1

    #     # Update validation progress bar with current validation loss
    #     valid_pbar.set_postfix({'val_loss': (total_val_loss / total_val_batches).numpy()})

    # avg_val_loss = total_val_loss / total_val_batches
    
    # train_pbar.set_postfix({'valid loss': avg_val_loss.numpy()})
    # # print(f"Validation Loss: {avg_val_loss.numpy()}")
    
    # # Checkpointing: Save the model if the validation loss improves
    # if avg_val_loss < best_val_loss:
    #     # print("Validation loss improved, saving the model...")
    #     best_val_loss = avg_val_loss
    #     train_pbar.set_postfix({'best loss': avg_val_loss.numpy()})
    #     model.save(filepath)

    # print("----- End of Epoch -----")
    
    
# Function to decode predictions and show accuracy after training
def display_final_predictions(model, data, char_list, original_texts, num_samples=20):
    print("\nDisplaying final predictions on sample data:")
    
    # Extract data for the sample
    sample_images, sample_padded_labels, sample_input_length, sample_label_length = data

    # Make predictions on sample images
    predictions = model.predict(sample_images[:num_samples])
    
    # Decode the predictions using CTC
    decoded = tf.keras.backend.ctc_decode(predictions,   
                           input_length=np.ones(predictions.shape[0]) * predictions.shape[1],
                           greedy=True)[0][0]

    out = tf.keras.backend.get_value(decoded)  # Get decoded outputs
    
    # Compare predictions with ground truth
    for i, x in enumerate(out[:num_samples]):
        print(f"Original text:  {original_texts[i]}")
        print("Predicted text: ", end='')
        
        # Convert predicted indices to characters
        for p in x:
            if int(p) != -1:  # Ignore the blank index (-1)
                print(char_list[int(p)], end='')
        print()
        
        # Display the corresponding image
        plt.imshow(tf.reshape(sample_images[i], (32, 128)), cmap=plt.cm.gray)
        plt.show()
        print("\n")

# Load a batch of data to test on after training is complete
def get_test_data(dataset, num_samples=20):
    # Fetch a small batch of validation data
    test_batch = next(iter(dataset))
    return test_batch[0][:num_samples], test_batch[1][:num_samples], test_batch[2][:num_samples], test_batch[3][:num_samples]

train_model(model, train_dataset, valid_dataset, epochs=1, checkpoint_dir='./best_model_checkpoints', resume_training=False)

# After training, call this function
# Assuming the model is already trained and `train_dataset`/`valid_dataset` is available

# Get some sample data for testing (you can use validation dataset for this)
sample_images, sample_labels, sample_input_length, sample_label_length = get_test_data(valid_dataset, num_samples=20)

# Original texts corresponding to the test data
test_original_texts = valid_original_text[:20]  # Adjust the indices based on your dataset

# Display predictions and accuracy after training
display_final_predictions(model=model, 
                          data=(sample_images, sample_labels, sample_input_length, sample_label_length), 
                          char_list=char_list, 
                          original_texts=test_original_texts)

# model.load_weights('adamo-10000r-5e-7850t-876v.keras')

# image = cv2.imread("/Users/dima/Downloads/note_2_2024-13_47_18.png", cv2.IMREAD_GRAYSCALE)
# image = process_image(image)
# image = image/255

# preds = model.predict(np.array([np.array([image]), the_labels, 1, label_length]).reshape(128, 32, 1))

# decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                    # greedy=True)[0][0])

# print(decoded)
