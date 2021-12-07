import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense,Embedding,GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RNN, LSTM, RepeatVector
import time
from datetime import datetime

start_time = time.time()
print("Kipling emulator")
def to_text(sample):
  return ''.join([idx2char[int(x)] for x in sample])


def generate_text(model, start_string, size=1000):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = size

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  #print(input_eval.shape)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0
    
  input_eval_64 = input_eval
  for i in range(63):
    input_eval_64 = np.vstack((input_eval_64,input_eval))
  input_eval = input_eval_64

# Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      full_predictions = predictions
      predictions = predictions[0]
      # remove the batch dimension
      #predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      input_eval_64 = input_eval
      for i in range(63):
        input_eval_64 = np.vstack((input_eval_64,input_eval))
      input_eval = input_eval_64
  
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


fin = open('kipling.txt', 'rb')
dataset_txt = fin.read().decode(encoding='utf-8')
fin.close()

# obtain the unique characters in the dataset and print out their length 
vocab = sorted(set(dataset_txt))
print(vocab)
print ('{} unique characters'.format(len(vocab)))
# Creating a mapping from unique characters to indices
char2idx = {char:index for index, char in enumerate(vocab)}
print('char2idx:\n',char2idx)
idx2char = np.array(vocab)
print('idx2char\n',idx2char)
vocab_size = len(vocab)

# Let's convert our dataset from 'characters' to 'integers'
dataset_int = np.array([char2idx[char] for char in dataset_txt])

LEN=100
samples = []
labels = []
for i in range(0,len(dataset_int)-LEN,LEN):
    samples.append(dataset_int[i:LEN+i])
    labels.append(dataset_int[(i+1):(LEN+i+1)])
#print(samples[:10])
samples = np.array(samples,dtype=float)
labels = np.array(labels,dtype=float)
#samples = np.array(samples)
#labels = np.array(labels)


def build_model():
  model = tf.keras.Sequential()
  model.add(Embedding(vocab_size, 256, batch_input_shape=[64, None]))
  model.add(LSTM(1024, return_sequences=True,
                        stateful=True,#!!!
                        recurrent_initializer='glorot_uniform'))
  model.add(Dense(vocab_size))
  return model  
        
model = build_model()
model.summary()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss) #loss='sparse_categorical_crossentropy')


import random
def sample_from_dataset(n,samples,labels):
    prev_numbers = []
    new_samples = []
    new_labels = []
    while len(new_samples)<n:
    #for i in range(n):
        number = random.randrange(len(samples))
        if number in prev_numbers: continue
        prev_numbers.append(number)
        new_samples.append(samples[number])
        new_labels.append(labels[number])
    new_samples = np.array(new_samples)    
    new_labels = np.array(new_labels)
    return new_samples,new_labels


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))


EPOCHS = 10
num_epochs = 0
last_loss=0
print('running...')
import codecs
for i in range(10):
    s,l = sample_from_dataset(64,samples,labels)
    #s = np.asarray(s).astype('float32')
    #l = np.asarray(l).astype('float32')
    H = model.fit(s,l,epochs=EPOCHS,verbose=0,batch_size=64)
    num_epochs += EPOCHS
 
    print(50*'=')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    print("Epoch {} - loss={:6.3f}".format(num_epochs,H.history['loss'][-1]))
    last_loss = H.history['loss'][-1]

    txt = generate_text(model, start_string="Mowgli ",size=500)

    print(txt)
    print('..saving to file')
#    f = open("output.txt", "a")
    f = codecs.open("output.txt", "a", "utf-8")
    f.write("\n=================================================================================\n")
    f.write("Epoch {} - loss ={:6.3f} time = {:.0f} s\n".format(num_epochs,H.history['loss'][-1],time.time()-start_time))
    f.write(txt)
    f.write('\n')
    f.close()	
    model.save('models/model_{}.h5'.format(num_epochs))
EPOCHS = 100
#num_epochs = 0
#last_loss=0
print('running...')
import codecs
for i in range(1000):
    s,l = sample_from_dataset(64,samples,labels)
    #s = np.asarray(s).astype('float32')
    #l = np.asarray(l).astype('float32')
    H = model.fit(s,l,epochs=EPOCHS,verbose=0,batch_size=64)
    num_epochs += EPOCHS

    print(50*'=')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
 
    print("Epoch {} - loss={:6.3f}".format(num_epochs,H.history['loss'][-1]))
    last_loss = H.history['loss'][-1]

    txt = generate_text(model, start_string="Mowgli ",size=500)

    print(txt)
    print('..saving to file')
    print()
#    f = open("output.txt", "a")
    f = codecs.open("output.txt", "a", "utf-8")
    f.write("\n=================================================================================\n")
    f.write("Epoch {} - loss ={:6.3f} time = {:.0f} s\n".format(num_epochs,H.history['loss'][-1],time.time()-start_time))
    f.write(txt)
    f.write('\n')
    f.close()	
    model.save('models/model_{}.h5'.format(num_epochs))
print('done!')    

