'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

The RNN (LSTM) is trained to translate Roman numbers to Arabic, e.g.
IN: MDCCXXXI, OUT: 1731
IN: DCXLIX, OUT: 649
etc.

@author: pawel@kasprowski.pl
inspired by: https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
'''
import random
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
import roman_numerals as cnv
import numpy as np

'''
########################################################################################
Data generation = DATASET_SIZE samples with 
INPUT length input sequences
and
OUTPUT length output sequences
########################################################################################
'''

from encoder import CharacterTable 

# Parameters for the model and dataset.
DATASET_SIZE = 1000
INPUT = 14
OUTPUT = 4
# object to encode roman numbers to one-hot 
romans = 'MDCLXVI '
rtable = CharacterTable(romans)

# object to encode arabic numbers to one-hot
chars = '0123456789 '
dtable = CharacterTable(chars)

seq_samples = []
seq_labels = []
used = []
repetitions = 0
generated = 0
print('Generating data...')
while len(seq_samples) < DATASET_SIZE:
    number = random.randint(1,2000)
    # skip if already in the dataset
    if number in used: continue
    used.append(number)
    # roman input
    roman = cnv.convert(number) 
    roman += ' ' * (INPUT - len(roman))
    
    # arabic output
    arabic = str(number)
    arabic += ' ' * (OUTPUT - len(arabic)) 
    
    seq_samples.append(roman)
    seq_labels.append(arabic)
print('Total roman numbers:', len(seq_samples))

# one-hot encoding of all romans and arabic numbers
print('Vectorization...')
x = np.zeros((DATASET_SIZE, INPUT, len(romans)), dtype=np.bool)
y = np.zeros((DATASET_SIZE, OUTPUT, len(chars)), dtype=np.bool)
for i, sentence in enumerate(seq_samples):
    x[i] = rtable.encode(sentence, INPUT)
for i, sentence in enumerate(seq_labels):
    y[i] = dtable.encode(sentence, OUTPUT)

# Train / test split: 3/4 samples for training, 1/4 samples for test
split_at = 3*DATASET_SIZE//4
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

# Build the network and train it in a loop
print('Build model...')
model = Sequential()
model.add(layers.LSTM(128, input_shape=(INPUT, len(romans))))
model.add(layers.RepeatVector(OUTPUT))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation dataset.
for iteration in range(1, 500):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    H = model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_val, y_val), verbose=0)
    print("Accuracy:{} loss:{}".format(H.history["val_accuracy"],H.history["loss"]))
    # Select 10 samples from the validation set at random to visualize errors.
    accuracy = 0
    for i in range(len(x_val)):
        rowx, rowy = x_val[np.array([i])], y_val[np.array([i])]
        preds = model.predict_classes(rowx, verbose=0)
        roman = rtable.decode(rowx[0])
        correct = dtable.decode(rowy[0])
        prediction = dtable.decode(preds[0], calc_argmax=False)
        if correct == prediction:
            accuracy+=1
        import random
        v = random.random()
        if v>0.95:
          print('expression:', roman, end=' ')
          print('correct:', correct, end=' ')
          print('predicted:', prediction, end=' ')
          if correct == prediction:
              print('OK!')
          else:
              print('')
    print('accuracy=',accuracy/len(x_val)) 
input("done")        
model.save('model.h5')