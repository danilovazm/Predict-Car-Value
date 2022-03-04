import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class NeuralNet(Model):
  def __init__(self):
    super(NeuralNet, self).__init__()
    self.fc1 = Dense(128, activation='relu')
    self.fc2 = Dense(128, activation='relu')
    self.fc3 = Dense(128, activation='relu')
    self.fc4 = Dense(128, activation='relu')
    self.fc5 = Dense(1)

  def call(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)
    x = self.fc5(x)
    return x
