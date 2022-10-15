# Predict-Car-Value
Using Tensorflow Framework to train a NN to predict car value in Poland.

Part of the dataset preprocessing was obtained from this kaggle https://www.kaggle.com/nguyenthanhlong/car-price-prediction-keras-nn-random-forest

## About

The main goal of this project is to create a neural network (NN) to predict the value of car.

### Dataset
For this project this [public dataset](https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland) from kaggle was used.

### Implementation
The code was implemented in python using tensorflow, numpy, pandas, scikit-learn, matplotlib libraries. The dataset was preprocessed with some columns dropped and categorical objects were transformed into numerical objects the dataset was also splited into train/test using 80/20 ratio. The NN used was simple, the architecture is compose by 5 consecutives dense layers and the first 4 layers use ReLU as activation function.

### Training
The training of the NN was perform through 20 epochs using the Adam optimizer with a learning rate of ``` 0.0001 ``` and the objective function was MSE. 

### Summary
`Main.py`: file that calls the training and testing of the model based on users choice, prints and plots the model's result.

`Network.py`: file that contains the definition of the model architecture.

`train.py`: file where the training routine is implemented.

`test.py`: file where the testing routine is implemented.

## How To
The python version used in this project was:
```
    Python 3.7.3
```

First you will need to install the following dependencies:
``` 
    pip install tensorflow
    pip install numpy
    pip install pandas
    pip install scikit-learn
    pip install matplotlib
```
Once the dependencies were installed to run the code go to the root of this repository and execute the command below:
``` 
    python Main.py
```
Also, when running the Main.py file, is possible to fit some methods' parameters. To do so choose at least one of the following flags defining the desired value:

`--epochs`: define the number of epochs you want to perform the training, default `100`.

`--lr`: Choose the learning rate you want to use, default `0.0001`.

`--margin`: Choose if you want to calculate de accuracy, default `True`.

`--threshold`: Define de error margin to calculate de accuracy, default `0.1`.

## Results

The r2 score was:
``` 
r2 score: 0.9217704236025149
```

The accuracy using 0.1 error margin:
``` 
accuracy: 44.61121003985415%
```

