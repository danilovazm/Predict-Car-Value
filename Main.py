import argparse
import tensorflow as tf
import numpy as np
from Network import NeuralNet
from PreProcess import preprocessData
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



def main(args):
    x_train, y_train, x_test, y_test = preprocessData(args.path)
    Net = NeuralNet()
    Net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=tf.keras.losses.MeanSquaredError())
    hist = Net.fit(x_train, y_train, epochs=args.epochs, verbose=1)
    plt.plot([i for i in range(1,args.epochs+1)], hist.history['loss'], color='blue')
    plt.show()
    pred = Net.predict(x_test)
    r2_rate = r2_score(pred, y_test)
    print(f"O r2 score é: {r2_rate}")
    if args.margin:
        count = 0
        for i in range(len(pred)):
            if abs(pred[i]-y_test.iloc[i]) <= args.threshold*y_test.iloc[i]:
                count +=1
        print(f"Acurácia de {count*100/len(y_test)}% usando uma margem de erro de {args.threshold*100}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./Car_Prices_Poland_Kaggle.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--margin', type=bool, default=True)
    parser.add_argument('--threshold', type=float, default=0.1)

    args = parser.parse_args()
    main(args)