import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class neural_network():
    def __init__(self, activation_choice):
         

    # Load text data 
        filePath = 'C:\\Users\\zaina\\OneDrive\\Documents\\Artificial Intelligence\\Assignment_5\\ANN_Iris_data.txt'
        irisData = pd.read_csv(filePath)

    # Assuming irisData is a DataFrame
    # Map string labels to integers
        self.label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        self.label_map_reverse = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2:'Iris-virginica'}
        self.colors = irisData.iloc[:, 4].map(self.label_map)
        
        X = irisData.iloc[:,:-1].values # all features
        Y = irisData.iloc[:,-1].map(self.label_map).values  # all labels
        x_train, x_temp, y_train_raw, y_temp_raw = train_test_split(X, Y, test_size=0.4, random_state=42, stratify=Y)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp_raw, test_size=0.5, random_state=42, stratify=y_temp_raw)

    # One hot encode after splitting
        self.y_train = np.eye(3)[y_train_raw]
        self.y_temp = np.eye(3)[y_temp_raw]
        self.y_val = np.eye(3)[y_val]
        self.y_test = np.eye(3)[y_test]
        self.x_train, self.x_val, self.x_test = x_train, x_val, x_test

    # Initializing parameters, weights and biases
        self.input_dim = 4
        self.hidden_dim = 10
        self.output_layer = 3
        self.epoch = 200
        self.learning_rate = 0.01
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_losses = []
        self.eps = 1e-15

        self.w1 = np.random.randn(self.hidden_dim, self.input_dim)
        self.b1 = np.random.randn(self.hidden_dim, 1)
        self.w2 = np.random.randn(self.output_layer, self.hidden_dim)
        self.b2 = np.random.randn(self.output_layer, 1)
        self.activation = self.ReLu if activation_choice == 1 else self.Sigmoid

    def classify_iris(self ):
        print("\n Let's classify your Iris plant!")
        try:
            sepal_length = float(input("Enter Sepal Length (cm): "))
            sepal_width = float(input("Enter Sepal Width (cm): "))
            petal_length = float(input("Enter Petal Length (cm): "))
            petal_width = float(input("Enter Petal Width (cm): "))
        except ValueError:
            print(" Invalid input. Please enter numeric values.")
            

        self.user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        y_pred, _, _ = self.forward_pass(self.user_input)
        predicted_class = np.argmax(y_pred, axis=0)[0]

        label_map_reverse = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        return print(f"\n Predicted Iris Type: {label_map_reverse[predicted_class]}")

    def ReLu(self, x):
            h = np.maximum(0, x)
            return h

    def Sigmoid(self, x):
            h = 1 / (1 + np.exp(-x))
            return h


    # Forward pass
    def softmax(self, a2):
        exp_a2 = np.exp(a2 - np.max(a2, axis = 0))
        return exp_a2 / np.sum(exp_a2, axis = 0)



    def forward_pass(self, X):
        # Layer 1  (10 x 4)
        a1 = np.dot(self.w1, X.T) + self.b1
        # Layer 1 activation (ReLU)
        h = self.activation(a1)
        # Layer 2
        a2 = np.dot(self.w2, h) + self.b2
        #  Softmax (converting to probabilities)
       
        y_pred = self.softmax(a2)
        return y_pred, h, a1

    # Backpropagation

    def backpropagation(self, X, y_true, y_pred, h, a1):
        eps = 1e-15
        # y_pred, h, a1 = self.forward_pass(X, self.w1, self.b1, self.w2, self.b2)
        # Compute the loss (cross-entropy)
        loss = -np.sum(y_true * np.log(y_pred.T + eps)) / X.shape[0]    # we transpsed y_pred.T to have shape of (3, 89), x.shape[0] does average across samples

        # Compute output layer error
        delta2 = y_pred - y_true.T   #(3, samples)
        # Gradient for w2 and b2 
        dw2 = np.dot(delta2, h.T) / X.shape[0] 
        db2 = np.sum(delta2, axis = 1, keepdims=True) / X.shape[0]

    # Compute hidden layer error (delta 1)
        if self.activation == self.ReLu:
            delta1 = np.dot(self.w2.T, delta2) * (a1 > 0)
        else:
            delta1 = np.dot(self.w2.T, delta2) * h * (1 - h)
        dw1 = np.dot(delta1, X) / self.x_train.shape[0]
        db1 = np.sum(delta1,  axis = 1, keepdims=True)/ X.shape[0]

    # Update weights and biases
        self.w1 = self.w1 - self.learning_rate * dw1
        self.b1 = self.b1 - self.learning_rate *db1
        self.w2 = self.w2 - self.learning_rate * dw2
        self.b2 = self.b2 - self.learning_rate * db2
        return loss

    # Optionally track and print the loss/accuracy every 10 epochs
    def train_validation(self):
        for i in range(self.epoch):
            y_pred, h, a1 = self.forward_pass(self.x_train)
            Loss = self.backpropagation(self.x_train, self.y_train, y_pred, h, a1)
            correct_predictions = np.argmax(y_pred, axis=0) == np.argmax(self.y_train.T, axis=0)    # np.argmax returns the index of the maximum value along the specified axis
            train_accuracy = np.mean(correct_predictions)
            self.train_accuracies.append(train_accuracy)
            self.train_losses.append(Loss)
            
            val_pred, _, _ = self.forward_pass(self.x_val)
            val_loss = -np.sum(self.y_val * np.log(val_pred.T + self.eps)) / self.x_val.shape[0]
            val_accuracy =  np.mean(np.argmax(val_pred, axis = 0) == np.argmax(self.y_val.T, axis = 0))
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            if i % 10 == 0:
                print(f"Epoch {i+1}, Train Loss: {Loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies

    def test_data(self):
        test_pred, _, _ = self.forward_pass(self.x_test)
        test_loss = -np.sum(self.y_test * np.log(test_pred.T + 1e-15)) / self.x_test.shape[0]
        test_accuracy = np.mean(np.argmax(test_pred, axis=0) == np.argmax(self.y_test.T, axis=0))
        label_map_reverse = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

        print("\n**** Final Test Performance: ****")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")


        print("\n🔍  **** Classification Results on Test Set ****:")
        for i in range(10):  # Show first 10 samples
            true_index = np.argmax(self.y_test[i])
            pred_index = np.argmax(test_pred[:, i])
            true_label = label_map_reverse[true_index]
            pred_label = label_map_reverse[pred_index]
            print(f"Sample {i+1}: True = {true_label}, Predicted = {pred_label}")




# Plotting 
    def plotting(self):
        plt.figure(figsize=(12, 5))    
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses,  label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', color='green')
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='brown')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

        


if __name__ == '__main__':

    activation_choice = int(input("Use  (1) Basic Experiment: [ReLU] (2) Extra Experiment: [Sigmoid]  "))
    Test_mode = str(input("Would you like to perform manual query; Yes or No: "))

    nn = neural_network(activation_choice)
    nn.train_validation()

    if Test_mode.lower() == 'yes':
        nn.classify_iris()
    else:
        print('Using test data')
        nn.test_data()
        nn.plotting()
    plt.show()






