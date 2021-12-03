import numpy as np
import matplotlib.pyplot as plt
import math



def accuracy(predictions, labels):
    preds_correct_boolean = np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy


def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    softmax_probs_array = np.where(softmax_probs_array == 0, 0.01, softmax_probs_array)
    y_onehot = np.where(softmax_probs_array == 0, 0.01, y_onehot)
    indices = np.argmax(y_onehot, axis=1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds) + 0.01
    return loss

def sigmoid_activation(x):
    expX = np.exp(x)
    return expX / np.sum(expX, axis=0)


def sigmoid_derivative(x):
    d = sigmoid_activation(x) * (1 - sigmoid_activation(x))
    np.nan_to_num(d)
    return d


df = [2, 33, 4]


sigmoid_activation(df)


def relu_activation(data_array):
    return np.maximum(data_array, 0)



def softmax(output_array):
    logits_exp = np.exp(output_array.astype(np.float32))
    return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)



def read_insurability(file_name):
    count = 0
    data = []
    with open(file_name, "rt") as f:
        for line in f:
            if count > 0:
                line = line.replace("\n", "")
                tokens = line.split(",")
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == "Good":
                        cls = 0
                    elif tokens[3] == "Neutral":
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls], [x1, x2, x3]])
            count = count + 1
    return data


def train_model_from_scratch(train, valid, test):

    X = []
    y = []

    for i in range(len(train)):
        X.append(np.array(train[i][1], dtype=np.float32))
        y.append(float(train[i][0][0]))
    X = np.array(X)
    new_y = []
    for i in y:
        if i == 0:
            new_y.append([1, 0, 0])
        if i == 1:
            new_y.append([0, 1, 0])
        if i == 2:
            new_y.append([0, 0, 1])
    y = np.array(new_y)

    X = np.where(X < 12, 10, X)

    np.random.seed(10)

    data = X
    labels = y

    hidden_nodes = 8
    num_labels = labels.shape[1]
    num_features = data.shape[1]

    learning_rate = .0001

    layer1_weights_array = np.random.normal(0, 1, [num_features, hidden_nodes])
    layer2_weights_array = np.random.normal(0, 1, [hidden_nodes, num_labels])

    for step in range(5500):
        input_layer = np.dot(data, layer1_weights_array)
        hidden_layer = sigmoid_activation(input_layer)
        output_layer = np.dot(hidden_layer, layer2_weights_array)
        output_probs = softmax(output_layer)

        loss = cross_entropy_softmax_loss_array(output_probs, labels)
        checker = np.isfinite(loss).all()

        output_error_signal = (output_probs - labels) / output_probs.shape[0]

        error_signal_hidden = np.dot(output_error_signal, layer2_weights_array.T)
        error_signal_hidden[hidden_layer <= 0] = 0

        gradient_layer2_weights = np.dot( sigmoid_derivative(hidden_layer.T), output_error_signal)

        gradient_layer1_weights = np.dot(data.T, error_signal_hidden)

        layer1_weights_array -= learning_rate * gradient_layer1_weights
        layer2_weights_array -= learning_rate * gradient_layer2_weights

        if step % 500 == 0:
            print('Loss at step {0}: {1}'.format(step, loss))


    X = []

    y = []

    for i in range(len(test)):
        X.append(np.array(test[i][1], dtype=np.float32))
        y.append(float(test[i][0][0]))
    X = np.array(X)
    new_y = []
    for i in y:
        if i == 0:
            new_y.append([1, 0, 0])
        if i == 1:
            new_y.append([0, 1, 0])
        if i == 2:
            new_y.append([0, 0, 1])
    y = np.array(new_y)

    input_layer = np.dot(X, layer1_weights_array)
    hidden_layer = relu_activation(input_layer)
    scores = np.dot(hidden_layer, layer2_weights_array)
    probs = softmax(scores)
    print("The accuracy is -> " + str(accuracy(probs, y)))