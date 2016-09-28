import numpy as np
import math
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


# Generate data from crazy function
from theano.gof.opt import optimizer


def get_data(query_list):
    data = list()
    for query in query_list:
        #data.append((query, math.sqrt(2 * query) + math.sin(3 * query) + math.cos(2 * query)))
        data.append((query, query ** 3))

    # Break up into training, validation, and testing
    last_idx = len(data) - 1
    training = data[0: int(math.floor(0.7 * last_idx))]
    validation = data[int(math.floor(0.7 * last_idx)) + 1: int(math.floor(0.85 * last_idx)) - 1]
    testing = data[int(math.floor(0.85 * last_idx)) + 1: last_idx]
    # (xs, ys)
    training_set = ([pt[0] for pt in training], [pt[1] for pt in training])
    validation_set = ([pt[0] for pt in validation], [pt[1] for pt in validation])
    testing_set = ([pt[0] for pt in testing], [pt[1] for pt in testing])

    train_set_x = np.asarray(training_set[0])
    train_set_y = np.asarray(training_set[1])

    valid_set_x = np.asarray(validation_set[0])
    valid_set_y = np.asarray(validation_set[1])

    testing_set_x = np.asarray(testing_set[0])
    testing_set_y = np.asarray(testing_set[1])

    return train_set_x, train_set_y, valid_set_x, valid_set_y, testing_set_x, testing_set_y


def update_line(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data))
    hl.set_ydata(np.append(hl.get_ydata(), new_data))
    plt.draw()


if __name__ == "__main__":
    input_size = 1
    output_size = 1
    number_of_hidden_units = 3
    regularization_factor = 0.01
    learning_rate = 0.1

    x = T.dmatrix('x')
    y = T.dmatrix('y')

    rng = np.random.RandomState(1234)

    # For input to hidden layer
    W1_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (input_size + number_of_hidden_units)),
                    high=np.sqrt(6. / (input_size + number_of_hidden_units)),
                    size=(input_size, number_of_hidden_units)
                ),
                dtype=theano.config.floatX
            )
    W1 = theano.shared(value=W1_values, name='W1', borrow=True)
    b1_values = np.zeros((input_size, number_of_hidden_units), dtype=theano.config.floatX)
    b1 = theano.shared(value=b1_values, name='b1', borrow=True)

    # For hidden to output layer
    W2_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (number_of_hidden_units + output_size)),
                    high=np.sqrt(6. / (number_of_hidden_units + output_size)),
                    size=(number_of_hidden_units, output_size)
                ),
                dtype=theano.config.floatX
            )
    W2 = theano.shared(value=W2_values, name='W2', borrow=True)

    b2_values = np.zeros((output_size,1), dtype=theano.config.floatX)
    b2 = theano.shared(value=b2_values, name='b2', borrow=True)

    # Output of layer hidden layer
    layer_1_out = T.nnet.relu(T.add(T.dot(x, W1), b1))
    #layer_1_out = T.nnet.sigmoid(T.add(T.dot(x, W1), b1))



    # Output of output layer
    final_output = T.nnet.relu(T.add(T.dot(layer_1_out, W2), b2))
    #final_output = T.nnet.sigmoid(T.add(T.dot(layer_1_out, W2), b2))

    L2_regularization = (
        T.add(T.sum(W1 ** 2), T.sum(W2 ** 2))
    )

    cost = T.mean(T.add((y - final_output) ** 2, regularization_factor * L2_regularization))
    #cost = T.mean(y - final_output) ** 2

    dw1, dw2, db1, db2 = T.grad(cost, [W1, W2, b1, b2])

    neural_network_train = theano.function(
        inputs=[x, y],
        outputs=[layer_1_out, final_output, cost],
        updates=[
            [W1, W1 - learning_rate * dw1],
            [W2, W2 - learning_rate * dw2],
            [b1, b1 - learning_rate * db1],
            [b2, b2 - learning_rate * db2]
        ],
        allow_input_downcast=True
    )

    neural_network_test = theano.function(
        inputs=[x, y],
        outputs=[final_output, cost],
        allow_input_downcast=True
    )

    query_list = np.linspace(0, 1, 1000)
    query_list = np.random.permutation(query_list)

    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = get_data(query_list)

    hl, = plt.plot([], [])
    errors = list()
    best_predictions = list()
    validation_error = np.inf
    for epoch in range(100):
        # train
        preds = []
        for iteration in range(len(train_set_x)):
            l1, pred, cost = neural_network_train([[train_set_x[iteration]]], [[train_set_y[iteration]]])
            preds.append(pred)
        #plt.scatter(train_set_x, train_set_y, c='r')
        #plt.scatter(train_set_x, preds, c='g')
        #plt.show()
        cost_lst = list()
        # validate
        preds = []
        for iteration in range(len(valid_set_x)):
            pred, cost = neural_network_test([[valid_set_x[iteration]]], [[valid_set_y[iteration]]])
            cost_lst.append(cost)
            preds.append(pred[0][0])
        error = sum(cost_lst)
        epoch += 1
        errors.append(error)
        if error < validation_error:
            validation_error = error
            best_predictions = preds
        else:
            break
        preds = []

    plt.plot(errors, 'r')
    plt.show()
    plt.scatter(valid_set_x, valid_set_y, c='r')
    plt.scatter(valid_set_x, best_predictions, c='g')
    plt.show()

    predictions = list()
    for iteration in range(len(test_set_x)):
        pred, cost = neural_network_test([[test_set_x[iteration]]], [[test_set_y[iteration]]])
        predictions.append(pred)

    plt.scatter(test_set_x, test_set_y, c='b')
    plt.scatter(test_set_x, predictions, c='g')
    plt.show()
