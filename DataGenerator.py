import numpy
import math
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


# Generate data from crazy function
def get_data(query_list):
    data = list()
    for query in query_list:
        data.append((query, math.sqrt(2 * query) + math.sin(3 * query) + math.cos(2 * query)))

    # Break up into training, validation, and testing
    last_idx = len(data) - 1
    training = data[0: int(math.floor(0.7 * last_idx))]
    validation = data[int(math.floor(0.7 * last_idx)) + 1: int(math.floor(0.85 * last_idx)) - 1]
    testing = data[int(math.floor(0.85 * last_idx)) + 1: last_idx]
    # (xs, ys)
    training_set = ([pt[0] for pt in training], [pt[1] for pt in training])
    validation_set = ([pt[0] for pt in validation], [pt[1] for pt in validation])
    testing_set = ([pt[0] for pt in testing], [pt[1] for pt in testing])

    train_set_x = theano.shared(numpy.asarray(training_set[0], dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(numpy.asarray(training_set[1], dtype=theano.config.floatX), borrow=True)

    valid_set_x = theano.shared(numpy.asarray(validation_set[0], dtype=theano.config.floatX), borrow=True)
    valid_set_y = theano.shared(numpy.asarray(validation_set[1], dtype=theano.config.floatX), borrow=True)

    testing_set_x = theano.shared(numpy.asarray(testing_set[0], dtype=theano.config.floatX), borrow=True)
    testing_set_y = theano.shared(numpy.asarray(testing_set[1], dtype=theano.config.floatX), borrow=True)

    return train_set_x, train_set_y, valid_set_x, valid_set_y, testing_set_x, testing_set_y


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.params = [self.W, self.b]

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            activation(lin_output)
        )


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.relu
        )

        self.outputLayer = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            activation=T.nnet.relu
        )

        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            T.sum(self.hiddenLayer.W ** 2)
            + T.sum(self.outputLayer.W ** 2)
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.output = (
            self.outputLayer.output
        )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.outputLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input

    def error(self, y):
        return T.mean(T.sqr(y - self.output))


def test_reg(
        learning_rate=0.1,
        L1_reg=0.00,
        L2_reg=0.01,
        n_epochs=80,
        data_generator=get_data,
        batch_size=20,
        n_hidden=100):

    query_list = numpy.linspace(0, 20, 120)
    query_list = numpy.random.permutation(query_list)

    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = data_generator(query_list)

    index = T.lscalar()
    x = T.dscalar('query')
    y = T.dscalar('out')

    rng = numpy.random.RandomState(1234)

    regression = MLP(
        rng=rng,
        input=x,
        n_in=1,
        n_hidden=n_hidden,
        n_out=1
    )

    cost = (
        regression.error(y)
        + L2_reg * regression.L2_sqr
    )

    test_model = theano.function(
        inputs=[index],
        outputs=[regression.error(y), regression.output],
        givens={
            x: test_set_x[index],
            y: test_set_y[index]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=[regression.error(y), regression.output],
        givens={
            x: valid_set_x[index],
            y: valid_set_y[index]
        }
    )

    gparams = [T.grad(cost, param) for param in regression.params]

    updates = [
        (param, param - learning_rate * gparam) for param, gparam in zip(regression.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=[cost, regression.output],
        updates=updates,
        givens={
            x: train_set_x[index],
            y: train_set_y[index]
        }
    )

    best_validation_error = numpy.inf
    improvement_threshold = 0.999
    epoch = 0
    test_score = 0
    done_looping = False
    validation_frequency = 30

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1

        for index in range(len(train_set_x.container.data)):
            cost, output = train_model(index)
        validation_error = 0
        for valid in range(len(valid_set_x.container.data)):
            validation_error += validate_model(valid)
        print('epoch = ' + str(epoch) + ' validation error = ' + str(validation_error))

        if validation_error < best_validation_error:
            if validation_error < best_validation_error:
                best_validation_error = validation_error
            #else:
            #    done_looping = True

    # Test on test set
    test_error = list()
    output = list()
    for test in range(len(test_set_x.container.data)):
        error, val = test_model(test)
        test_error.append(error)
        output.append(val[0][0])
    print('test error = ' + str(sum(test_error)))

    # Plot the data
    plt.title("data")
    plt.scatter(train_set_x.container.data, train_set_y.container.data, c='b', hold=True)
    # plt.scatter([pt[0] for pt in validation], [pt[1] for pt in validation], c='g', hold=True)
    # plt.scatter([pt[0] for pt in testing], [pt[1] for pt in testing], c='r', hold=True)
    plt.scatter(test_set_x.container.data, test_set_y.container.data, c='r')
    plt.scatter(test_set_x.container.data, output, c='g')
    plt.show()

if __name__ == "__main__":
    test_reg()
