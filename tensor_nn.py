import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split


def convert_to_class(code_nm):
    init_class = -1
    if code_nm == '가로보':
        return init_class + 1
    elif code_nm == '교각':
        return init_class + 2
    elif code_nm == '교량받침':
        return init_class + 3
    elif code_nm == '교명포장':
        return init_class + 4
    elif code_nm == '난간연석':
        return init_class + 5
    elif code_nm == '바닥판':
        return init_class + 6
    elif code_nm == '배수시설':
        return init_class + 7
    elif code_nm == '신축이음':
        return init_class + 8
    elif code_nm == '주형':
        return init_class + 9
    return init_class


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def forwardprop(X, w_1, w_2):
    h = tf.nn.sigmoid(tf.matmul(X, w_1))
    yhat = tf.matmul(h, w_2)
    return yhat


def load_data(use_small_set=False, hot_encoding=False):
    feature_index = [7, 8, 9, 10, 11]
    class_index = 14
    file = pandas.read_excel('bridge_data_2.xlsx')
    y = np.array(list(map(convert_to_class, file[file.columns[class_index]].values)))
    x = file[file.columns[feature_index]].values.astype(float)
    if use_small_set:
        x, y = get_shriked_data(x, y)
    return (x, y)


def split_data(x, y, hot_encoding=False):
    if hot_encoding:
        n, m = x.shape
        all_x = np.ones((n, m + 1))
        all_x[:, 1:] = x
        num_class = len(np.unique(y))
        all_y = np.eye(num_class)[y]
    else:
        all_x = x
        all_y = y
    print(all_x.shape, all_y.shape)
    print(all_y)
    return train_test_split(all_x, all_y, test_size=0.33, random_state=42)


def get_shriked_data(x, y):
    # Hack for test
    index56 = []
    for i, j in enumerate(y):
        if j == 4 or j == 5:
            index56.append(i)
    idx_56 = np.asarray(index56)
    x56 = x[idx_56]
    y56 = y[idx_56] - 4
    return x56, y56


def run_nn(ax, train_x, test_x, train_y, test_y):
    # Layer's sizes
    x_size = train_x.shape[1]
    h_size = 4
    y_size = train_y.shape[1]
    iteration = 10
    nodes = []
    trained = []
    tested = []

    while h_size < 128:
        # Symbols
        X = tf.placeholder("float", shape=[None, x_size])
        y = tf.placeholder("float", shape=[None, y_size])

        # Weight initializations
        w_1 = init_weights((x_size, h_size))
        w_2 = init_weights((h_size, y_size))

        # Forward propagation
        yhat = forwardprop(X, w_1, w_2)
        predict = tf.argmax(yhat, axis=1)

        tf.set_random_seed(42)
        # Backward propagation
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
        updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        train_accuracy = 0
        test_accuracy = 0

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(iteration):
            # Train with each example
            for i in range(len(train_x)):
                sess.run(updates, feed_dict={X: train_x[i: i + 1], y: train_y[i: i + 1]})

            train_accuracy = np.mean(
                np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_x, y: train_y}))
            test_accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_x, y: test_y}))

            print("hidden node = ", h_size, ", Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        sess.close()
        nodes.append(h_size)
        trained.append(100 * train_accuracy)
        tested.append(100 * test_accuracy)
        h_size *= 2
    if ax != None:
        ax.plot(nodes, trained, label='train accuracy')
        ax.plot(nodes, tested, label='test accuracy')
        for i in range(len(nodes)):
            ax.annotate('{:.2f}%'.format(trained[i]), xy=(nodes[i], trained[i]), textcoords='data', color='blue')
            ax.annotate('{:.2f}%'.format(tested[i]), xy=(nodes[i], tested[i]), textcoords='data', color='red')
        ax.set_xlabel("N hidden nodes")
        ax.set_ylabel("Accuracy")
        ax.legend()
    return np.max(tested)


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
    ax1.set_title("9 classes")
    ax2.set_title("2 classes")
    full_data = load_data()
    small_data = load_data(use_small_set=True)
    # 2layer MLP
    run_nn(ax1, *split_data(*full_data, hot_encoding=True))
    run_nn(ax2, *split_data(*small_data, hot_encoding=True))
    # Result
    plt.savefig('mlp.png')
    plt.show()


if __name__ == '__main__':
    main()