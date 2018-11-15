from keras import backend as K
from keras.losses import mean_squared_error
import numpy as np


def ramp_up_weight(ramp_period, weight_max):
    """Ramp-Up weight generator.
    The function is used in unsupervised component of loss.
    Returned weight ramps up until epoch reaches ramp_period
    """
    cur_epoch = 0

    while True:
        if cur_epoch <= ramp_period - 1:
            T = (1 / (ramp_period - 1)) * cur_epoch
            yield np.exp(-5 * (1 - T) ** 2) * weight_max
        else:
            yield 1 * weight_max

        cur_epoch += 1


def ramp_down_weight(ramp_period):
    """Ramp-Down weight generator"""
    cur_epoch = 1

    while True:
        if cur_epoch <= ramp_period - 1:
            T = (1 / (ramp_period - 1)) * cur_epoch
            yield np.exp(-12.5 * T ** 2)
        else:
            yield 0

        cur_epoch += 1


def semi_supervised_loss(num_class):
    """custom loss function"""
    epsilon = 1e-08

    def loss_func(y_true, y_pred):
        """semi-supervised loss function
        the order of y_true:
        unsupervised_target(num_class), supervised_label(num_class), supervised_flag(1), unsupervised weight(1)
        """
        unsupervised_target = y_true[:, 0:num_class]
        supervised_label = y_true[:, num_class:num_class * 2]
        supervised_flag = y_true[:, num_class * 2]
        weight = y_true[0, -1]

        model_pred = y_pred[:, 0:num_class]

        # weighted unsupervised loss over batchsize
        unsupervised_loss = weight * K.mean(mean_squared_error(unsupervised_target, model_pred))

        # To sum over only supervised data on categorical_crossentropy, supervised_flag(1/0) is used
        supervised_loss = - K.mean(
            K.sum(supervised_label * K.log(K.clip(model_pred, epsilon, 1.0 - epsilon)), axis=1) * supervised_flag)

        return supervised_loss + unsupervised_loss

    return loss_func


def update_weight(y, unsupervised_weight, next_weight):
    """update weight of the unsupervised part of loss"""
    y[:, -1] = next_weight
    unsupervised_weight[:] = next_weight

    return y, unsupervised_weight


def update_unsupervised_target(ensemble_prediction, y, num_class, alpha, cur_pred, epoch):
    """update ensemble_prediction and unsupervised weight when an epoch ends"""
    # Z = αZ + (1 - α)z
    ensemble_prediction = alpha * ensemble_prediction + (1 - alpha) * cur_pred

    # initial_epoch = 0
    y[:, 0:num_class] = ensemble_prediction / (1 - alpha ** (epoch + 1))

    return ensemble_prediction, y


def evaluate(model, num_class, num_test, test_x, test_y):
    """evaluate"""
    test_supervised_label_dummy = np.zeros((num_test, num_class))
    test_supervised_flag_dummy = np.zeros((num_test, 1))
    test_unsupervised_weight_dummy = np.zeros((num_test, 1))

    test_x_ap = [test_x, test_supervised_label_dummy, test_supervised_flag_dummy, test_unsupervised_weight_dummy]
    p = model.predict(x=test_x_ap)
    pr = p[:, 0:num_class]
    pr_arg_max = np.argmax(pr, axis=1)
    tr_arg_max = np.argmax(test_y, axis=1)
    cnt = np.sum(pr_arg_max == tr_arg_max)
    print('Test Accuracy: ', cnt / num_test, flush=True)
