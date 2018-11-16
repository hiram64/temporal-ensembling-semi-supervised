import argparse

import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from sklearn.utils import shuffle

from lib.ops import ramp_up_weight, semi_supervised_loss, update_unsupervised_target, evaluate, \
    ramp_down_weight, update_weight
from lib.utils import load_data, split_supervised_train, make_train_test_dataset, normalize_images, whiten_zca, \
    data_augmentation_tempen


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Ensembling')
    parser.add_argument('--data_path', default='./data/cifar10.npz', type=str, help='path to dataset')
    parser.add_argument('--num_labeled_train', default=4000, type=int,
                        help='the number of labeled data used for supervised training componet')
    parser.add_argument('--num_test', default=10000, type=int,
                        help='the number of data kept out for test')
    parser.add_argument('--num_class', default=10, type=int, help='the number of class')
    parser.add_argument('--num_epoch', default=351, type=int, help='the number of epoch')
    parser.add_argument('--batch_size', default=100, type=int, help='mini batch size')
    parser.add_argument('--ramp_up_period', default=80, type=int, help='ramp-up period of loss function')
    parser.add_argument('--ramp_down_period', default=50, type=int, help='ramp-down period')
    parser.add_argument('--weight_max', default=30, type=float, help='related to unsupervised loss component')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate of optimizer')
    parser.add_argument('--whitening_flag', default=True, type=bool, help='Whitening')
    parser.add_argument('--weight_norm_flag', default=True, type=bool,
                        help='Weight normalization is applied. Otherwise Batch normalization is applied')
    parser.add_argument('--augmentation_flag', default=True, type=bool, help='Data augmentation')
    parser.add_argument('--trans_range', default=2, type=int, help='random_translation_range')

    args = parser.parse_args()

    return args


def main():
    # Prepare args
    args = parse_args()

    num_labeled_train = args.num_labeled_train
    num_test = args.num_test
    ramp_up_period = args.ramp_up_period
    ramp_down_period = args.ramp_down_period
    num_class = args.num_class
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    weight_max = args.weight_max
    learning_rate = args.learning_rate
    weight_norm_flag = args.weight_norm_flag
    augmentation_flag = args.augmentation_flag
    whitening_flag = args.whitening_flag
    trans_range = args.trans_range

    # Data Preparation
    train_x, train_y, test_x, test_y = load_data(args.data_path)
    ret_dic = split_supervised_train(train_x, train_y, num_labeled_train)

    ret_dic['test_x'] = test_x
    ret_dic['test_y'] = test_y
    ret_dic = make_train_test_dataset(ret_dic, num_class)

    unsupervised_target = ret_dic['unsupervised_target']
    supervised_label = ret_dic['supervised_label']
    supervised_flag = ret_dic['train_sup_flag']
    unsupervised_weight = ret_dic['unsupervised_weight']
    test_y = ret_dic['test_y']

    train_x, test_x = normalize_images(ret_dic['train_x'], ret_dic['test_x'])

    # pre-process
    if whitening_flag:
        train_x, test_x = whiten_zca(train_x, test_x)

    if augmentation_flag:
        train_x = np.pad(train_x, ((0, 0), (trans_range, trans_range), (trans_range, trans_range), (0, 0)), 'reflect')

    # make the whole data and labels for training
    # x = [train_x, supervised_label, supervised_flag, unsupervised_weight]
    y = np.concatenate((unsupervised_target, supervised_label, supervised_flag, unsupervised_weight), axis=1)

    num_train_data = train_x.shape[0]

    # Build Model
    if weight_norm_flag:
        from lib.model_WN import build_model
        from lib.weight_norm import AdamWithWeightnorm
        optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    else:
        from lib.model_BN import build_model
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

    model = build_model(num_class=num_class)
    model.compile(optimizer=optimizer,
                  loss=semi_supervised_loss(num_class))

    model.metrics_tensors += model.outputs
    model.summary()

    get_prediction = K.function(inputs=[model.layers[0].input, K.learning_phase()],
                                outputs=[model.get_layer('activation_1').output])

    # prepare weights and arrays for updates
    gen_weight = ramp_up_weight(ramp_up_period, weight_max * (num_labeled_train / num_train_data))
    gen_lr_weight = ramp_down_weight(ramp_down_period)
    idx_list = [v for v in range(num_train_data)]

    # Training
    for epoch in range(num_epoch):
        print('epoch: ', epoch)
        idx_list = shuffle(idx_list)

        if epoch > num_epoch - ramp_down_period:
            weight_down = next(gen_lr_weight)
            K.set_value(model.optimizer.lr, weight_down * learning_rate)
            K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)

        ave_loss = 0
        for i in range(0, num_train_data, batch_size):
            target_idx = idx_list[i:i + batch_size]

            if augmentation_flag:
                x1 = data_augmentation_tempen(train_x[target_idx], trans_range)
            else:
                x1 = train_x[target_idx]

            x2 = supervised_label[target_idx]
            x3 = supervised_flag[target_idx]
            x4 = unsupervised_weight[target_idx]
            y_t = y[target_idx]

            # get the first prediction
            y_t[:, 0:num_class] = get_prediction(inputs=[x1, 1])[0]

            x_t = [x1, x2, x3, x4]
            tr_loss, output = model.train_on_batch(x=x_t, y=y_t)
            ave_loss += tr_loss

        print('Training Loss: ', (ave_loss * batch_size) / num_train_data, flush=True)

        # Update phase
        next_weight = next(gen_weight)
        y, unsupervised_weight = update_weight(y, unsupervised_weight, next_weight)

        # Evaluation
        if epoch % 5 == 0:
            print('Evaluate epoch :  ', epoch, flush=True)
            evaluate(model, num_class, num_test, test_x, test_y)


if __name__ == '__main__':
    main()
