

def pre_processing(args, config):
    data_pkl = os.path.join(args.out_dir, 'data.pkl')
    if not os.path.exists(data_pkl):
        data_file = os.path.join(args.data_dir, config['data']['data_file'])
        df = read_dataframe(config, data_file)
        train_data, test_data = load_data(df, config)
        print('make pkl')
        with open(data_pkl, 'wb') as f:
            pickle.dump((train_data, test_data), f)
    else:
        print('load pkl')
        with open(data_pkl, 'rb') as f:
            train_data, test_data = pickle.load(f)
    print(train_data[0].shape, train_data[1].shape, test_data[0].shape, test_data[1].shape)
    return train_data, test_data


def train(args, config, train_data, test_data):
    model = build_model(config)
    model.summary()

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=config['optimizer']['learning_rate']))

    # early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # save weights
    save_weights = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, 'weights.hdf5'),
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      mode='min',
                                                      save_freq='epoch',
                                                      save_weights_only=True)
    # save weights
    csv_log = tf.keras.callbacks.CSVLogger(os.path.join(args.out_dir, 'history.csv'),
                                           separator=',',
                                           append=False)

    history = model.fit(train_data[0], train_data[1], epochs=args.epochs, batch_size=args.batch_size,
                        validation_data=test_data,
                        callbacks=[early_stopping, save_weights, csv_log])
    draw_history(args, history)


def eval(args, config, test_data):
    model = build_model(config)

    y_true = test_data[1][:,-1, 0]
    y_pred = model.predict(test_data[0])[:,-1, 0]

    draw_pred(args, y_true, y_pred)
    draw_error(args, y_true, y_pred)

    rmse = tf.sqrt(tf.keras.losses.MSE(y_true, y_pred))
    with open(os.path.join(args.out_dir, 'rsult.txt'), 'w') as f:
        f.write(f'rmse: {rmse}')
        f.write('\n')
    print(f'rmse: {rmse}')


def main(args, config):
    # data 전처리
    train_data, test_data = pre_processing(args, config)
    # train
    train(args, config, train_data, test_data)
    # 평가
    eval(args, config, test_data)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Train kma .")
    parser.add_argument("--config", type=str, default="config/temp_to_temp.yaml", required=False,
                        help="configuration file")
    parser.add_argument("--out_dir", type=str, default=None, required=False, help="result save directory")
    parser.add_argument("--data_dir", type=str, default="./data", required=False, help="data directory")
    parser.add_argument("--epochs", type=int, default=10, required=False, help="train epoch count")
    parser.add_argument("--batch_size", type=int, default=4096, required=False, help="train batch size")
    parser.add_argument("--seed", type=int, default=1234, required=False, help="random seed value")
    parser.add_argument("--scratch", action='store_true', help="scratch learning flag")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # make out_dir by config name
    if args.out_dir is None:
        basename, ext = os.path.splitext(os.path.basename(args.config))
        args.out_dir = os.path.join("result", basename)
    print(f"result save dir: " + args.out_dir)

    # output directory create
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"create {args.out_dir}")

    with open(args.config, encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    main(args, config)