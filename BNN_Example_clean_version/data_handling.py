VAL_SPLIT = 0.1
FRACTION = 0.1


def apply_data_settings_keras(new_dataset):
    (x_train, y_train), (x_test, y_test) = new_dataset
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # Split training data into train and validation sets
    val_size = int(len(x_train) * VAL_SPLIT)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    return x_train, y_train, x_val, y_val, x_test, y_test
