from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling1D, Add

def conv_block(inputs, filters, kernel_size, strides,
    pool_size, pool_strides):
    """
    args:
        inputs: a tensor of shape (input_dim, 1)
        filters(int): how many filters in that layer
        kernel_size(int): the size of the filter
        strides(int): the interval at which we apply the convolutional filter
        pool_size(int): the size of the pooling filters
        pool_strides(int): the interval at which we apply the pooling filter
    returns:
        a tensor
    """
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
                 padding="same", activation=None)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=pool_size, strides=pool_strides)(x)
    return x

def residual_block(inputs, filters, kernel_size, strides,
    pool_size, pool_strides):
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=1,
                 padding="same", activation=None)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=1,
                 padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([inputs, x])
    x = MaxPooling1D(pool_size=pool_size, strides=pool_strides)(x)
    return x
