import tensorflow as tf
import tensorflow.keras as keras
# import numpy as np

test1 = tf.keras.layers.Dense(2)
test2 = tf.keras.layers.Dense(2)
test = test2(test1(tf.constant([[1,2,3]])))

print(dir(test2))
print(test2.weights)
print(test2.variables)
print(test2.get_config())
#input = tf.keras.Input(type_spec=tf.TensorSpec([5]))
#output = selector(input)
#model = tf.keras.Model(input, output)
#model.compile(loss=tf.keras.losses.mean_squared_error)
# config = tensorflowND.spatial_config([2], [2,3], [4, 5])
# instance = tf.random.normal([2,2,3,4,5])
# #instance = tf.constant([0,1], dtype = float)
#
# print("stitch_test")
#
# test = tf.random.normal([2,3,4])
# test2  = tf.random.normal([2,3, 3])
# access = [1,2]
# print(tf.dynamic_stitch(access, [test,test2]))
# reference = tensorflowND.reference_op.Reference_Op(config)()
# reference = tensorflowND.reference_op.Spatial_Kernel(config, 3, kernel_bias = 0)()
# #new_reference = tensorflowND.reference.Reference(config)()
# #print(new_reference)
#
#
# test = tf.constant([1,2,3,4])
# tf.print(test[1:-2])
# #selector = tensorflowND.selection.Selector(config)
# #selector.select(instance, reference)
# selector = tensorflowND.selection.Selector(config, reference)
# selector.select(instance, reference)
#
