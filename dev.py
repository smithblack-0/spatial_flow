import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import spatial_flow as tensorflowND


config = tensorflowND.spatial_config([2], [2,3], [4, 5])
instance = tf.random.normal([2,2,3,4,5])
#instance = tf.constant([0,1], dtype = float)

reference = tensorflowND.reference.Reference(config)()
reference = tensorflowND.reference.Spatial_Kernel(config, 3, kernel_bias = 0)()
#new_reference = tensorflowND.reference.Reference(config)()
#print(new_reference)


test = tf.constant([1,2,3,4])
tf.print(test[1:-2])
#selector = tensorflowND.selection.Selector(config)
#selector.select(instance, reference)
selector = tensorflowND.selection.Selector(config, reference)
selector.select(instance, reference)

