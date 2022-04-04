import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

# model_dir = './'
out_path = './'
model_name = 'classify_gold.pb'

def create_graph():
    with tf.gfile.FastGFile(os.path.join(out_path + model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name)
