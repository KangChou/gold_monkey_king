#转换成pb文件
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os.path
import argparse
from tensorflow.framework import graph_util


def freeze_graph(output_node_names):
    #checkpoint = tf.train.get_checkpoint_state(model_folder)
    #input_checkpoint = checkpoint.model_checkpoint_path

    saver = tf.train.import_meta_graph('./model/.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, './model/')
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names
        )
        with tf.gfile.GFile('classify-car.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    freeze_graph(['output'])
