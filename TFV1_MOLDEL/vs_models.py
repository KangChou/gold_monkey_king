import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()
tf.train.import_meta_graph("./model2/.meta")
tf.summary.FileWriter("./summary2", sess.graph)
