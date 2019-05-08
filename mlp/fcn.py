import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer
from tensorflow.contrib.metrics import streaming_accuracy, streaming_mean_relative_error
from tensorflow.contrib.slim import batch_norm
from tensorflow.python.ops import control_flow_ops
from config_reader import get_task_sections


class FCN:

    def __init__(self, config, params):

        self.column_size = None
        self.total_column_size = params['total_columns']

        self.input_features_slicer = config.get_as_slice(self.total_column_size, "FEATURES", "columns")

        # self.l1_reg = [config.getfloat("TRAINING", "l1_regularization", fallback=0.0)]
        # self.l2_reg = [config.getfloat("TRAINING", "l2_regularization", fallback=0.0)]
        self.l1_reg = params['l1_reg']
        self.l2_reg = params['l2_reg']
        self.learning_rate = params['learning_rate']
        self.l1_l2_regularizer = lambda t: tf.add(l1_regularizer(self.l1_reg)(t), l2_regularizer(self.l2_reg)(t))
        # self.num_hidden_units = config.getint("NETWORK", "layer_size")
        # self.num_layers = config.getint("NETWORK", "num_layers")
        self.num_hidden_units = params['layer_size']
        self.num_layers = params['num_layers']
        # self.learning_rate = config.getfloat("TRAINING", "learning_rate")
        self.is_residual = config.getboolean("TRAINING", "residual", fallback=False)

        self.batch_norm = config.getboolean("NETWORK", "batch_norm", fallback=True)

        self.optimizer = config.get("TRAINING", "optimizer")

        self.config_task_sections = get_task_sections(config)

        self.reg_ground_truth_slicer = None
        self.ground_truth_slicer = None
        self.populate_ground_truth(self.config_task_sections, config)

        self.add_placeholders()

    def populate_ground_truth(self, task_config, config):
        for task in task_config:
            if config.get(task, 'type') == 'linear':
                self.reg_ground_truth_slicer = config.get_as_slice(self.total_column_size, task, "ground_truth_column")
            elif config.get(task, 'type') == 'classification':
                self.ground_truth_slicer = config.get_as_slice(self.total_column_size, task, "ground_truth_column")
                self.multilabel = config.get(task, 'multilabel') == 'True'
                self.stratified = config.get(task, 'stratified') == 'True'

    def variable_summaries(self, var, name, task_tag):
        """Attach a lot of summaries to a Tensor."""
        tf.summary.scalar("%s_%s" % (task_tag, name), var)

    def make_hidden_FN_layers(self, input_layer):
        previous_out = input_layer
        normalizer_fun = batch_norm if self.batch_norm else None

        with tf.variable_scope("hidden_layers"):
            for i in range(1, self.num_layers + 1):
                with tf.variable_scope("layer%d" % i) as layer_scope:
                    if self.is_residual and i > 1:
                        previous_out = tf.add(previous_out, tf.ones_like(previous_out))
                    normalizer_params = {"scale": i == self.num_layers, "is_training": self.is_training,
                                         "decay": 0.99} if self.batch_norm else None
                    previous_out = fully_connected(previous_out,
                                                   self.num_hidden_units, activation_fn=tf.nn.relu,
                                                   normalizer_fn=normalizer_fun,
                                                   normalizer_params=normalizer_params,
                                                   weights_regularizer=self.l1_l2_regularizer,
                                                   scope=layer_scope)

                    # TODO no dropout in between (discuss with Yanli)
                    # if i % 2 == 0:
                    if i == self.num_layers:
                        previous_out = tf.nn.dropout(previous_out, self.keep_prob)

        last_hidden_layer = previous_out
        return last_hidden_layer

    def add_multiple_classification_output_layer(self, last_hidden_layer, gt_labels, num_classes, corpus_tag, task_tag,
                                        loss_weight=1):
        with tf.variable_scope("output_layer_%s" % task_tag) as layer_scope:
            output_dim = gt_labels.shape[1].value
            last_out = fully_connected(last_hidden_layer, output_dim, activation_fn=tf.identity,
                                       weights_regularizer=self.l1_l2_regularizer,
                                       scope=layer_scope)
            self.predictions = tf.map_fn(tf.nn.sigmoid, last_out)

        with tf.name_scope("%s_%s_stats" % (corpus_tag, task_tag)):
            loss = loss_weight * tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_labels, logits=last_out))

            self.variable_summaries(loss, "loss", task_tag)

            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

            str_accu, _ = streaming_accuracy(tf.map_fn(tf.rint, last_out), gt_labels, name="stracc_%s" % corpus_tag,
                                             updates_collections=tf.GraphKeys.UPDATE_OPS)

            str_accu = 100 * str_accu
            # utils.variable_summaries(str_accu, "streaming_accuracy", corpus_tag)
            self.variable_summaries(str_accu, "streaming_accuracy", task_tag)

            updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.streaming_accu_op = control_flow_ops.with_dependencies(updates_op, str_accu)

            correct_prediction = tf.equal(tf.map_fn(tf.rint, last_out), gt_labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

            auc = tf.metrics.auc(gt_labels, self.predictions)
            # utils.variable_summaries(accuracy, "accuracy", corpus_tag)
            self.variable_summaries(accuracy, "accuracy", task_tag)
            self.accuracy = accuracy
            self.auc = auc

    def add_classification_output_layer(self, last_hidden_layer, gt_labels, num_classes, corpus_tag, task_tag,
                                           loss_weight=1):
        with tf.variable_scope("output_layer_%s" % task_tag) as layer_scope:
            last_out = fully_connected(last_hidden_layer, num_classes, activation_fn=tf.identity,
                                       weights_regularizer=self.l1_l2_regularizer,
                                       scope=layer_scope)
            self.predictions = tf.nn.softmax(last_out)

        with tf.name_scope("%s_%s_stats" % (corpus_tag, task_tag)):
            loss = loss_weight * tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_labels, logits=last_out))
            # utils.variable_summaries(loss, "loss", corpus_tag)
            self.variable_summaries(loss, "loss", task_tag)

            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

            str_accu, _ = streaming_accuracy(tf.argmax(last_out, 1), gt_labels, name="stracc_%s" % corpus_tag,
                                             updates_collections=tf.GraphKeys.UPDATE_OPS)
            str_accu = 100 * str_accu
            # utils.variable_summaries(str_accu, "streaming_accuracy", corpus_tag)
            self.variable_summaries(str_accu, "streaming_accuracy", task_tag)

            updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.streaming_accu_op = control_flow_ops.with_dependencies(updates_op, str_accu)

            correct_prediction = tf.equal(tf.argmax(last_out, 1), gt_labels)

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
            a = tf.cast(tf.argmax(last_out, 1), tf.float32)
            b = tf.one_hot(gt_labels, num_classes)

            # print(tf.argmax(last_out, 1).dtype.name)
            # print(gt_labels.dtype.name)
            # TODO: double check
            auc = tf.metrics.auc(b, self.predictions)
            # auc = tf.contrib.metrics.streaming_auc(a, b)
            # utils.variable_summaries(accuracy, "accuracy", corpus_tag)
            self.variable_summaries(accuracy, "accuracy", task_tag)
            self.accuracy = accuracy
            self.auc = auc

    def add_linear_output_layer(self, last_hidden_layer, ground_truth, corpus_tag, task_tag, loss_weight=1):
        with tf.variable_scope("output_layer_%s" % task_tag) as layer_scope:
            output_dim = ground_truth.shape[1].value if len(ground_truth.shape) > 1 else 1
            last_out = fully_connected(last_hidden_layer, output_dim, activation_fn=tf.identity,
                                       weights_regularizer=self.l1_l2_regularizer,
                                       scope=layer_scope)
            last_out = tf.squeeze(last_out)
            self.predictions = last_out

        with tf.name_scope("%s_%s_stats" % (corpus_tag, task_tag)):
            loss = loss_weight * tf.reduce_mean(tf.squared_difference(last_out, ground_truth))

            # utils.variable_summaries(loss, "loss", corpus_tag)
            self.variable_summaries(loss, "loss", task_tag)
            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

            str_mre, _ = streaming_mean_relative_error(last_out, ground_truth, ground_truth,
                                                       updates_collections=tf.GraphKeys.UPDATE_OPS)
            str_accu = tf.multiply(100.0, (1.0 - str_mre), name="acc_%s" % corpus_tag)
            # utils.variable_summaries(str_accu, "streaming_accuracy", corpus_tag)
            self.variable_summaries(str_accu, "streaming_accuracy", task_tag)

            updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.streaming_accu_op = control_flow_ops.with_dependencies(updates_op, str_accu)

            accuracy = 100 * (1 -
                              tf.reduce_mean(tf.where(
                                  tf.equal(ground_truth, tf.zeros_like(ground_truth)),
                                  tf.zeros_like(ground_truth),
                                  tf.divide(tf.abs(last_out - ground_truth), ground_truth)))
                              )
            # utils.variable_summaries(accuracy, "accuracy", corpus_tag)
            self.variable_summaries(accuracy, "accuracy", task_tag)
            self.accuracy = accuracy


    def add_all_outputs_and_losses(self, input_features, ground_truth, reg_ground_truth, corpus_tag):
        hidden_output = self.make_hidden_FN_layers(input_features)
        for task_name, task_config in self.config_task_sections.items():
            if task_config["type"] == "linear":
                task_name += "_lin"
                self.add_linear_output_layer(hidden_output, reg_ground_truth, corpus_tag, task_name)
            elif task_config["type"] == "classification":
                task_name += "_classf"
                num_classes = int(task_config["num_classes"])

                if self.multilabel == True:
                    self.add_multiple_classification_output_layer(hidden_output, ground_truth, num_classes, corpus_tag,
                                                     task_name)
                else:
                    self.add_classification_output_layer(hidden_output, ground_truth, num_classes, corpus_tag,
                                                     task_name)
            else:
                assert False
        losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.LOSSES)) \
                 + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return losses

    def make_placeholders(self, column_size):
        # Fix the shapes
        input_features = tf.placeholder(dtype=tf.float32, shape=(None, column_size))

        input_labels = None
        reg_input_labels = None

        if self.ground_truth_slicer:
            labels_columns = list(range(self.total_column_size))[self.ground_truth_slicer]
            if type(labels_columns) is list:
                num_labels = len(labels_columns)
            else:
                num_labels = 1
            if num_labels > 1:
                input_labels = tf.placeholder(dtype=tf.float32, shape=(None, num_labels))

        if input_labels is None:
            input_labels = tf.placeholder(dtype=tf.int64, shape=(None,))

        if self.reg_ground_truth_slicer:
            labels_columns = list(range(self.total_column_size))[self.reg_ground_truth_slicer]
            if type(labels_columns) is list:
                num_labels = len(labels_columns)
            else:
                num_labels = 1
            if num_labels > 1:
                reg_input_labels = tf.placeholder(dtype=tf.float32, shape=(None, num_labels))

        if reg_input_labels is None:
            reg_input_labels = tf.placeholder(dtype=tf.float32, shape=(None,))

        return input_features, input_labels, reg_input_labels

    def bind_graph_dataframe(self, corpus_tag, input_data_cols, batch_size, reuse=False, with_training_op=False):

        if self.column_size is None:
            self.column_size = input_data_cols.iloc[:, self.input_features_slicer].shape[1]
            self.input_features_placeholder, self.input_label_placeholder, self.reg_input_placeholder = \
                self.make_placeholders(self.column_size)

        with tf.variable_scope("network", reuse=reuse):
            # self.Y_logits = self.make_FN_layers()
            loss_sum = self.add_all_outputs_and_losses(self.input_features_placeholder,
                                                       self.input_label_placeholder,
                                                       self.reg_input_placeholder,
                                                       corpus_tag)

            updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.loss = control_flow_ops.with_dependencies(updates_op, loss_sum)  # all losses
            if with_training_op:
                self.train_op = self.add_optimizer(type=self.optimizer)

            all_weight_vars = [tf.reshape(var, [-1]) for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if
                               "/weights" in var.name]
            tf.summary.histogram("weight_hist", tf.concat(axis=0, values=all_weight_vars),
                                 collections=["%s_summaries" % corpus_tag])

            # self.summaries_merged = self.get_summaries(corpus_tag)

    def bind_graph(self, corpus_tag, input_data_cols, batch_size, reuse=False, with_training_op=False):
        # Builds all ops that correspond to the NN graph and its evaluators and optimizers.
        # Needs the input data Tensors/Queues as argument
        # Any of the built ops, e.g. self.loss,
        # operates on the input data (given as arguments to this function) every time it is called.
        # If reuse=True , the TF graph is not built, but simply reused from the memory with the most recent parameters.

        input_features = tf.reshape(tf.transpose(tf.stack(input_data_cols[self.input_features_slicer])),
                                    [batch_size, -1])

        with tf.variable_scope("network", reuse=reuse):
            # self.Y_logits = self.make_FN_layers()
            loss_sum = self.add_all_outputs_and_losses(input_features,
                                                       input_data_cols,
                                                       corpus_tag)

            updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.loss = control_flow_ops.with_dependencies(updates_op, loss_sum)  # all losses
            if with_training_op:
                self.train_op = self.add_optimizer(type=self.optimizer)

            all_weight_vars = [tf.reshape(var, [-1]) for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if
                               "/weights" in var.name]
            tf.summary.histogram("weight_hist", tf.concat(axis=0, values=all_weight_vars),
                                 collections=["%s_summaries" % corpus_tag])

            # self.summaries_merged = self.get_summaries(corpus_tag)

    def add_placeholders(self):
        """
        Returns:
        """
        with tf.name_scope("Hypers"):
            self.keep_prob = tf.placeholder(dtype=tf.float32)
            self.is_training = tf.placeholder(dtype=tf.bool, shape=[])

    def add_optimizer(self, type="vanilla"):
        """
        Add the optimizer function to perform Gradient Descent
        Args:
            type: The type of update that is needed
                  ["vanilla", "adam", "adagrad", "rmsprop"]
        Returns: None
        """
        if type not in ["vanilla", "adam", "adagrad", "rmsprop"]:
            raise ValueError("Please provide any of [vanilla, adam, adagrad, rmsprop] for optimisation")

        with tf.name_scope("gradient_descent"):
            optimizer = None
            if type == "vanilla":
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif type == "adam":
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif type == "adagrad":
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif type == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

            grads_and_vars = optimizer.compute_gradients(self.loss)

            capped_grads_and_vars = [(tf.clip_by_value(g, -10, 10), v) for g, v in grads_and_vars]
            return optimizer.apply_gradients(capped_grads_and_vars)

    def get_summaries(self):
        # return tf.summary.merge(tf.get_collection("%s_summaries" % corpus_tag))
        return tf.summary.merge_all()
