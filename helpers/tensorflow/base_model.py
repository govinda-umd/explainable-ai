'''
defines meta structure of any machine learning model.
i.e. defines -
0. batch size, slicing the input (for both training/testing)
1. training procedure: loss function, regularizer, optimizer, 
2. evaluation procedure: evaluation metric
'''


import numpy as np
import tensorflow as tf


class base_model():
    def __init__(self, task_type, model, loss_object, L1_scale, L2_scale, optimizer, 
                 eval_metric, batch_size, slice_input, eval_metric_name):
        
        assert(task_type in ["classification", "regression"])
        
        self.task_type = task_type
        self.model = model
        self.loss_object = loss_object
        self.L1_regularizer = tf.keras.regularizers.L1(L1_scale)
        self.L2_regularizer = tf.keras.regularizers.L2(L2_scale)
        self.L1L2_regularizer = tf.keras.regularizers.L1L2(l1=L1_scale, l2=L2_scale) 
        self.optimizer = optimizer
        self.loss_avg = tf.keras.metrics.Mean()
        self.eval_metric = eval_metric
        self.batch_size = batch_size
        
        self.slice_input = slice_input
        self.eval_metric_name = eval_metric_name
        
    def _loss(self, x, y, training):
        # y_._keras_mask ensures that loss ignores the masked timesteps/samples
        y_ = self.model(x, training=training)
        loss = self.loss_object(y_true=y[y_._keras_mask],
                                y_pred=y_[y_._keras_mask])
        for var in self.model.trainable_variables:
            loss += self.L1L2_regularizer(var)
        return loss

    def _grad(self, inputs, targets, training):
        with tf.GradientTape() as tape:
            loss_value = self._loss(inputs, targets, training=training)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def fit(self, train_X, train_Y, val_X, val_Y, num_epochs):

        template = ("Epoch {:03d}: "
                "Train Loss: {:.3f}, " 
                "Train {eval_metric_name}: {:.3%}  "
                "Val Loss: {:.3f}, " 
                "Val {eval_metric_name}: {:.3%}  ")

        train_loss = np.zeros((num_epochs,), dtype=np.float32)
        train_eval_metric = np.zeros((num_epochs,), dtype=np.float32)
        val_loss = np.zeros((num_epochs,), dtype=np.float32)
        val_eval_metric = np.zeros((num_epochs,), dtype=np.float32)

        for epoch in range(num_epochs):
            # print(f"epoch: {epoch}")
            # training step
            train_loss[epoch], train_eval_metric[epoch] = self.evaluate(inputs=train_X,
                                                                        outputs=train_Y,
                                                                        is_training=True)
            # validation step
            val_loss[epoch], val_eval_metric[epoch] = self.evaluate(inputs=val_X,
                                                                    outputs=val_Y,
                                                                    is_training=False)
            if epoch % 1 == 0:
                print(template.format(epoch,
                                      train_loss[epoch],
                                      train_eval_metric[epoch],
                                      val_loss[epoch],
                                      val_eval_metric[epoch],
                                      eval_metric_name=self.eval_metric_name))


        return {'train_loss':train_loss, 
                'train_eval_metric':train_eval_metric, 
                'val_loss':val_loss, 
                'val_eval_metric':val_eval_metric}

    def evaluate(self, inputs, outputs, is_training):

        self.loss_avg.reset_states()
        self.eval_metric.reset_states()

        num_samples = outputs.shape[0]

        # using batches of <self.batch_size>
        for i in range(0, num_samples, self.batch_size):

            x = self.slice_input(inputs, 
                                 i, i + self.batch_size)
            y = outputs[i: i + self.batch_size, ...]

            # training/loss calculation --------
            if is_training:
                # Optimize the model
                loss_value, grads = self._grad(x, y, is_training)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            else:
                loss_value = self._loss(x, y, is_training)

            # Track progress
            self.loss_avg.update_state(loss_value)  # Add current batch loss

            # evaluation metric --------------
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            y_pred = self.model(x, training=is_training)
            mask = y_pred._keras_mask
            if self.task_type == "regression":
                self.eval_metric.update_state(tf.keras.backend.flatten(y[mask]), 
                                              tf.keras.backend.flatten(y_pred[mask]))
            elif self.task_type == "classification":
                self.eval_metric.update_state(y[mask],
                                              y_pred[mask])

        # print(f"loss_value: {loss_value}")
        return (self.loss_avg.result(), self.eval_metric.result())

        '''
    # @tf.function
    def train_step(x, y):
        is_training=True
        
        # Optimize the model
        loss_value, grads = self._grad(x, y, is_training)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Track progress
        self.loss_avg.update_state(loss_value)  # Add current batch loss

        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        if self.task_type is "regression":
            self.eval_metric.update_state(tf.keras.backend.flatten(y), 
                                          tf.keras.backend.flatten(self.model(x, training=is_training)))
        elif self.task_type is "classification":
            self.eval_metric.update_state(y,
                                          self.model(x, training=is_training))
        
        return (self.loss_avg.result(), self.eval_metric.result())
    
    # @tf.function
    def test_step(x, y):
        is_training=False
        
        loss_value = self._loss(x, y, is_training)
        
        # Track progress
        self.loss_avg.update_state(loss_value)
        if self.task_type is "regression":
            self.eval_metric.update_state(tf.keras.backend.flatten(y), 
                                          tf.keras.backend.flatten(self.model(x, training=is_training)))
        elif self.task_type is "classification":
            self.eval_metric.update_state(y,
                                          self.model(x, training=is_training))
        
        return 
    '''