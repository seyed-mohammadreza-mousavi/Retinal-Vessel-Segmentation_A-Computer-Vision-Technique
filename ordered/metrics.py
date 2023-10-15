from keras.losses import BinaryCrossentropy, mean_squared_error
from keras.metrics import Mean, MeanAbsoluteError, BinaryCrossentropy, F1Score

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(tp, tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(fp, tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(fn, tf.float32)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

class Precision(tf.keras.metrics.Metric):
    def __init__(self, name='precision', **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(tp, tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(fp, tf.float32)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        return precision

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)

class Sensitivity(tf.keras.metrics.Metric):
    def __init__(self, name='sensitivity', **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(tp, tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(fn, tf.float32)))

    def result(self):
        sensitivity = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return sensitivity

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_negatives.assign(0.)

class Specificity(tf.keras.metrics.Metric):
    def __init__(self, name='specificity', **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        tn = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))

        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(tn, tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(fp, tf.float32)))

    def result(self):
        specificity = self.true_negatives / (self.true_negatives + self.false_positives + K.epsilon())
        return specificity

    def reset_state(self):
        self.true_negatives.assign(0.)
        self.false_positives.assign(0.)
		
