class CustomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer=optimizer

        self.bce_metric = BinaryCrossentropy(name="bce")
        self.val_bce_metric = tf.keras.metrics.BinaryCrossentropy(name="Bce")

        self.f1_metric = F1Score(name="f1Score")
        self.val_f1_metric = F1Score(name="F1score")

        self.prec_metric = Precision(name="prec")
        self.val_prec_metric = Precision(name="Prec")

        self.se_metric = Sensitivity(name="se")
        self.val_se_metric = Sensitivity(name="Se")

        self.sp_metric = Specificity(name="sp")
        self.val_sp_metric = Specificity(name="Sp")
    
    def compile(self, optimizer, loss):
        super().compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            #loss = BinaryCrossEntropy(y, y_pred)
            #loss=mean_squared_error(y, y_pred)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.bce_metric.update_state(y, y_pred)
        self.f1_metric.update_state(tf.cast(y, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32))
        self.prec_metric.update_state(y, y_pred)
        self.se_metric.update_state(y, y_pred)
        self.sp_metric.update_state(y, y_pred)

        return {"bce": self.bce_metric.result(), "f1score": self.f1_metric.result(), "prec": self.prec_metric.result(),
                "se": self.se_metric.result(), "sp": self.sp_metric.result()}

        def test_step(self, data):
          x, y = data

          y_pred = self(x, training=False)
          loss = self.compute_loss(y, y_pred)
          
          self.val_bce_metric.update_state(y, y_pred)
          self.val_f1_metric.update_state(y, y_pred)
          self.val_prec_metric.update_state(y, y_pred)
          self.val_se_metric.update_state(y, y_pred)

          return {"Bce": self.val_bce_metric.result(), "F1_score": self.val_f1_metric.result(), "Prec": self.val_prec_metric.result(),
                  "Se": self.val_se_metric.result(), "Sp": self.val_sp_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.bce_metric, self.val_bce_metric, self.f1_metric, self.val_f1_metric,
                self.prec_metric, self.val_prec_metric, self.se_metric, self.val_se_metric,
                self.sp_metric, self.val_sp_metric]
				
				
x_input = Input(shape=(patch_size, patch_size, 3), name="input_img")
x_linear = LinearTransform()(x_input)

x_resinit = ResBlock(16,residual_path=True)(x_linear)
x_up = UpSampling2D(size=(2,2),interpolation='bilinear')(x_resinit)
x_resup = ResBlock(32,residual_path=True)(x_up)

x1 = MaxPool2D(pool_size=(2,2))(x_resup)
x1 = ResBlock(64,residual_path=True)(x1)
x1 = ResBlock(64,residual_path=False)(x1)

x2 = MaxPool2D(pool_size=(2,2))(x1)
x2 = ResBlock(128,residual_path=True)(x2)
x2 = ResBlock(128,residual_path=False)(x2)

x3 = MaxPool2D(pool_size=(2,2))(x2)
x3 = ResBlock(256,residual_path=True)(x3)
x3 = ResBlock(256,residual_path=False)(x3)

x4 = MaxPool2D(pool_size=(2,2))(x3)
x4 = ResBlock(512,residual_path=True)(x4)

x3 = Concatenate(axis=3)([x3, UpSampling2D(size=(2,2),interpolation='bilinear')(x4)])
x3 = ResBlock(256,residual_path=True)(x3)
x3 = ResBlock(256,residual_path=False)(x3)

x2 = Concatenate(axis=3)([x2, UpSampling2D(size=(2,2),interpolation='bilinear')(x3)])
x2 = ResBlock(128,residual_path=True)(x2)
x2 = ResBlock(128,residual_path=False)(x2)

x1 = Concatenate(axis=3)([x1, UpSampling2D(size=(2,2),interpolation='bilinear')(x2)])
x1 = ResBlock(64,residual_path=True)(x1)

x = Concatenate(axis=3)([x_resup, UpSampling2D(size=(2,2),interpolation='bilinear')(x1)])
x = ResBlock(32,residual_path=True)(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = ResBlock(32)(x)
x = Conv2D(1,kernel_size=1,strides=1,padding='same',use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('sigmoid')(x)

unet_model = CustomModel(x_input, x, name="Unet")	

# Learning rate and optimizer
cosine_decay = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=LR, first_decay_steps=12000,t_mul=1000,m_mul=0.5,alpha=1e-5)
optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay)
loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
unet_model.compile(optimizer=optimizer, loss=loss)