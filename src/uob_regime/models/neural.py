"""Small LSTM baselines and a differentiable zero-order TSK expert gate."""

import itertools

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers


class GaussianFeatures(layers.Layer):
    """Append three trainable Gaussian memberships per numerical feature."""

    def __init__(self, centers, **kwargs):
        super().__init__(**kwargs)
        self.initial_centers = np.asarray(centers, dtype=np.float32)

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers", shape=self.initial_centers.shape,
            initializer=keras.initializers.Constant(self.initial_centers))
        self.raw_widths = self.add_weight(
            name="raw_widths", shape=self.initial_centers.shape, initializer="zeros")
        super().build(input_shape)

    def call(self, values):
        widths = tf.nn.softplus(self.raw_widths) + 1e-3
        membership = tf.exp(-0.5 * tf.square((values[..., None] - self.centers) / widths))
        flat = tf.reshape(membership, [tf.shape(values)[0], tf.shape(values)[1], -1])
        flat.set_shape((None, values.shape[1], self.initial_centers.size))
        return tf.concat([values, flat], axis=-1)


class TSKGate(layers.Layer):
    """Normalize 27 Gaussian rule activations and mix zero-order TSK consequents.

    Each rule has three constant consequents, one logit per expert. Softmax of
    the aggregated logits produces nonnegative expert weights summing to one.
    Expert identities are learned, rather than hard-coded regime assignments.
    """

    def __init__(self, centers, entropy_strength=0.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_centers = np.asarray(centers, dtype=np.float32)
        self.entropy_strength = entropy_strength
        combinations = list(itertools.product(range(3), repeat=3))
        self.rule_masks = tf.constant(np.eye(3, dtype=np.float32)[combinations])

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers", shape=(3, 3),
            initializer=keras.initializers.Constant(self.initial_centers))
        self.raw_widths = self.add_weight(name="raw_widths", shape=(3, 3), initializer="zeros")
        self.consequents = self.add_weight(
            name="consequents", shape=(27, 3),
            initializer=keras.initializers.RandomNormal(stddev=0.05))
        super().build(input_shape)

    def call(self, regime):
        widths = tf.nn.softplus(self.raw_widths) + 1e-3
        log_membership = -0.5 * tf.square((regime[..., None] - self.centers) / widths)
        log_firing = tf.einsum("bdk,rdk->br", log_membership, self.rule_masks)
        normalized_rules = tf.nn.softmax(log_firing, axis=-1)
        weights = tf.nn.softmax(tf.matmul(normalized_rules, self.consequents), axis=-1)
        entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-8), axis=-1)
        self.add_metric(tf.reduce_mean(entropy), name="gating_entropy", aggregation="mean")
        if self.entropy_strength:
            self.add_loss(-self.entropy_strength * tf.reduce_mean(entropy))
        return weights


class SeriesDecomposition(layers.Layer):
    """Split each feature into a centered moving average and a residual."""

    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = kernel_size

    def call(self, values):
        padding = self.kernel_size // 2
        front = tf.repeat(values[:, :1, :], padding, axis=1)
        back = tf.repeat(values[:, -1:, :], padding, axis=1)
        extended = tf.concat([front, values, back], axis=1)
        trend = tf.nn.avg_pool1d(
            extended, ksize=self.kernel_size, strides=1, padding="VALID")
        return values - trend, trend


class ANFISLayer(layers.Layer):
    """First-order Sugeno ANFIS with 27 Gaussian antecedent rules."""

    def __init__(self, centers, **kwargs):
        super().__init__(**kwargs)
        self.initial_centers = np.asarray(centers, dtype=np.float32)
        self.rule_masks = tf.constant(
            np.eye(3, dtype=np.float32)[list(itertools.product(range(3), repeat=3))])

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers", shape=(3, 3),
            initializer=keras.initializers.Constant(self.initial_centers))
        self.raw_widths = self.add_weight(
            name="raw_widths", shape=(3, 3), initializer="zeros")
        self.consequents = self.add_weight(
            name="consequents", shape=(27, 4),
            initializer=keras.initializers.RandomNormal(stddev=0.02))
        super().build(input_shape)

    def call(self, regime):
        widths = tf.nn.softplus(self.raw_widths) + 1e-3
        log_membership = -0.5 * tf.square((regime[..., None] - self.centers) / widths)
        log_firing = tf.einsum("bdk,rdk->br", log_membership, self.rule_masks)
        normalized_rules = tf.nn.softmax(log_firing, axis=-1)
        augmented = tf.concat([tf.ones_like(regime[:, :1]), regime], axis=1)
        rule_outputs = tf.einsum("bi,ri->br", augmented, self.consequents)
        return tf.reduce_sum(normalized_rules * rule_outputs, axis=1, keepdims=True)


def quantile_centers(values):
    """Estimate initial fuzzy centers from already scaled training data only."""
    return np.quantile(values, [0.2, 0.5, 0.8], axis=0).T.astype(np.float32)


def lstm_parameter_count(features, hidden):
    return 4 * hidden * (features + hidden + 1) + hidden + 1


def capacity_units(features, expert_units):
    target = 3 * lstm_parameter_count(features, expert_units) + 99
    return min(range(1, max(16, expert_units * 5)),
               key=lambda hidden: abs(lstm_parameter_count(features, hidden) - target))


def build_model(name, X_train, z_train, config):
    """Return a prediction model, an optional diagnostic model and metadata."""
    sequence = keras.Input(shape=X_train.shape[1:], name="sequence")
    regime = keras.Input(shape=(3,), name="regime")
    units = config["hidden_units"]
    diagnostics = None
    details = {"model": name, "hidden_units": units}
    if name in {"fuzzy_moe", "equal_moe"}:
        predictions = [layers.Dense(1, name=f"expert_{i + 1}_output")(
            layers.LSTM(units, name=f"expert_{i + 1}")(sequence)) for i in range(3)]
        experts = layers.Concatenate(name="expert_predictions")(predictions)
        if name == "fuzzy_moe":
            weights = TSKGate(quantile_centers(z_train), config["entropy_strength"],
                              name="tsk_gate")(regime)
        else:
            weights = layers.Lambda(lambda z: tf.ones_like(z) / 3.0,
                                    name="equal_weights")(regime)
        prediction = layers.Dot(axes=1, name="mixture")([experts, weights])
        diagnostics = keras.Model([sequence, regime], [experts, weights])
    elif name == "dlinear":
        seasonal, trend = SeriesDecomposition(kernel_size=5, name="decomposition")(sequence)
        seasonal = layers.Permute((2, 1))(seasonal)
        trend = layers.Permute((2, 1))(trend)
        seasonal_forecast = layers.Dense(1, name="seasonal_linear")(seasonal)
        trend_forecast = layers.Dense(1, name="trend_linear")(trend)
        channel_forecast = layers.Flatten()(layers.Add()([seasonal_forecast, trend_forecast]))
        prediction = layers.Dense(1, name="channel_mixer")(channel_forecast)
        details.update(decomposition_kernel=5, forecast_horizon=1)
    elif name == "anfis":
        prediction = ANFISLayer(quantile_centers(z_train), name="anfis")(regime)
        details.update(antecedent_inputs=["volatility", "trend", "sti_drawdown"],
                       membership_sets_per_input=3, rules=27,
                       consequent_order=1)
    else:
        hidden_input = sequence
        if name == "fuzzy_feature":
            centers = quantile_centers(X_train.reshape(-1, X_train.shape[-1]))
            hidden_input = GaussianFeatures(centers, name="fuzzy_features")(sequence)
        elif name == "capacity_lstm":
            units = capacity_units(X_train.shape[-1], units)
            details["hidden_units"] = units
        elif name != "lstm":
            raise ValueError(f"Not a neural model: {name}")
        prediction = layers.Dense(1, name="return_output")(
            layers.LSTM(units, name="lstm")(hidden_input))
    model = keras.Model([sequence, regime], prediction, name=name)
    if config["loss"] == "mse":
        loss = "mse"
    elif config["loss"] == "huber":
        loss = keras.losses.Huber()
    else:
        raise ValueError("loss must be mse or huber")
    model.compile(optimizer=keras.optimizers.Adam(config["learning_rate"]), loss=loss)
    details["parameters"] = model.count_params()
    details["capacity_reference_parameters"] = 3 * lstm_parameter_count(
        X_train.shape[-1], config["hidden_units"]) + 99
    return model, diagnostics, details
