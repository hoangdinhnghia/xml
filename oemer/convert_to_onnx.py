import sys
import json
import pickle
from pathlib import Path

import tf2onnx
import numpy as np
import tensorflow as tf

import oemer.train as omr_train

from tensorflow.keras.layers import MultiHeadAttention


def load_trained_model(input_path: Path):
    keras_file = input_path if input_path.suffix in {".keras", ".h5", ".hdf5"} else None
    if keras_file is not None:
        return tf.keras.models.load_model(
            keras_file,
            custom_objects={
                "MultiHeadAttention": MultiHeadAttention,
                "WarmUpLearningRate": omr_train.WarmUpLearningRate,
            },
            compile=False,
        )

    arch_path = input_path / "arch.json"
    weights_path = input_path / "model.weights.h5"
    if not weights_path.exists():
        weights_path = input_path / "weights.h5"
    if arch_path.exists() and weights_path.exists():
        model_config = json.loads(arch_path.read_text())
        model_config.pop("compile_config", None)
        model = tf.keras.models.model_from_json(
            json.dumps(model_config),
            custom_objects={
                "MultiHeadAttention": MultiHeadAttention,
                "WarmUpLearningRate": omr_train.WarmUpLearningRate,
            },
        )
        model.load_weights(weights_path)
        return model

    return tf.keras.models.load_model(
        input_path,
        custom_objects={
            "MultiHeadAttention": MultiHeadAttention,
            "WarmUpLearningRate": omr_train.WarmUpLearningRate,
        },
        compile=False,
    )


def convert(input_path, output_path=None):
    input_path = Path(input_path)
    model = load_trained_model(input_path)
    inp_shape = model.input_shape[1:]
    model(np.random.random((1,)+inp_shape))
    spec = (tf.TensorSpec(model.input_shape, tf.uint8, name="input"),)

    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    output_model = output_path / "model.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_model)
    output_names = [n.name for n in model_proto.graph.output]
    pickle.dump(
        {
            'output_names': output_names,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        },
        open(output_path / "metadata.pkl", "wb")
    )


if __name__ == "__main__":
    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert(model_path, output_path)
