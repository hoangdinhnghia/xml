import sys
import time
import os
import shutil
from multiprocessing import freeze_support

import tensorflow as tf

import oemer.train as omr_train
from oemer import classifier


FAST_CPU_PRESET = {
    "epochs": 6,
    "steps": 600,
    "val_steps": 80,
    "early_stop": 3,
    "batch_size": 6,
    "val_batch_size": 6,
    "num_worker": 2,
}

#train giả, train test
GPU_4GB_FAST_PRESET = {
    "epochs": 4,
    "steps": 300,
    "val_steps": 50,
    "early_stop": 2,
    "batch_size": 1,
    "val_batch_size": 1,
    "num_worker": 2,
}

#train thật
GPU_4GB_PRESET = {
    "epochs": 10,
    "steps": 900,
    "val_steps": 120,
    "early_stop": 4,
    "batch_size": 1,
    "val_batch_size": 1,
    "num_worker": 4,
}

FULL_CPU_PRESET = {
    "epochs": 15,
    "steps": 1500,
    "batch_size": 6,
    "num_worker": 2,
}


def write_text_to_file(text, path):
    with open(path, "w") as f:
        f.write(text)


def get_model_base_name(model_name: str) -> str:
    timestamp = str(round(time.time()))
    return f"{model_name}_{timestamp}"


def save_arch_and_weights(model, filename):
    """Save model architecture and weights in Keras 3 compatible format."""
    os.makedirs(filename, exist_ok=True)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    
    # Keras 3 requires .weights.h5 suffix; save then copy for backward compatibility
    keras_weights = os.path.join(filename, "model.weights.h5")
    model.save_weights(keras_weights)
    # Keep backward-compatible artifact name used across this repository.
    shutil.copyfile(keras_weights, os.path.join(filename, "weights.h5"))


def resolve_unet_dataset_path():
    """Resolve UNet dataset path, trying multiple possible names."""
    candidates = ["CvcMuscima-Distortions", "CvcMuscima"]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # If neither exists, return the default (will fail with helpful error message)
    return candidates[0]


def prepare_classifier_data():
    if not os.path.exists("train_data"):
        classifier.collect_data(2000)


def train_and_save(dataset_path, model_type, **kwargs):
    model = omr_train.train_model(dataset_path, data_model=model_type, **kwargs)
    filename = get_model_base_name(model_type)
    save_arch_and_weights(model, filename)


def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <model_name>")
        print("Examples: unet, segnet, unet_fast, segnet_fast, unet_gpu4g, segnet_gpu4g")
        sys.exit(1)

    model_type = sys.argv[1]

    if model_type == "segnet":
        train_and_save("ds2_dense", model_type, **FULL_CPU_PRESET)
    elif model_type == "unet":
        dataset_path = resolve_unet_dataset_path()
        train_and_save(dataset_path, model_type, **FULL_CPU_PRESET)
    elif model_type == "segnet_fast":
        train_and_save("ds2_dense", "segnet", **FAST_CPU_PRESET)
    elif model_type == "unet_fast":
        dataset_path = resolve_unet_dataset_path()
        train_and_save(dataset_path, "unet", **FAST_CPU_PRESET)
    elif model_type == "segnet_gpu4g":
        train_and_save("ds2_dense", "segnet", **GPU_4GB_PRESET)
    elif model_type == "unet_gpu4g":
        dataset_path = resolve_unet_dataset_path()
        train_and_save(dataset_path, "unet", **GPU_4GB_PRESET)
    elif model_type == "segnet_gpu4g_fast":
        train_and_save("ds2_dense", "segnet", **GPU_4GB_FAST_PRESET)
    elif model_type == "unet_gpu4g_fast":
        dataset_path = resolve_unet_dataset_path()
        train_and_save(dataset_path, "unet", **GPU_4GB_FAST_PRESET)
    elif model_type == "unet_from_checkpoint" or model_type == "segnet_from_checkpoint":
        model = tf.keras.models.load_model("seg_unet", custom_objects={"WarmUpLearningRate": omr_train.WarmUpLearningRate})
        filename = get_model_base_name(model_type.split("_")[0])
        save_arch_and_weights(model, filename)
    elif model_type == "rests_above8":
        prepare_classifier_data()
        classifier.train_rests_above8(get_model_base_name(model_type))
    elif model_type == "rests":
        prepare_classifier_data()
        classifier.train_rests(get_model_base_name(model_type))
    elif model_type == "all_rests":
        prepare_classifier_data()
        classifier.train_all_rests(get_model_base_name(model_type))
    elif model_type == "sfn":
        prepare_classifier_data()
        classifier.train_sfn(get_model_base_name(model_type))
    elif model_type == "clef":
        prepare_classifier_data()
        classifier.train_clefs(get_model_base_name(model_type))
    else:
        print("Unknown model: " + model_type)
        sys.exit(1)


if __name__ == "__main__":
    freeze_support()
    main()