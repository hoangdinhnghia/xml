import random
import pickle
import warnings
import argparse
from sklearn.exceptions import InconsistentVersionWarning
from os import remove
from pathlib import Path
from PIL import Image

import augly.image as imaugs
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

from oemer.bbox import get_bbox, merge_nearby_bbox, draw_bounding_boxes, rm_merge_overlap_bbox
from oemer.build_label import find_example


SVM_PARAM_GRID = {
    'degree': [2, 3, 4],
    'decision_function_shape': ['ovo', 'ovr'],
    'C':[0.1, 1, 10, 100],
    'gamma':[0.0001, 0.001, 0.1, 1],
    'kernel':['rbf', 'poly']
}
TARGET_WIDTH = 40
TARGET_HEIGHT = 70
DISTANCE = 10

DATASET_PATH = "./ds2_dense/segmentation"


def _collect(color, out_path, samples=100):
    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir()

    cur_samples = 0
    add_space = 10
    idx = 0
    while cur_samples < samples:
        arr = find_example(DATASET_PATH, color)
        if arr is None:
            continue
        arr[arr!=200] = 0
        boxes = get_bbox(arr)
        if len(boxes) > 1:
            boxes = merge_nearby_bbox(boxes, DISTANCE)
        boxes = rm_merge_overlap_bbox(boxes)
        for box in boxes:
            if idx >= samples:
                break
            print(f"{idx+1}/{samples}", end='\r')
            patch = arr[box[1]-add_space:box[3]+add_space, box[0]-add_space:box[2]+add_space]
            ratio = random.choice(np.arange(0.6, 1.3, 0.1))
            tar_w = int(ratio * patch.shape[1])
            tar_h = int(ratio * patch.shape[0])
            img = imaugs.resize(Image.fromarray(patch.astype(np.uint8)), width=tar_w, height=tar_h)

            seed = random.randint(0, 1000)
            np.float = float  # Monkey patch to workaround removal of np.float
            img = imaugs.perspective_transform(img, seed=seed, sigma=3)
            img = np.where(np.array(img)>0, 255, 0)
            Image.fromarray(img.astype(np.uint8)).save(out_path / f"{idx}.png")
            idx += 1

        cur_samples += len(boxes)
    print()


def collect_data(samples=400):
    color_map = {
        74: "sharp",
        70: "flat",
        72: "natural",
        97: 'rest_whole',
        98: 'rest_half',
        99: 'rest_quarter',
        100: 'rest_8th',
        101: 'rest_16th',
        102: 'rest_32nd',
        103: 'rest_64th',
        10: 'gclef',
        13: 'fclef',
    }

    for color, name in color_map.items():
        print('Current', name)
        _collect(color, f"train_data/{name}", samples=samples)
        _collect(color, f"test_data/{name}", samples=samples)


def train(folders):
    class_map = {idx: Path(ff).name for idx, ff in enumerate(folders)}
    train_x = []
    train_y = []
    samples = None
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        for ff in folder.glob('*.png'):
            if samples is not None and idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img).flatten()
            train_x.append(arr)
            train_y.append(cidx)
            idx += 1

    print("Train model")
    model = svm.SVC()#C=0.1, gamma=0.0001, kernel='poly', degree=2, decision_function_shape='ovo')
    #model = AdaBoostClassifier(n_estimators=50)
    #model = BaggingClassifier(n_estimators=50)  # For sfn classification
    #model = RandomForestClassifier(n_estimators=50)
    #model = GradientBoostingClassifier(n_estimators=50, verbose=1)
    #model = GridSearchCV(svm.SVC(), SVM_PARAM_GRID)
    #model = KNeighborsClassifier(n_neighbors=len(folders))#, weights='distance')
    model.fit(train_x, train_y)
    return model, class_map

def build_class_map(folders):
    return {idx: Path(ff).name for idx, ff in enumerate(folders)}

def train_tf(folders):
    import tensorflow as tf
    class_map = build_class_map(folders)
    train_x = []
    train_y = []
    samples = None
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        for ff in folder.iterdir():
            if samples is not None and idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img)
            train_x.append(arr)
            train_y.append(cidx)
            idx += 1
    train_x = np.array(train_x)[..., np.newaxis]
    train_y = tf.one_hot(train_y, len(folders))
    output_types = (tf.uint8, tf.uint8)
    output_shapes = ((TARGET_HEIGHT, TARGET_WIDTH, 1), (len(folders)))
    dataset = tf.data.Dataset.from_generator(lambda: zip(train_x, train_y), output_types=output_types, output_shapes=output_shapes)
    dataset = dataset.shuffle(len(train_y), reshuffle_each_iteration=True)
    dataset = dataset.repeat(5)
    dataset = dataset.batch(16)

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(folders), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, batch_size=16, epochs=10)
    return model, class_map


def test(model, folders):
    test_x = []
    test_y = []
    samples = 100
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        files = list(folder.glob('*.png'))
        random.shuffle(files)
        for ff in files:
            if idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img).flatten()
            test_x.append(arr)
            test_y.append(cidx)
            idx += 1

    pred_y = model.predict(test_x)
    tp_idx = (pred_y == test_y)
    tp = len(pred_y[tp_idx])
    acc = tp / len(test_y)
    print("Accuracy: ", acc)


def test_tf(model, folders):
    test_x = []
    test_y = []
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        files = list(folder.iterdir())
        random.shuffle(files)
        for ff in files:
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img)
            test_x.append(arr)
            test_y.append(cidx)

    test_x = np.array(test_x)[..., np.newaxis]
    test_y = np.array(test_y)
    test_result = []
    batch_size = 16
    for idx in range(0, len(test_x), batch_size):
        data = test_x[idx:idx+batch_size]
        pred = model.predict(data)
        pidx = np.argmax(pred, axis=-1)
        test_result.extend(list(pidx))

    test_result = np.array(test_result)
    tp = test_result[test_result==test_y].size
    acc = tp / len(test_y)
    print("Accuracy: ", acc)


def predict(region, model_name):
    if np.max(region) == 1:
        region *= 255
    # suppress sklearn InconsistentVersionWarning when unpickling models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        m_info = pickle.load(open(f"sklearn_models/{model_name}.model", "rb"))
    model = m_info['model']
    w = m_info['w']
    h = m_info['h']
    region = Image.fromarray(region.astype(np.uint8)).resize((w, h))
    pred = model.predict(np.array(region).reshape(1, -1))
    return m_info['class_map'][pred[0]]

def train_rests_above8(filename = "rests_above8.model"):
    folders = ["rest_8th", "rest_16th", "rest_32nd", "rest_64th"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_rests(filename = "rests.model"):
    folders = ["rest_whole", "rest_quarter", "rest_8th"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_all_rests(filename = "all_rests.model"):
    folders = ["rest_whole", "rest_quarter", "rest_8th", "rest_16th", "rest_32nd", "rest_64th"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_sfn(filename = "sfn.model"):
    folders = ["sharp", "flat", "natural"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_clefs(filename = "clef.model"):
    folders = ["gclef", "fclef"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_noteheads():
    folders = ["notehead_solid", "notehead_hollow"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(f"notehead.model", "wb"))
    

if __name__ == "__main__":
    def _run_sklearn_train(folders, model_name):
        # Train sklearn model on DS2 variant used in repo
        model, class_map = train([f"ds2_dense/train_data/{folder}" for folder in folders])
        test(model, [f"ds2_dense/test_data/{folder}" for folder in folders])
        output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
        out_dir = Path(__file__).parent
        out_dir.joinpath('sklearn_models').mkdir(exist_ok=True)
        pickle.dump(output, open(out_dir.joinpath(f"sklearn_models/{model_name}.model"), "wb"))

    parser = argparse.ArgumentParser(description="Train small symbol classifiers (sklearn or TF)")
    parser.add_argument('model', nargs='?', default='rests', choices=['rests','rests_above8','all_rests','sfn','clef','noteheads','sklearn_rests'], help='Model to train')
    parser.add_argument('--collect', type=int, help='Collect N augmented samples per class before training')
    parser.add_argument('--method', choices=['tf','sklearn'], default='tf', help='Training backend (tf or sklearn). Some targets only support one backend')
    args = parser.parse_args()

    if args.collect:
        collect_data(samples=args.collect)

    if args.model == 'sklearn_rests':
        folders = ["rest_whole", "rest_quarter", "rest_8th"]
        _run_sklearn_train(folders, 'rests')
    elif args.model == 'rests':
        # default: TF-based rests classifier
        if args.method == 'sklearn':
            folders = ["rest_whole", "rest_quarter", "rest_8th"]
            _run_sklearn_train(folders, 'rests')
        else:
            train_rests()
    elif args.model == 'rests_above8':
        if args.method == 'sklearn':
            folders = ["rest_8th", "rest_16th", "rest_32nd", "rest_64th"]
            _run_sklearn_train(folders, 'rests_above8')
        else:
            train_rests_above8()
    elif args.model == 'all_rests':
        train_all_rests()
    elif args.model == 'sfn':
        train_sfn()
    elif args.model == 'clef':
        train_clefs()
    elif args.model == 'noteheads':
        train_noteheads()