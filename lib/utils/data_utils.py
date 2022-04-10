import numpy as np
from config import cfg
import dataset.dataset_factory as dataset_factory

def change_semisupervised_sample_ratio(dataset, ratio, total):
    no_supervised_datapoints = int(ratio * total)
    no_unsupervised_datapoints = int((1-ratio) * total)
    dataset._image_data = np.random.RandomState(seed=1).permutation(dataset._image_data)
    watch_ids = [image_data['id'] for image_data in dataset._image_data[:cfg.TRAIN.SUP_SAMPLES]]
    supervised_samples = np.random.choice(dataset._image_data[:cfg.TRAIN.SUP_SAMPLES], no_supervised_datapoints)
    unsupervised_samples = np.random.choice(dataset._image_data[cfg.TRAIN.SUP_SAMPLES:], no_unsupervised_datapoints)
    dataset._image_data = list(supervised_samples) + list(unsupervised_samples)
    dataset._image_index = [image_data['id'] for image_data in dataset._image_data]
    return dataset, watch_ids

def combined_pascal_dataset(strong_dataset, add_params):
    del add_params['devkit_path']
    weak_dataset, _ = dataset_factory.get_dataset("voc_2012_trainval", add_params, mode='train')
    unsupervised_samples = np.random.choice(weak_dataset._image_data, cfg.TRAIN.NUM_SAMPLES_VOC)
    strong_dataset._image_data += list(unsupervised_samples)
    strong_dataset._image_index = [image_data['id'] for image_data in strong_dataset._image_data]
    return strong_dataset

def semisupervised_sampling_coco(dataset, total=11000):
    total_supervised_samples = int(cfg.TRAIN.SUP_SAMPLES_PERCENTAGE * len(dataset))
    no_supervised_datapoints = int(cfg.TRAIN.SAMPLING_RATIO * total)
    no_unsupervised_datapoints = int((1-cfg.TRAIN.SAMPLING_RATIO) * total)
    #dataset._image_data = np.random.RandomState(seed=1).permutation(dataset._image_data)
    watch_ids = [image_data['id'] for image_data in dataset._image_data[:total_supervised_samples]]
    supervised_samples = np.random.choice(dataset._image_data[:total_supervised_samples], no_supervised_datapoints)
    unsupervised_samples = np.random.choice(dataset._image_data[total_supervised_samples:], no_unsupervised_datapoints)
    dataset._image_data = list(supervised_samples) + list(unsupervised_samples)
    dataset._image_index = [image_data['id'] for image_data in dataset._image_data]
    return dataset, watch_ids