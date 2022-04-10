from selective_search import selective_search
import os
import json
import pickle
import numpy as np
from PIL import Image
import sys
import logging
log = logging.getLogger('All_Logs')
log.setLevel(logging.INFO)
fh = logging.FileHandler("ss_processed_last_81000", mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
log.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)
#from dataset.coco.pycocotools.coco import COCO as MSCOCO

def main():
    data_path = os.path.join('/raid/notebook/akhil/data_FS_WS/', 'COCO')
    image_set = 'train2017'
    ann_file = os.path.join(data_path, 'annotations',  'instances_' + image_set +  '.json')
    dataset = json.load(open(ann_file, 'r'))
    ss_boxes = {}
    filename = os.path.join(data_path, 'selective_search_data', 'coco_last_81000'+image_set+'.pkl')
    no_images = len(dataset['images'])
    log.info("No images: {}".format(no_images))
    no_boxes = []
    bad_image_ids = []
    for i in range(len(dataset['images'])-81000, len(dataset['images'])-78000):
        image_path = os.path.join(data_path, 'images', image_set, dataset['images'][i]['file_name'])
        image = np.asarray(Image.open(image_path))
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)
        try:
            boxes = selective_search(image, mode='fast', random_sort=False)
        except:
            bad_image_ids.append(dataset['images'][i]['id'])
            continue
        ss_boxes[dataset['images'][i]['id']] = boxes
        no_boxes.append(len(boxes))
        log.info("processing image {}/{}".format(i+1, no_images-78000))
    with open(filename, 'wb') as fid:
        pickle.dump(ss_boxes, fid, pickle.HIGHEST_PROTOCOL)
    log.info('Wrote proposals to {}'.format(filename))
    log.info("Average no of proposals per image: {}".format(np.array(no_boxes).mean()))
    log.info("No. of bad images: {}".format(len(bad_image_ids)))

    with open("bad_images_last_81000", "w") as f:
        for item in bad_image_ids:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()