import os.path as osp
import glob
import imagesize
import json


class Convertor(object):
    def __init__(self, root_dir, output_dir, source='drones', target='coco'):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.source = source
        self.target = target

        self.splits = ['train', 'val', 'test']
        if source == 'drones' and target == 'coco':
            self.start = self.drones2coco

    def load_drones(self):
        splits_names = {}
        for split in self.splits:
            img_path = osp.join(self.root_dir, split, 'images')
            image_names = glob.glob(osp.join(img_path, '*.jpg'))
            names = [x.split('/')[-1].split('.')[0] for x in image_names]
            splits_names[split] = names
        return splits_names

    def drones2coco(self):
        splits_names = self.load_drones()
        for split in self.splits:
            coco_json = {
                "info": "",
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 0,
                        "name": "ignore",
                        "supercategory": "",
                    },
                    {
                        "id": 1,
                        "name": "pedestrian",
                        "supercategory": "",
                    },
                    {
                        "id": 2,
                        "name": "people",
                        "supercategory": "",
                    },
                    {
                        "id": 3,
                        "name": "bicycle",
                        "supercategory": "",
                    },
                    {
                        "id": 4,
                        "name": "car",
                        "supercategory": "",
                    },
                    {
                        "id": 5,
                        "name": "van",
                        "supercategory": "",
                    },
                    {
                        "id": 6,
                        "name": "truck",
                        "supercategory": "",
                    },
                    {
                        "id": 7,
                        "name": "tricycle",
                        "supercategory": "",
                    },
                    {
                        "id": 8,
                        "name": "awning-tricycle",
                        "supercategory": "",
                    },
                    {
                        "id": 9,
                        "name": "bus",
                        "supercategory": "",
                    },
                    {
                        "id": 10,
                        "name": "motor",
                        "supercategory": "",
                    },
                    {
                        "id": 11,
                        "name": "others",
                        "supercategory": "",
                    }
                ]
            }

            names = splits_names[split]
            img_id = 0
            anno_id = 0
            for name in names:
                width, height = imagesize.get(osp.join(self.root_dir, split, 'images', '{}.jpg'.format(name)))
                image = {
                    "license": 3,
                    "file_name": "{}.jpg".format(name),
                    "coco_url": "",
                    "height": height,
                    "width": width,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": img_id
                }
                coco_json["images"].append(image)
                if split is not 'test':
                    with open(osp.join(self.root_dir, split, 'annotations', '{}.txt'.format(name)), 'r') as reader:
                        annos = reader.readlines()
                    for anno in annos:
                        anno = anno.strip().split(',')
                        x, y, w, h, score, cls, trc, occ = \
                            int(anno[0]), int(anno[1]), int(anno[2]), int(anno[3]), anno[4], anno[5], anno[6], anno[7]
                        annotation = {
                            "id": anno_id,
                            "image_id": img_id,
                            "category_id": cls,
                            "segmentation": [],
                            "area": w*h,
                            "bbox": [x, y, w, h],
                            "iscrowd": 0,
                        }
                        coco_json["annotations"].append(annotation)
                        anno_id += 1
                else:
                    annotation = {
                        "id": anno_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "segmentation": [],
                        "area": 0,
                        "bbox": [0, 0, 0, 0],
                        "iscrowd": 0,
                    }
                    coco_json["annotations"].append(annotation)
                    anno_id += 1
                img_id += 1

            with open(osp.join(self.output_dir, '{}.json'.format(split)), 'w') as outfile:
                json.dump(coco_json, outfile)


if __name__ == '__main__':
    convertor = Convertor('/root/DronesDET/', '/root/DronesDET')
    convertor.start()
