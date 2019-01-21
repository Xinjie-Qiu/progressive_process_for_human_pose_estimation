import json

image_dir = '/data/COCO2014/'
with open('/data/COCO2014/annotations/person_keypoints_train2014.json') as f:
    data = json.load(f)

print(data)
