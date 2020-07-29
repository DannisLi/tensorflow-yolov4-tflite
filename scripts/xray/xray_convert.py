# coding: utf8

import os, sys
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split



base_dir = "/mnt/lzm/data/Xray/train"
ratio = 0.2

class_names = ["knife", "scissors", "lighter", "zippooil", "pressure", "slingshot", "handcuffs", "nailpolish", "powerbank", "firecrackers"]

train_output = "/mnt/lzm/data/Xray/train.txt"
val_output = "/mnt/lzm/data/Xray/val.txt"



def convert_annotation(_list, output_path):

    with open(output_path, 'w') as wf:
        for fpath in _list:

            # Get annotation.
            tree = ET.parse(fpath)
            root = tree.getroot()

            box_annotations = []
            for obj_element in root.findall("object"):
                name = obj_element.find("name").text
                class_idx = class_names.index(name)
                
                box_element = obj_element.find("bndbox")
                xmin = int(float(box_element.find('xmin').text))
                ymin = int(float(box_element.find('ymin').text))
                xmax = int(float(box_element.find('xmax').text))
                ymax = int(float(box_element.find('ymax').text))

                box_annotations.append(','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_idx)]))

            
            img_path = os.path.join(base_dir, root.find("filename").text)
            annotation = img_path + ' ' + ' '.join(box_annotations) + '\n'

            wf.write(annotation)


def convert_voc(train_list, val_list, train_output, val_output):

    # Training set.
    convert_annotation(train_list, train_output)

    # Validation set.
    convert_annotation(val_list, val_output)


def main():
    train_list = []
    val_list = []

    for domain_no in range(1, 7):
        domain_list = []
        domain_base_dir = os.path.join(base_dir, "domain%d/XML" % domain_no)
        for fname in os.listdir(domain_base_dir):
            fpath = os.path.join(domain_base_dir, fname)
            domain_list.append(fpath)
        domain_train_list, domain_val_list = train_test_split(domain_list, test_size=ratio)
        train_list += domain_train_list
        val_list += domain_val_list

    convert_voc(train_list, val_list, train_output, val_output)
    


if __name__ == "__main__":
    main()
