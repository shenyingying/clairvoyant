#
# Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import argparse
import heapq
import numpy as np
import os
import math


import cv2


def main():
    parser = argparse.ArgumentParser(description='Display inception v3 classification results.')
    parser.add_argument('-i', '--input_list',
                        help='File containing input list used to generate output_dir.', required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Output directory containing Result_X/prob.raw files matching input_list.', required=True)
    parser.add_argument('-l', '--labels_file',
                        help='Path to ilsvrc_2012_labels.txt', required=True)
    parser.add_argument('-v', '--verbose_results',
                        help='Display top 5 classifications', action='store_true')
    args = parser.parse_args()

    input_list = os.path.abspath(args.input_list)
    output_dir = os.path.abspath(args.output_dir)
    labels_file = os.path.abspath(args.labels_file)
    display_top5 = args.verbose_results

    if not os.path.isfile(input_list):
        raise RuntimeError('input_list %s does not exist' % input_list)
    if not os.path.isdir(output_dir):
        raise RuntimeError('output_dir %s does not exist' % output_dir)
    if not os.path.isfile(labels_file):
        raise RuntimeError('labels_file %s does not exist' % labels_file)
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if len(labels) != 1:
        raise RuntimeError('Invalid labels_file: need 1 categories')
    with open(input_list, 'r') as f:
        input_files = [line.strip() for line in f.readlines()]

    if len(input_files) <= 0:
        print('No files listed in input_files')
    else:
        print('Classification results')
        max_filename_len = max([len(file) for file in input_files])

        for idx, val in enumerate(input_files):
            cur_results_dir = 'Result_' + str(idx)
            cur_results_file = os.path.join(output_dir, cur_results_dir, 'detection_classes:0.raw')
            if not os.path.isfile(cur_results_file):
                raise RuntimeError('missing results file: ' + cur_results_file)

            classes = os.path.join(output_dir, cur_results_dir, 'detection_classes:0.raw')
            scores_ = os.path.join(output_dir, cur_results_dir,
                                   'Postprocessor/BatchMultiClassNonMaxSuppression_scores.raw')
            classes_ = os.path.join(output_dir, cur_results_dir,
                                    'Postprocessor/BatchMultiClassNonMaxSuppression_classes.raw')
            boxes_ = os.path.join(output_dir, cur_results_dir,
                                  'Postprocessor/BatchMultiClassNonMaxSuppression_boxes.raw')

            classes_array = np.fromfile(classes, dtype=np.float32)
            # if len(classes_array) != 100:
            #     raise RuntimeError(str(len(classes_array)) + ' outputs in ' + cur_results_file)
            boxes_array = np.fromfile(boxes_, dtype=np.float32)
            scores_array = np.fromfile(scores_, dtype=np.float32)

            boxs=[]
            for i in range(len(classes_array)):
                if (scores_array[i] > 0.5):
                    classes_index = np.where(classes_array[i])[0][0]
                    classes_label = labels[classes_index]

                    box_float=[(boxes_array[4 * i + 1]*300), (boxes_array[4 * i]*300),(boxes_array[4 * i + 3]*300),(boxes_array[4 * i + 2]*300)]
                    print(box_float)

                    box = [int(boxes_array[4 * i + 1] * 300.0-2), int(boxes_array[4 * i] * 300.0-2),
                           math.ceil(boxes_array[4 * i + 3] * 300.0+4),
                           math.ceil(boxes_array[4 * i + 2] * 300.0+2)]
                    # print(box)
                    boxs.append(box)

                    display_text = '%s %s %f %s' % (val.ljust(max_filename_len), classes_label, scores_array[i], box)
                    print(display_text)
                    path = val.split('.')[0] + '.' + val.split('.')[1] + '.' + val.split('.')[2] + '.' + val.split('.')[
                        3] + '.jpg'

            # print(boxs.__len__())
            image = cv2.imread(path)
            for i in range (boxs.__len__()):
                cv2.rectangle(image, (boxs[i][0],boxs[i][1]),(boxs[i][2],boxs[i][3]), (0, 255, 0), 1)
                p=path+'.jpg'
            cv2.imwrite(p,image)

            # if not display_top5:
            #     max_prob = max(classes_array)
            #     max_prob_index = np.where(classes_array == max_prob)[0][0]
            #     max_prob_category = labels[max_prob_index]
            #     box = [boxes_array[4 * idx + 1]*300.0, boxes_array[4 * idx]*300.0, boxes_array[4 * idx + 3]*300.0,
            #            boxes_array[4 * idx + 2]*300.0]
            #
            #     display_text = '%s %f %s %s %s' % (
            #         val.ljust(max_filename_len), max_prob, str(max_prob_index).rjust(3), max_prob_category,box)
            #     print(display_text)
            # else:
            #     top5_prob = heapq.nlargest(5, xrange(len(classes_array)), classes_array.take)
            #     for i, idx in enumerate(top5_prob):
            #         prob = classes_array[idx]
            #         prob_category = labels[idx]
            #         box=[boxes_array[4*idx+1],boxes_array[4*idx],boxes_array[4*idx+3],boxes_array[4*idx+2]]
            #         print(box)
            #         display_text = '%s %f %s %s' % (
            #             val.ljust(max_filename_len), prob, str(idx).rjust(3), prob_category,box)
            #         print(display_text)


if __name__ == '__main__':
    main()
