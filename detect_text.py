#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

import os
curr_dir = os.path.abspath(os.path.dirname(__file__))


#caffe_root = '/home/eugen/projects/deep_learning/caffe'  # this file is expected to be in {caffe_root}/examples
caffe_root="/home/eugen/projects/deep_learning/ssd_caffe"
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
#if gpu is used:
caffe.set_device(0)
caffe.set_mode_gpu()

#load label data
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load plate  labels
labelmap_file = curr_dir+'/training_file/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load plate  labels

#init caffee
model_def = curr_dir+'/training_file/deploy.prototxt'
model_weights = curr_dir+'/training_file/VGG_scenetext_SSD_300x300_iter_60000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)
if len(sys.argv)==1:
    img_path=os.path.expanduser("~")+"/projects/cpp/videoocr/testdata/report1/v1_2.jpg"
else:
    img_path=sys.argv[1]
print "Loading file: "+str(img_path)
#img_path=curr_dir+'/test_file/120.jpg'
image = caffe.io.load_image(img_path)
plt.imshow(image)

#process the image
transformed_image = transformer.preprocess('data', image)
#send to caffe
net.blobs['data'].data[...] = transformed_image
import time
# Forward pass.
start = time.clock()
detections = net.forward()['detection_out']
end = time.clock()
print "read: %f s" % (end - start)
# Parse the outputs.
det_label = detections[0,0,:,1]
det_conf = detections[0,0,:,2]
det_xmin = detections[0,0,:,3]
det_ymin = detections[0,0,:,4]
det_xmax = detections[0,0,:,5]
det_ymax = detections[0,0,:,6]

# Get detections with confidence higher than 0.6.
top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]

top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist()
top_labels = get_labelname(labelmap, top_label_indices)
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]


#display the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

plt.imshow(image)
currentAxis = plt.gca()

for i in xrange(top_conf.shape[0]):
    xmin = int(round(top_xmin[i] * image.shape[1]))
    ymin = int(round(top_ymin[i] * image.shape[0]))
    xmax = int(round(top_xmax[i] * image.shape[1]))
    ymax = int(round(top_ymax[i] * image.shape[0]))
    score = top_conf[i]
    label = int(top_label_indices[i])
    label_name = top_labels[i]
    display_txt = '%s: %.2f'%(label_name, score)
    print score
    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    color = colors[label]
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    #currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
dest_dir="/".join(img_path.split("/")[0:-1])
inp_file_name=img_path.split("/")[-1]

dest_path=dest_dir+"/SSD_"+inp_file_name.split(".")[0]+".jpg"
plt.savefig(dest_path, bbox_inches='tight')
print "Output file saved in "+str(dest_path)
plt.show()