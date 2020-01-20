import numpy as np
import sys, os
import cv2
import datetime

g_net = ""
g_CLASSES = (
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
)
g_resultFile = ""


def PrintUsage():
    print(
        "Usage:\n"
        "\tSet .caffemodel file and images directory in command line,\n"
        "\te.g., python demo.py mobilenet_iter_120000.caffemodel /root/data/VOCdevkit/VOC2007/JPEGImages/\n"
    )


def main(argv):
    PrintUsage()
    if len(sys.argv) < 3:
        exit()

    caffe_root = '/soft/caffe/'
    sys.path.insert(0, caffe_root + "/python")
    import caffe

    net_file = caffe_root + "/examples/MobileNet-SSD/deploy.prototxt"

    caffe_model = sys.argv[1]
    test_dir = sys.argv[2]
    if not os.path.exists(caffe_model):
        print(caffe_model + " does not exist.")
        exit()
    if not os.path.exists(net_file):
        print(net_file + " does not exist.")
        exit()

    result_file_name = "detection_results.txt"
    if os.path.isfile(result_file_name):
        print(result_file_name + " already exists.")
        exit()

    print("Loading caffe.")
    global g_net
    g_net = caffe.Net(net_file, caffe_model, caffe.TEST)
    print("Caffe loaded.")

    global g_resultFile
    g_resultFile = open(result_file_name, "w")
    for f in os.listdir(test_dir):
        if not detect(test_dir + "/" + f):
            break


def preprocess(src):
    img = cv2.resize(src, (300, 300))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    return box.astype(np.int32), conf, cls


def detect(imgfile):
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now + ", processing " + imgfile)

    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    global g_net
    g_net.blobs['data'].data[...] = img
    out = g_net.forward()
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
        global g_CLASSES
        title = "%s,%.2f" % (g_CLASSES[int(cls[i])], conf[i])

        global g_resultFile
        g_resultFile.write(
            imgfile
            + "," + str(box[i][0])
            + "," + str(box[i][1])
            + "," + str(box[i][2])
            + "," + str(box[i][3])
            + "," + title + "\n"
        )

    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now + ", end processing " + imgfile)

    return True


if __name__ == "__main__":
    main(sys.argv[1:])
