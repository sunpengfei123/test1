import csv
import numpy
import tensorflow as tf

def get_description(s):
    description = []
    len = 0
    mmax = 0
    mmin = 5
    with open(s, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        # print((int)(result[0][1].split("|")[1].split(" ")[1]))
        for i in range(result.__len__()-1000):
            re = result[i]
            de = []
            s = "" + re[1].split("|")[1]
            for d in s.split(" "):
                if not d == '':
                    de.append(int(d))
                    mmax = max(mmax,int(d))
                    mmin = min(mmin,int(d))
            len = max(len, de.__len__())
            description.append(de)
    print(mmax)
    print(mmin)

    re = numpy.array((description.__len__(), len))
    description2 = []
    for de in description:
        dde = numpy.array(list(de + [858] * (104 - de.__len__())))
        # if de.__len__()>50:
        #     dde = tf.convert_to_tensor(numpy.array(list(de[:50])))
        # else:
        #     dde = tf.convert_to_tensor(numpy.array(list(de + [858] * (50 - de.__len__()))))
        # dde = tf.broadcast_to(tf.convert_to_tensor(numpy.array(list(de + [858] * (104 - de.__len__())))),[104,104])
        description2.append(dde)

        # ji = numpy.tile(numpy.array(list(de + [858] * (104 - de.__len__()))), 2)
        # for x in range(10):
        #     description2.append(ji[x:104 + x])



    re = numpy.array(description2, dtype=numpy.int32)

    return re

def get_test_de(s):
    description = []
    len = 0
    mmax = 0
    with open(s, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        # print((int)(result[0][1].split("|")[1].split(" ")[1]))
        for i in range(result.__len__()):
            re = result[i]
            de = []
            s = "" + re[1].split("|")[1]
            for d in s.split(" "):
                if not d == '':
                    de.append(int(d))
                    mmax = max(mmax,int(d))
            len = max(len, de.__len__())
            description.append(de)
    print(mmax)

    re = numpy.array((description.__len__(), len))
    description2 = []
    for de in description:
        dde = numpy.array(list(de + [858] * (104 - de.__len__())))
        # if de.__len__()>50:
        #     dde = tf.convert_to_tensor(numpy.array(list(de[:50])))
        # else:
        #     dde = tf.convert_to_tensor(numpy.array(list(de + [858] * (50 - de.__len__()))))
        # dde = tf.broadcast_to(tf.convert_to_tensor(numpy.array(list(de + [858] * (104 - de.__len__())))),[104,104])
        description2.append(dde)

        # ji = numpy.tile(numpy.array(list(de + [858] * (104 - de.__len__()))), 2)
        # for x in range(10):
        #     description2.append(ji[x:104 + x])

    re = numpy.array(description2, dtype=numpy.int32)

    return re


def get_label(s):
    label = []

    with open(s, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        # print((int)(result[0][1].split("|")[1].split(" ")[1]))
        for i in range(result.__len__() - 1000):
            re = result[i]
            la = numpy.zeros(17)
            s = "" + re[2].split("|")[1]
            for l in s.split(" "):
                if not l == '':
                    index = ((int)(l) -1)
                    la[index] = 1
            label.append(la)
            # for x in range(10):
            #     label.append(la)

            # de = []
            # s = "" + re[1].split("|")[1]
            # for d in s.split(" "):
            #     if not d == '':
            #         de.append(int(d))
            # lab = []
            # lab.append(la)
            # lab.append(list(de + [0] * (104 - de.__len__())))
            # print(lab)
            # label.append(lab)

    return  numpy.array(label, dtype=numpy.int32)

def get_test(s):
    description = []
    len = 0
    mmax = 0
    label = []

    with open(s, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        for i in range(1000):
            re = result[i+9000-1]
            la = numpy.zeros(17)
            s = "" + re[2].split("|")[1]
            for l in s.split(" "):
                if not l == '':
                    index = ((int)(l) - 1)
                    la[index] = 1
            label.append(la)

            de = []
            s = "" + re[1].split("|")[1]
            for d in s.split(" "):
                if not d == '':
                    de.append(int(d))
                    mmax = max(mmax, int(d))
            len = max(len, de.__len__())
            description.append(de)

        re = numpy.array((description.__len__(), len))
        description2 = []
        for de in description:
            dde = numpy.array(list(de + [858] * (104 - de.__len__())))
            # if de.__len__()>50:
            #     dde = tf.convert_to_tensor(numpy.array(list(de[:50])))
            # else:
            #     dde = tf.convert_to_tensor(numpy.array(list(de + [858] * (50 - de.__len__()))))
            # dde = tf.broadcast_to(tf.convert_to_tensor(numpy.array(list(de + [858] * (104 - de.__len__())))),[104,104])
            description2.append(dde)

            # ji = numpy.tile(numpy.array(list(de + [858] * (104 - de.__len__()))), 2)
            # for x in range(10):
            #     description2.append(ji[x:10 + x])

        re = numpy.array(description2, dtype=numpy.int32)

        return tf.convert_to_tensor(re),tf.convert_to_tensor(label)

if __name__ == '__main__':
    description = get_description('track1_round1_train_20210222.csv')
    print(description[0])
    # label = get_label('track1_round1_train_20210222.csv')
    # print(label[0])
    # description = get_test_de('track1_round1_testA_20210222.csv')
    # print(description[0])
