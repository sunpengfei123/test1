import csv
import numpy


def get_description(s):
    description = []
    len = 0
    mmax = 0
    with open(s, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        # print((int)(result[0][1].split("|")[1].split(" ")[1]))
        for re in result:
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
        dde = numpy.array(list(de + [0] * (len - de.__len__())))
        description2.append(dde)

    re = numpy.array(description2, dtype=numpy.int32)

    return re


def get_label(s):
    label = []

    with open(s, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        # print((int)(result[0][1].split("|")[1].split(" ")[1]))
        for re in result:
            la = numpy.zeros(17)
            s = "" + re[2].split("|")[1]
            for l in s.split(" "):
                if not l == '':
                    index = ((int)(l) -1)
                    la[index] = 1
            label.append(la)
    return  numpy.array(label, dtype=numpy.int32)


if __name__ == '__main__':
    description = get_description('track1_round1_train_20210222.csv')
    print(description[0])
    # label = get_label()
    # print(label[15])
