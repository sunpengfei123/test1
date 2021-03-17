import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import data
# 导入CSV安装包
import csv

new_model=tf.keras.models.load_model("model\\model_conv_att316.h5")

test_x = tf.convert_to_tensor(data.get_test_de('..\\track1_round1_testA_20210222.csv'))

#使用模型进行预测
pr = new_model.predict(test_x)

np.set_printoptions(suppress=True)

pr = np.array(pr)

# 1. 创建文件对象
f = open('result\\testA_result_conv316_1.csv','w',encoding='utf-8')


# 3. 构建列表头
for i in range(pr.__len__()):
    f.write((str)(i)+"|,|")
    for re in pr[i]:
        f.write(str(round(re,2))+" ")
    f.write("\r")

# 5. 关闭文件
f.close()
