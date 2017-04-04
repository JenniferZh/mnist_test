import numpy as np
import scipy.misc
import timeit
from scipy import stats


def ReadImageData(train_cnt, filename):
    '''根据mnist数据格式，将数据读取到list中，list的每个元素是一个图片像素信息的ndarray
    Args:
        train_cnt: 读取图片的个数，若输入小于零，将读取全部
        filename: 读取图片路径
        
    Returns:
        image_list: 图片像素信息的list
    '''
    image_list = []

    train_image_file = open(filename,"rb")
    magic = int.from_bytes(train_image_file.read(4), byteorder='big')
    image_total = int.from_bytes(train_image_file.read(4), byteorder='big')
    row = int.from_bytes(train_image_file.read(4), byteorder='big')
    col = int.from_bytes(train_image_file.read(4), byteorder='big')

    if train_cnt < 0:
        train_cnt = image_total

    for i in range(train_cnt):
        image = np.array([])
        for ith_row in range(row):
            for jth_col in range(col):
                pixel = int.from_bytes(train_image_file.read(1), byteorder='big')
                image = np.append(image, pixel)
        #image = np.reshape(image,(row, col))
        #scipy.misc.imsave('pic'+str(i)+'.jpg', image)
        image_list.append(image)
    train_image_file.close()
    return image_list


def ReadResult(train_cnt, filename):
    '''根据mnist数据格式，将label读取到list中，list的每个元素是一个图像的label，顺序和图像的list相对应
    Args:
        train_cnt: 读取图片的个数，若输入小于零，将读取全部
        filename: 读取图片路径

    Returns:
        image_list: 图片label信息的list
    '''

    resutl_list = []

    train_result_file = open(filename,"rb")
    magic = int.from_bytes(train_result_file.read(4), byteorder='big')
    result_total = int.from_bytes(train_result_file.read(4), byteorder='big')

    if train_cnt < 0:
        train_cnt = result_total

    for i in range(train_cnt):
        label = int.from_bytes(train_result_file.read(1), byteorder='big')
        resutl_list.append(label)
    return resutl_list

def distance_compute(image1, image2, type = 'euclidean'):
    if type == 'euclidean':
        pixel_cnt = len(image1)
        return np.sqrt(np.sum((image1 - image2) ** 2))
        #return np.linalg.norm(image1 - image2)
    else:
        return 0

def predict(image, image_list, image_label, k):
    k_list = []

    image_cnt = len(image_list)
    for i in range(k):
        distance = distance_compute(image, image_list[i])
        item = [distance, image_label[i]]
        k_list.append(item)

    k_list.sort(key=lambda x:x[0], reverse=True)

    for i in range(k, image_cnt):
        distance = distance_compute(image, image_list[i])
        for j in range(k):
            if(distance < k_list[j][0]):
                k_list[j] = [distance, image_label[i]]

    k_list = np.transpose(np.array(k_list))
    mode = stats.mode(k_list[1])
    return int(mode[0][0])


def evaluate(eval, label):
    cnt = 0
    for i in range(len(eval)):
        if eval[i] == label[i]:
            cnt = cnt + 1
    print(cnt)
    return cnt

def predict_all(trainsize, testsize, k, file):
    #file = open('result','wb')

    file.write('k='+str(k)+'\n')
    start = timeit.default_timer()

    image = ReadImageData(trainsize, 'train-images.idx3-ubyte')
    test = ReadImageData(testsize, 't10k-images.idx3-ubyte')
    label = ReadResult(trainsize, 'train-labels.idx1-ubyte')
    testlabel = ReadResult(testsize, 't10k-labels.idx1-ubyte')

    end = timeit.default_timer()

    print("time",end-start)
    file.write("read time"+str(end-start)+"\n")

    start = timeit.default_timer()
    evalresult = []
    c = 0
    for item in test:
        print(c)
        c = c+1
        evalresult.append(predict(item, image, label, k))
    file.write("cnt="+str(evaluate(evalresult, testlabel))+"\n")
    end = timeit.default_timer()

    file.write("evaluate time"+str(end-start)+"\n")
    print("time",end-start)
    #print(testlabel)

if __name__ == "__main__":
    file = open('result1', 'w')

    predict_all(60000,10000,1, file)
    predict_all(60000,10000,3, file)
    predict_all(60000,10000,5, file)
    predict_all(60000, 10000, 7, file)

    file.close()


