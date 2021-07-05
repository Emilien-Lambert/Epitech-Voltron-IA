import cv2
import numpy as np
import random

from tensorflow.python.ops.image_ops_impl import flip_left_right
from mlpClassifier import classifier
from termcolor import colored
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from augmentations import *

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if (100 * (iteration / float(total)) > 99.9):
        print(colored(f'\r{prefix} |{bar}| {percent}% {suffix}', 'green'), end = printEnd)
    else:
        print(colored(f'\r{prefix} |{bar}| {percent}% {suffix}', 'yellow'), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def pca_3d(x, y):
    fig = plt.figure()
    pca = PCA(n_components=3)
    pca_res = pca.fit_transform(x)
    ax = plt.axes(projection='3d')
    ax.scatter3D(pca_res[:, 0], pca_res[:, 1], pca_res[:, 2], c=y)
    plt.show()



def load_dataset():
    dataset, train_images, train_labels, test_images, test_labels = [], [], [], [], []
    printProgressBar(0, 1770 * 8, prefix = 'Loading Dataset:', suffix = 'Complete', length = 50)


    with open("./annotations.csv") as f:
        for line in f:
            path, label = line.split(',')
            myDict = {"path": path, "label": label}
            dataset.append(myDict)    
            
    test_set = random.sample(dataset, int(len(dataset)/4))

    for pair in dataset:
        if pair not in test_set:
            image = cv2.imread(pair['path'])
            blurred = blur(image)
            horizontal_flip = np.array(flip_left_right(image))
            vertical_flip = np.array(flip_left_right(image))
            contrasted = np.array(contrast(image))
            saturated = np.array(saturation(image))
            hued = np.array(hue(image))
            gamma_img = np.array(gamma(image))


            image = cv2.resize(image, (72, 128), interpolation=cv2.INTER_AREA).flatten()
            blurred = cv2.resize(blurred, (72, 128), interpolation=cv2.INTER_AREA).flatten()
            horizontal_flip = cv2.resize(horizontal_flip, (72, 128), interpolation=cv2.INTER_AREA).flatten()
            vertical_flip = cv2.resize(vertical_flip, (72, 128), interpolation=cv2.INTER_AREA).flatten()
            contrasted = cv2.resize(contrasted, (72, 128), interpolation=cv2.INTER_AREA).flatten()
            saturated = cv2.resize(saturated, (72, 128), interpolation=cv2.INTER_AREA).flatten()
            hued = cv2.resize(hued, (72, 128), interpolation=cv2.INTER_AREA).flatten()
            gamma_img = cv2.resize(gamma_img, (72, 128), interpolation=cv2.INTER_AREA).flatten()

            train_images.append(image)
            train_labels.append(int(pair['label'].rstrip()))

            train_images.append(blurred)
            train_labels.append(int(pair['label'].rstrip()))

            train_images.append(horizontal_flip)
            train_labels.append(int(pair['label'].rstrip()))

            train_images.append(vertical_flip)
            train_labels.append(int(pair['label'].rstrip()))

            train_images.append(contrasted)
            train_labels.append(int(pair['label'].rstrip()))

            train_images.append(saturated)
            train_labels.append(int(pair['label'].rstrip()))

            train_images.append(hued)
            train_labels.append(int(pair['label'].rstrip()))

            train_images.append(gamma_img)
            train_labels.append(int(pair['label'].rstrip()))
        else:
            image = cv2.imread(pair['path'])
            blurred = blur(image)
            horizontal_flip = np.array(flip_left_right(image))
            vertical_flip = np.array(flip_left_right(image))
            contrasted = np.array(contrast(image))
            saturated = np.array(saturation(image))
            hued = np.array(hue(image))
            gamma_img = np.array(gamma(image))


            image = cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA).flatten()
            blurred = cv2.resize(blurred, (128, 72), interpolation=cv2.INTER_AREA).flatten()
            horizontal_flip = cv2.resize(horizontal_flip, (128, 72), interpolation=cv2.INTER_AREA).flatten()
            vertical_flip = cv2.resize(vertical_flip, (128, 72), interpolation=cv2.INTER_AREA).flatten()
            contrasted = cv2.resize(contrasted, (128, 72), interpolation=cv2.INTER_AREA).flatten()
            saturated = cv2.resize(saturated, (128, 72), interpolation=cv2.INTER_AREA).flatten()
            hued = cv2.resize(hued, (128, 72), interpolation=cv2.INTER_AREA).flatten()
            gamma_img = cv2.resize(gamma_img, (128, 72), interpolation=cv2.INTER_AREA).flatten()

            test_images.append(image)
            test_labels.append(int(pair['label'].rstrip()))

            test_images.append(blurred)
            test_labels.append(int(pair['label'].rstrip()))

            test_images.append(horizontal_flip)
            test_labels.append(int(pair['label'].rstrip()))

            test_images.append(vertical_flip)
            test_labels.append(int(pair['label'].rstrip()))

            test_images.append(contrasted)
            test_labels.append(int(pair['label'].rstrip()))

            test_images.append(saturated)
            test_labels.append(int(pair['label'].rstrip()))

            test_images.append(hued)
            test_labels.append(int(pair['label'].rstrip()))

            test_images.append(gamma_img)
            test_labels.append(int(pair['label'].rstrip()))

        printProgressBar(len(train_images)+len(test_images), 1770 * 8, prefix = 'Loading Dataset:', suffix = 'Complete', length = 50)


    print(train_images, '\n')
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)




def main():
    print(colored("Starting main.py \n", "green"))
    train_images, train_labels, test_images, test_labels = load_dataset()
    healthy_train = np.sum(train_labels == 1)
    healthy_test = np.sum(test_labels == 1)

    print(colored("\n\nTraining set :", "cyan"))
    print(f'healthy : ', colored(f'{healthy_train}', 'green'))
    print(f'esca : ', colored(f'{len(train_images) - healthy_train}\n\n', 'green'))


    print(colored("Testing set :", "magenta"))
    print(f'healthy : ', colored(f'{healthy_test}', 'green'))
    print(f'esca : ', colored(f'{len(test_images) - healthy_test}\n\n', 'green'))

    # pca_3d(train_images, train_labels)
    classifier(train_images, train_labels, test_images, test_labels)



if __name__ == '__main__':
    main()

