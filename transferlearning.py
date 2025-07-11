#echo = "Downloading 101_Object_Categories for image notebooks"
#curl -L -o 101_ObjectCategories.tar.gz --progress-bar http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz

import os, random, numpy, keras, matplotlib.pyplot as plt

#if using Theano with GPU
#os.environ["KERAS_BACKEND"] = "tensorflow"

from matplotlib.pyplot import imshow
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model 


class TransferLearning(object):
    def __init__(self):
        pass
    
    def transferlearning(self):
        root = 'ObjectCategories'
        exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplaines', 'Faces_easy', 'Faces']
        train_split, val_split = 0.7, 0.15
        
        categories = [x[0] for x in os.walk(root) if x[0]][1:]
        categories = [c for c in categories if c not in [os.path.join(root, e) for e in exclude]]        
        
        print(categories)

                    
        def get_image(path):
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = numpy.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return img, x
        
        #Load all the images from root folder
        data = []
        for c, category in enumerate(categories):
            images = [os.path.join(dp, f) for dp, dn, filenames
                      in os.walk(category) for f in filenames
                      if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']
            ]
            for img_path in images:
                img, x = get_image(img_path)
                data.append({'x':numpy.array(x[0]), 'y':c})
        #count the number of classes
        num_classes = len(categories)
        
        #Randomize the data order.
        random.shuffle(data)
        
        #create training / validation / test split (70%, 15%, 15%)
        idx_val = int(train_split * len(data))
        idx_test = int((train_split + val_split) * len(data))
        train = data[:idx_val]
        val = data[idx_val:idx_test]
        test = data[idx_test:]
        
        #Separate data for labels
        x_train, y_train = numpy.array([t["x"] for t in train]), [t["y"] for t in train]
        x_val, y_val = numpy.array([t["x"] for t in val]), [t["y"] for t in val]
        x_test, y_test = numpy.array([t["x"] for t in test]), [t["y"] for t in test]
        print(y_test)