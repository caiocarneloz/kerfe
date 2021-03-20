import kerfe as krf
import pandas as pd
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16 as vgg
from keras.applications.vgg16 import preprocess_input as vgg_p

def main():
    
    path = 'images/'
    images = ['caglaroskay.jpg', 'davidbraud.jpg', 'jessicaknowlden.jpg']
    
    image_list = []
    for img in images:
        image_list.append(image.load_img(path+img, target_size=(300, 500)))
        
    df_features = krf.extract(vgg, vgg_p, image_list, (300, 500, 3))
    
    print(df_features)
    
    
if __name__ == "__main__":
    main()