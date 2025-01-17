# from keras.datasets import mnist
from tensorflow.keras import layers
# from tensorflow.keras import Sequential
# from tensorflow import keras  
from huggingface_hub import login
from datasets import load_dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler
import random
from PIL import ImageEnhance
from sklearn.decomposition import PCA
from skimage.color import rgb2gray
from skimage.feature import canny
import numpy as np
import tensorflow as tf


name_change = {
0: 'antelope'         ,
1: 'badger'           ,
2: 'bat'              ,
3: 'bear'             ,
4: 'bee'              ,
5: 'beetle'           ,
6: 'bison'            ,
7: 'boar'             ,
8: 'butterfly'        ,
9: 'cat'              ,
10: 'caterpillar'     ,
11: 'chimpanzee'      ,
12: 'cockroach'       ,
13: 'cow'             ,
14: 'coyote'          ,
15: 'crab'            ,
16: 'crow'            ,
17: 'deer'            ,
18: 'dog'             ,
19: 'dolphin'         ,
20: 'donkey'          ,
21: 'dragonfly'       ,
22: 'duck'            ,
23: 'eagle'           ,
24: 'elephant'        ,
25: 'flamingo'        ,
26: 'fly'             ,
27: 'fox'             ,
28: 'goat'            ,
29: 'goldfish'        ,
30: 'goose'           ,
31: 'gorilla'         ,
32: 'grasshopper'     ,
33: 'hamster'         ,
34: 'hare'            ,
35: 'hedgehog'        ,
36: 'hippopotamus'    ,
37: 'hornbill'        ,
38: 'horse'           ,
39: 'hummingbird'     ,
40: 'hyena'           ,
41: 'jellyfish'       ,
42: 'kangaroo'        ,
43: 'koala'           ,
44: 'ladybugs'        ,
45: 'leopard'         ,
46: 'lion'            ,
47: 'lizard'          ,
48: 'lobster'         ,
49: 'mosquito'	      ,
50: 'moth'	          ,
51: 'mouse'           ,
52: 'octopus'	      ,
53: 'okapi'           ,
54: 'orangutan'       ,
55: 'otter'           ,
56: 'owl'             ,
57: 'ox'              ,
58: 'oyster'          ,
59: 'panda'	          ,
60: 'parrot'	      ,
61: 'pelecaniformes',	
62: 'penguin'	      ,
63: 'pig'	          ,
64: 'pigeon'          ,
65: 'porcupine'	      ,
66: 'possum'          ,
67: 'raccoon'	      ,
68: 'rat'	          ,
69: 'reindeer'	      ,
70: 'rhinoceros'      ,
71: 'sandpiper'       ,
72: 'seahorse'        ,
73: 'seal'            ,
74: 'shark'           ,
75: 'sheep'	          ,
76: 'snake'	          ,
77: 'sparrow'	      ,
78: 'squid'	          ,
79: 'squirrel'	      ,
80: 'starfish'        ,
81: 'swan'            ,
82: 'tiger'	          ,
83: 'turkey'          ,
84: 'turtle'          ,
85: 'whale'	          ,
86: 'wolf'            ,
87: 'wombat'	      ,
88: 'woodpecker'	  ,
89: 'zebra'           ,
}

class DataLoadAndInitProcess:
    # login() #- tylko raz przy 1 puszczaniu, przy ponownych próbach nie trzeba
    ds = load_dataset("mertcobanov/animals")
    df = ds['train'].to_pandas()
    # df = df[:100]
    unique_labels = set(df['label'])
    df['name_label'] = df['label'].map(name_change)


class DataProcessing(DataLoadAndInitProcess):
    def __init__(self):
        super().__init__()
        # self.process_data()

    def crop_image(self, image, crop_size=(300, 300)):
        width, height = image.size
        crop_width, crop_height = crop_size
        left = random.randint(0, width - crop_width)
        upper = random.randint(0, height - crop_height)
        cropped_image = image.crop((left, upper, left + crop_width, upper + crop_height))
        return cropped_image

    def adjust_brightness(self, image, brightness_range=(0.8, 1.2)):
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = random.uniform(*brightness_range)
        enhanced_image = enhancer.enhance(brightness_factor)
        return enhanced_image
    
    def process_data(self):
        self.df['image_paths'] = self.df['image'].apply(lambda row: row['path'])
        self.df['raw_images'] = self.df['image_paths'].apply(lambda image_path: Image.open(image_path))
        self.df['resized_images'] = self.df['raw_images'].apply(lambda raw_image: raw_image.resize((300, 300)))  # później zmienić na 300x300
        # self.df.drop(['image', 'image_paths', 'raw_images'], axis=1, inplace=True)
        self.df['cropped_images'] = self.df['resized_images'].apply(lambda resized_image: self.crop_image(resized_image, crop_size=(280, 280))) #później na 280x280
        self.df['cropped_images'] = self.df['cropped_images'].apply(lambda cropped_image: cropped_image.resize((300, 300)))
        self.df['brightness_adjusted_images'] = self.df['resized_images'].apply(lambda resized_image: self.adjust_brightness(resized_image))

class PcaAndRandomForestModel(DataProcessing):
    def __init__(self):
        super().__init__()
        # self.process_data()

    def extract_edges(self, image):
        grayscale = rgb2gray(image)  
        edges = canny(grayscale)    
        return edges

    def apply_pca(self, features, n_components=None):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        max_components = min(features.shape[0], features.shape[1])
        if n_components is None or n_components > max_components:
            n_components = max_components
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(scaled_features)
        return reduced_features
    
    def extract_color_histogram(self, image):
        image_array = np.array(image)
        hist_red = np.histogram(image_array[:, :, 0], bins=32, range=(0, 255))[0]
        hist_green = np.histogram(image_array[:, :, 1], bins=32, range=(0, 255))[0]
        hist_blue = np.histogram(image_array[:, :, 2], bins=32, range=(0, 255))[0]
        return np.concatenate([hist_red, hist_green, hist_blue])

    def process_data(self):
        super().process_data()
        self.df['edge_features'] = self.df['resized_images'].apply(self.extract_edges)
        self.df['color_histograms'] = self.df['resized_images'].apply(self.extract_color_histogram)
        combined_features = np.array(self.df['edge_features'].apply(lambda x: x.flatten()).tolist())
        color_features = np.array(self.df['color_histograms'].tolist())
        all_features = np.hstack([combined_features, color_features])
        reduced_features = self.apply_pca(all_features)
        self.df['pca_features'] = list(reduced_features)
        # self.df.drop(['image', 'image_paths', 'raw_images'], axis=1, inplace=True) #przeniesione z klasy wyżej

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)  # dla 'name', 'trainable' i innych arg
        self.patch_size = patch_size

    def call(self, images):
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.image.extract_patches(
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(
            patches,
            [
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ],
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim if projection_dim else 64  # Domyślna wartość
        self.projection = layers.Dense(units=self.projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=self.projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config
