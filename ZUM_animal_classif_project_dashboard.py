from streamlit_option_menu import option_menu
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from PIL import Image
from ProcessingAndModels import DataLoadAndInitProcess
from ProcessingAndModels import PcaAndRandomForestModel
from PIL import Image
import joblib
from sklearn.preprocessing import LabelBinarizer
import torch
import torchvision.transforms as transforms
from ProcessingAndModels import Patches
from ProcessingAndModels import PatchEncoder
from tensorflow import keras


# Puszczenie: w konsoli python -m streamlit run 'pelna\sciezka\do\pliku.py'
# token: np. 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
# loign() w klasie ProcessingAndModels odkomentować dla pierwszego puszczenia


rf_model = joblib.load('D:\Studia stuff\ZUM\dashboard_wlasciwy\\best_rf_model.pkl')
results = pd.read_csv('D:\Studia stuff\ZUM\dashboard_wlasciwy\grid_search_results.csv')
best_params = results.loc[results['rank_test_score'] == 1, 'params'].values[0]

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

dataload = DataLoadAndInitProcess()

def return_imgs():
    imgs = dataload.df['image']
    labels = dataload.df['name_label']
    images = []
    selected_labels = []
    
    random_indices = random.sample(range(len(imgs)), 3)
    
    for i in random_indices:
        image_path = imgs[i]['path']
        image = Image.open(image_path)
        images.append(image)
        selected_labels.append(labels[i])
    
    return images, selected_labels

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Animal classification models dashboard</h1>", unsafe_allow_html=True)
selected = option_menu(
    menu_title=None,
    options=['Home', 'Model results', 'Model test'],
    default_index=0,
    orientation='horizontal',
    icons=['house', 'border-all', 'wrench']
)

if selected == 'Home':
    st.markdown("<h5 style='text-align: center;'>Data overview</h5>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 5, 2])
    if st.button('Reload'):
        images, labels = return_imgs()
    else:
        images, labels = return_imgs()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for i, ax in enumerate(axs):
        ax.set_title(f"Label: {labels[i]}")
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    st.pyplot(fig)

    # if st.button('Show edge detection:'):   #schowane za przyciskiem bo dość długo się ładuje
    fig_edge, axs_edge = plt.subplots(1, 3, figsize=(12, 3))
    for i, ax in enumerate(axs_edge):
        edge_image = PcaAndRandomForestModel().extract_edges(images[i])
        ax.set_title(f"Edge Detection: {labels[i]}")
        ax.imshow(edge_image, cmap='gray')
        ax.axis('off')
    st.pyplot(fig_edge)

    n_bins = 90 #liczba klas
    fig = px.histogram(list(dataload.df['label']), nbins=n_bins, title='Distribution of classes - even distribution')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


elif selected == 'Model results':
    chosen_model = st.radio(
    "Choose model",
    key="visibility",
    options=["Random Forest Model", "CNN Model", "Transformer Model"]
    )
    col1, col2 = st.columns(2)

    if chosen_model == 'Random Forest Model':
        best_params = results.loc[results['rank_test_score'] == 1, 'params'].values[0]
        st.write("Best Parameters: ", best_params)

        rf_val_acc_df = pd.read_csv('D:\Studia stuff\ZUM\dashboard_wlasciwy\\rf_val_accuracy.csv')
        rf_val_acc = rf_val_acc_df['val_accuracy'].iloc[0]
        st.write("Validation Accuracy: ", rf_val_acc)
        
        rf_test_acc_df = pd.read_csv('D:\Studia stuff\ZUM\dashboard_wlasciwy\\rf_test_accuracy.csv')
        rf_test_acc = rf_test_acc_df['test_accuracy'].iloc[0]
        st.write("Test Accuracy: ", rf_test_acc)


        rf_report_df = pd.read_csv('D:\Studia stuff\ZUM\dashboard_wlasciwy\\rf_classification_report.csv', index_col=0)
        st.write("Classification Report")
        st.dataframe(rf_report_df)
    elif chosen_model == 'CNN Model':
        cnn_results_df = pd.read_csv('D:\Studia stuff\ZUM\dashboard_wlasciwy\\cnn_results.csv')
        st.write("CNN Results:")
        # st.dataframe(cnn_results_df)

        test_loss = cnn_results_df.loc[cnn_results_df['Metric'] == 'Test Loss', 'Value'].values[0]
        test_accuracy = cnn_results_df.loc[cnn_results_df['Metric'] == 'Test Accuracy', 'Value'].values[0]

        st.write("Test Loss: ", test_loss)
        st.write("Test Accuracy: ", test_accuracy)

        cnn_history_df = pd.read_csv('D:\Studia stuff\ZUM\dashboard_wlasciwy\\training_history_cnn.csv')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(cnn_history_df['accuracy'], label='Train Accuracy')
        ax1.plot(cnn_history_df['val_accuracy'], label='Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.set_title('Training and Validation Accuracy')

        ax2.plot(cnn_history_df['loss'], label='Train Loss')
        ax2.plot(cnn_history_df['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.set_title('Training and Validation Loss')

        st.pyplot(fig)
    elif chosen_model == 'Transformer Model':
        transformer_results_df = pd.read_csv('D:\Studia stuff\ZUM\dashboard_wlasciwy\\transformer_results.csv')
        st.write("Transformer Results:")
        # st.dataframe(cnn_results_df)

        test_loss = transformer_results_df.loc[transformer_results_df['Metric'] == 'Test Loss', 'Value'].values[0]
        test_accuracy = transformer_results_df.loc[transformer_results_df['Metric'] == 'Test Accuracy', 'Value'].values[0]

        st.write("Test Loss: ", test_loss)
        st.write("Test Accuracy: ", test_accuracy)

        transformer_history_df = pd.read_csv('D:\Studia stuff\ZUM\dashboard_wlasciwy\\training_history_transformer.csv')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(transformer_history_df['accuracy'], label='Train Accuracy')
        ax1.plot(transformer_history_df['val_accuracy'], label='Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.set_title('Training and Validation Accuracy')

        ax2.plot(transformer_history_df['loss'], label='Train Loss')
        ax2.plot(transformer_history_df['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.set_title('Training and Validation Loss')

        st.pyplot(fig)

elif selected == 'Model test':
    st.title("Model Testing")
    def preprocess_image(image):
        resized_image = image.resize((300, 300))
        
        cropped_image = resized_image.crop((10, 10, 290, 290))
        cropped_image = cropped_image.resize((300, 300))
        
        transform = transforms.Compose([transforms.ToTensor()])
        tensor_image = transform(cropped_image).permute((1, 2, 0)).numpy()
        
        batch_image = np.expand_dims(tensor_image, axis=0)

        return batch_image

    custom_objects = {
    "Patches": Patches(patch_size=30),
    "PatchEncoder": PatchEncoder(num_patches=100, projection_dim=64),
    }

    model_path = "D:\Studia stuff\ZUM\ZUM_Project\model2.keras"
    transformer_model = keras.models.load_model(model_path, custom_objects=custom_objects)

    uploaded_image = st.file_uploader(
        "Upload an image for testing:", accept_multiple_files=False, type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict!"):
            image = Image.open(uploaded_image)
            processed_image = preprocess_image(image)
            
            predictions = transformer_model.predict(processed_image)
            predicted_class = np.argmax(predictions, axis=-1)[0]

            predicted_label = name_change.get(predicted_class, "Unknown")
            st.write(f"Predicted Label: {predicted_label}")