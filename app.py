import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow



def preprocess(img):
        
    # Convertissez l'image en niveaux de gris
    img_pil_greyscale = img.convert("L")
    # Si vous avez besoin de l'image en tableau NumPy, vous pouvez la convertir
    img_array_greyscale = np.array(img_pil_greyscale)
    img_array_greyscale_divided = img_array_greyscale / 255.0

    # Ajoutez une dimension supplémentaire pour le batch
    img_array_greyscale_divided = np.expand_dims(img_array_greyscale_divided, axis=0)

    return img_array_greyscale_divided





def decouper_image(image, taille_cote = 64):
    # Spécifiez le chemin vers le dossier du modèle SavedModel (.pb)
    saved_model_path = './data/best_model_try_2/mat64.h5'

    # Chargez le modèle SavedModel
    model = tensorflow.keras.models.load_model(saved_model_path)

    # Obtenir les dimensions de l'image
    largeur, hauteur = image.size

    #Créer la nouvellee image
    image_globale_np = np.zeros((hauteur, largeur, 3), dtype=np.uint8)
    print('création de l"image globale', image_globale_np.shape)
    # Calculer le nombre de carrés en largeur et en hauteur
    nombre_cotes_x = largeur // taille_cote
    nombre_cotes_y = hauteur // taille_cote

    # Découper l'image en carrés

    for i in range(nombre_cotes_x):
        for j in range(nombre_cotes_y):

            # Coordonnées du coin supérieur gauche du carré
            x1 = i * taille_cote
            y1 = j * taille_cote

            # Coordonnées du coin inférieur droit du carré
            x2 = x1 + taille_cote
            y2 = y1 + taille_cote

            # Découper le carré
            carre = image.crop((x1, y1, x2, y2))
            # i, j  sont les coordonnées de l'image

            img_array = preprocess(carre)
            prediction = model.predict(img_array)
            if prediction[0][0] > 0.6: 
                label = 1
            elif prediction[0][0] < 0.4:
                label = 0
            else:
                label = 2

            image_globale_np = reconstituer_image(image_globale_np, carre, i, j, label)

    return image_globale_np

def reconstituer_image(image_globale, image_echantillon, i, j, label):
    
    largeur, hauteur = image_echantillon.size
    x_position = i * largeur
    y_position = j * hauteur

    # Convertir l'image échantillon en mode RGB
     # Convertir l'image échantillon en tableau NumPy
    image_echantillon_np = np.array(image_echantillon)

    # Assigner l'image échantillon au canal approprié de l'image globale
    if label == 0:
        image_globale[y_position:y_position + hauteur, x_position:x_position + largeur, 0] = image_echantillon_np[:,:,0]
    elif label == 1:
        image_globale[y_position:y_position + hauteur, x_position:x_position + largeur, 1] = image_echantillon_np[:,:,1]
    elif label == 2:
        image_globale[y_position:y_position + hauteur, x_position:x_position + largeur, 2] = image_echantillon_np[:,:,2]
    return image_globale

# Fonction pour charger et afficher une image téléchargée
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Définition de l'application Streamlit
def main():
    st.title("Application test - Guilhem")

    # Sélection de l'image
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])


    # Si l'utilisateur a téléchargé une image
    if uploaded_file is not None:
        # Affichage de l'image
        image = load_image(uploaded_file)
        st.image(image, caption="Image téléchargée", use_column_width=True)

        image_predit = decouper_image(image)
        st.image(image_predit, caption="Image avec label", use_column_width=True)

 
        # Action à effectuer avec l'image (ex: passer l'image à un modèle)
        # Ici, vous pouvez appeler une fonction pour effectuer une prédiction avec votre modèle
        # Par exemple, si vous avez une fonction predict(image) qui renvoie le résultat de la prédiction,
        # vous pouvez appeler predict(image) comme suit :
        # prediction = predict(image)
        # st.write(prediction)

if __name__ == "__main__":
    main()
