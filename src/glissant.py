import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from absl import flags, app
from sample import load_model
import tensorflow as tf

FLAGS = flags.FLAGS

# Définition des drapeaux pour les chemins d'image et les paramètres de prédiction
flags.DEFINE_string('imagePathIn', 'port98x64.jpg', 'Chemin vers le fichier image en entrée')
flags.DEFINE_string('outputPath', 'ImageResultat/slidingPrediction', 'Chemin pour sauvegarder l\'image produite')
flags.DEFINE_integer('stride', 2, 'Valeur de stride pour la fenêtre glissante')
flags.DEFINE_integer('upscaleFactor', 4, "Facteur d'agrandissement pour l'image de sortie")
flags.DEFINE_integer('patchSize', 8, 'Taille du patch pour la prédiction par fenêtre glissante')
flags.DEFINE_float('tempo', 1, "Temps d'attente entre chaque patch (en secondes)")
flags.DEFINE_integer('affichage', 1, "Manière dont la progression est affichée, 0 : afficher uniquement la prédiction terminée, 1 : afficher la prediction à chaque rangée prédite, 2 : afficher à chaque prédiction")

def extractPatch(image, yCoord, xCoord, patchSize):
    """
    Extrait un patch de l'image en fonction des coordonnées et de la taille du patch fournies.
    """
    return image[yCoord:yCoord + patchSize, xCoord:xCoord + patchSize, :]

def predictPatch(model, patch):
    """
    Prédit le patch de sortie en utilisant le modèle fourni.
    """
    patch = np.expand_dims(patch, axis=0)  
    predictedPatch = model.predict(patch)
    return predictedPatch[0] 

def updateOutput(upscaledImage, overlapCounter, predictedPatch, yCoord, xCoord, upscaleFactor):
    """
    Met à jour l'image de sortie et le compteur de chevauchement avec le patch prédit.
    """
    yStart = yCoord * upscaleFactor
    xStart = xCoord * upscaleFactor
    
    yEnd = yStart + predictedPatch.shape[0]
    xEnd = xStart + predictedPatch.shape[1]

    upscaledImage[yStart:yEnd, xStart:xEnd, :] += predictedPatch
    overlapCounter[yStart:yEnd, xStart:xEnd, :] += 1

def slidingWindowPrediction(model, image, patchSize, stride, naiveResizedImage):
    """
    Effectue une prédiction par fenêtre glissante sur l'image d'entrée en utilisant le modèle donné.
    Affiche l'image agrandie naïvement et l'image mise à jour après chaque patch prédit.
    """
    height, width, channels = image.shape
    upscaleFactor = FLAGS.upscaleFactor
    outHeight = height * upscaleFactor
    outWidth = width * upscaleFactor

    upscaledImage = np.zeros((outHeight, outWidth, channels), dtype=np.float32)
    overlapCounter = np.zeros((outHeight, outWidth, channels), dtype=np.float32)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ImageProgess(naiveResizedImage, upscaledImage / overlapCounter / 255, axs) 
    for y in range(0, height - patchSize + 1, stride):
        for x in range(0, width - patchSize + 1, stride):
            patch = extractPatch(image, y, x, patchSize)
            predictedPatch = predictPatch(model, patch)
            updateOutput(upscaledImage, overlapCounter, predictedPatch, y, x, upscaleFactor)
            if(FLAGS.affichage == 2) :
                ImageProgess(naiveResizedImage, upscaledImage / overlapCounter / 255, axs)
        if(FLAGS.affichage == 1) :
            ImageProgess(naiveResizedImage, upscaledImage / overlapCounter / 255, axs)    
        


    upscaledImage /= overlapCounter
    return upscaledImage

def ImageProgess(naiveResizedImage, upscaledImage, axs):
    """
    Affiche la progression de la prédiction de l'image en cours.
    """
    axs[0].imshow(naiveResizedImage.astype(np.uint8), cmap='gray')
    axs[0].set_title('Image Agrandie Sans Interpolation')
    axs[0].axis('off')
    
    axs[1].imshow(upscaledImage, cmap='gray')
    axs[1].set_title('Progression de la prédiction glissante')
    axs[1].axis('off')
    plt.draw() 
    plt.pause(FLAGS.tempo)

def loadImage(imagePathIn):
    """
    Charge et prépare l'image à partir du chemin spécifié.
    """
    image = Image.open(os.path.join('ImageTest/', imagePathIn))
    image = image.convert('RGB')
    imageData = np.array(image).astype(np.float32)
    return imageData 

def naiveResize(image, upscaleFactor):
    """
    Agrandit l'image d'origine 4 fois en utilisant la méthode NEAREST_NEIGHBOR.
    """
    imageTensor = tf.convert_to_tensor(image)
    resizedImage = tf.image.resize(imageTensor, size=(image.shape[0] * upscaleFactor, image.shape[1] * upscaleFactor), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return resizedImage.numpy()

def saveImage(image, outputPath):
    """
    Sauvegarde l'image spécifiée sur le disque.
    """
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    image = Image.fromarray(image)
    image.save(outputPath)
    print(f"Image sauvegardée avec succès sous : {outputPath}")

def prediction(argv):
    """
    Exécute la prédiction par fenêtre glissante sur l'image au chemin spécifié.
    """
    model = load_model()[2]  
    imgData = loadImage(FLAGS.imagePathIn)
    naiveResizedImage = naiveResize(imgData,  FLAGS.upscaleFactor)
    patchSize = FLAGS.patchSize
    stride = FLAGS.stride

    upscaledImage = slidingWindowPrediction(model, imgData, patchSize, stride, naiveResizedImage) / 255
    
    plt.subplot(1, 2, 1)
    plt.imshow(naiveResizedImage.astype(np.uint8))
    plt.title("Image Agrandie Sans Interpolation")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(upscaledImage, cmap='gray')
    plt.title('Image redimensionnée par prédiction glissante')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    outputImagePath = os.path.abspath(FLAGS.outputPath)
    saveImage(upscaledImage,  outputImagePath + FLAGS.imagePathIn) 


def main(argv):
    prediction(argv)

if __name__ == '__main__':
    app.run(main)
