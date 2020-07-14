from PIL import Image
import math
import numpy as np
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

#Zoom por replicação
def nearestZoom(srcImg, x):
    srcWidth = srcImg.width
    srcHeight = srcImg.height

    x = int(x)
    
    dstWidth = srcWidth * x
    dstHeight = srcHeight * x

    #Numpy arrays
    srcData = np.asarray(srcImg)
    dstData = np.empty((dstHeight, dstWidth, 3), dtype=np.uint8)

    #mensagem para usuario
    print('Nearest zoom')
    print()
    print('Imagem original: ' + str(srcWidth) + ' x ' + str(srcHeight) + '.')
    print('Imagem ampliada: ' + str(dstWidth) + ' x ' + str(dstHeight) + '.')
    print()
    print('Executando...')

    #Percorre a imagem
    for i in range(srcHeight):
        for j in range(srcWidth):
            pixel = srcData[i][j]
            #expande a imagem
            for k in range(x):
                for l in range(x):
                    dstData[i*x+k][j*x+l] = pixel
                                      
    #Imagem destino
    dstImg = Image.fromarray(dstData)

    return dstImg

#Zoom por interpolação bilinear
def bilinearZoom(srcImg, factor):
    #inverte fator
    factor = 1.0/factor
    
    srcWidth = srcImg.width
    srcHeight = srcImg.height   
    
    dstWidth = int(srcWidth / factor)
    dstHeight = int(srcHeight / factor)

    #Numpy arrays
    srcData = np.asarray(srcImg, dtype=np.int)
    dstData = np.empty((dstHeight, dstWidth, 3), dtype=np.uint8)

    #indice de imagem origem
    i=0.0
    j=0.0
    #indices de imagem destino
    k=0
    l=0

    #posicao inicial do pixel
    #x = (dstWidth - srcWidth) / 2 * factor
    #y = (dstHeight - srcHeight) / 2 * factor
    x = 0
    y = 0


    #mensagem para usuario
    print('Bilinear interpolation zoom')
    print()
    print('Imagem original: ' + str(srcWidth) + ' x ' + str(srcHeight) + '.')
    print('Imagem ampliada: ' + str(dstWidth) + ' x ' + str(dstHeight) + '.')
    print()
    print('Executando...')

    #percorre a imagem
    l = 0
    for j in np.arange(y, srcWidth, factor):
        k = 0
        for i in np.arange(x, srcHeight, factor):
            #trata limites da imagem
            if (j >= srcWidth - 1):
                j = srcWidth - 2
            if (i >= srcHeight - 1):
                i = srcHeight - 2

            #Pixels utilizados
            #-----------------
            #1 2
            #3 4

            #f(i, j)
            pixel1 = srcData[int(i)][int(j)]

            #f(i, j+1)
            pixel2 = srcData[int(i)][int(j+1)]

            #f(i+1, j)
            pixel3 = srcData[int(i+1)][int(j)]

            #f(i+1, j+1)
            pixel4 = srcData[int(i+1)][int(j+1)]

            #variaveis temporarias
            x1 = 0.0    #f(i, y)
            x2 = 0.0    #f(i+1, y)
            x3 = 0.0    #f(x, y)

            #novo pixel
            newPixel = [0, 0, 0]

            #Algoritmo Zoom por Interpolação Bilinear
            #----------------------------------------
            
            #RGB
            #---
            for m in range (3):
                x1 = pixel1[m] + (j - math.floor(j)) * (pixel2[m] - pixel1[m])
                x2 = pixel3[m] + (j - math.floor(j)) * (pixel4[m] - pixel3[m])
                x3 = x1 + (i - math.floor(i)) * (x2 - x1)

                newPixel[m] = math.floor(x3 + 0.5)

            #Armazena pixel na imagem
            dstData[k][l] = newPixel
            
            k += 1          
        l += 1
                   
    #Imagem destino
    dstImg = Image.fromarray(dstData)

    return dstImg

#Zoom com Redes Neurais
#Fatores inteiros devem ser utilizados para simplificar o processo
def annZoom(srcImg, factor, n):
    srcWidth = srcImg.width
    srcHeight = srcImg.height
    
    dstWidth = int(srcWidth * factor)
    dstHeight = int(srcHeight * factor)

    #Numpy arrays
    srcData = np.asarray(srcImg, dtype=np.int)
    dstData = np.empty((dstHeight, dstWidth, 3), dtype=np.uint8)

    #mensagem para usuario
    print('Neural network zoom')
    print()
    print('Imagem original: ' + str(srcWidth) + ' x ' + str(srcHeight) + '.')
    print('Imagem ampliada: ' + str(dstWidth) + ' x ' + str(dstHeight) + '.')
    print()
    print('Executando...')

    #Vizinhança
    nCenter = int(n/2)

    #Rede Neural
    ann = artificialNeuralNetwork()

    #Conjunto de entradas para treinamento da rede neural
    XtrainingSet = np.empty((n*n, 2))
    r = 0
    for p in range(n):
        for q in range(n):
            XtrainingSet[r][0] = p
            XtrainingSet[r][1] = q
            r += 1

    #Teste
    #print('XtrainingSet')
    #print(XtrainingSet.shape)
    #print(XtrainingSet)
    #print()


    #Conjunto de predicao para consultar a rede neural
    predSetSize = int((n*factor) - (factor-1))
    predSetSize2 = predSetSize**2
    predSet = np.empty((predSetSize2, 2), dtype = np.float)

    r = 0
    for p in range(predSetSize):
        for q in range(predSetSize):
            predSet[r][0] = float(p / factor)
            predSet[r][1] = float(q / factor)
            r += 1

    #Teste
    #print('predSet')
    #print(predSet.shape)
    #print(predSet)
    #print()

    #Executa 1 vez para cada componente de cor
    for rgb in range(3):
        #Percorre a imagem
        i = 0
        while (i < srcWidth):
            j = 0
            while (j < srcHeight):           
            
                #Conjunto de saidas para treinamento da rede neural                
                YtrainingSet = np.empty((n*n, 1))
            
                #Para cada pixel, percorre a vizinhanca
                m = 0
                for k in range(n):
                    for l in range(n):
                        #coordenadas da imagem origem
                        x = i + k
                        y = j + l

                        #trata coordenadas fora dos limites da imagem
                        # x e y não pode ser menor que 0
                        if (x >= srcWidth):
                            x = srcWidth-1
                        if (y >= srcHeight):
                            y = srcHeight-1

                        YtrainingSet[m][0] = float(srcData[y][x][rgb])/255 #normaliza o dado

                        #itera indice
                        m += 1

                #Teste
                #print('YtrainingSet')
                #print(YtrainingSet.shape)
                #print(YtrainingSet)
                #print()
           
                #Treina rede neural
                ann.train(XtrainingSet, YtrainingSet, 32, 2500)

                #Estima valores com a rede neural
                resultSet = ann.pred(predSet)

                #Verifica taxa de erro da rede
                #resultSet = ann.pred(XtrainingSet, YtrainingSet)
    
                #Conjunto para testes
                #resultSet = []
                #for z in range(predSetSize2):
                #    list = [0.99]
                #    resultSet.append(list)

                #Teste
                #print('resultSet')
                #print(resultSet.shape)
                #print(resultSet)
                #print()

                #Monta imagem ampliada
                for r in range(predSetSize2):
                   
                    x = int((i + predSet[r][0])* factor)
                    y = int((j + predSet[r][1]) * factor)

                    if (x < 0):
                        x = 0
                    elif (x >= dstWidth):
                        x = dstWidth-1
                    
                    if (y < 0):
                        y = 0
                    elif (y >= dstHeight):
                        y = dstHeight-1

                    x2 = int(i + predSet[r][0])
                    y2 = int(j + predSet[r][1])

                    if (x2 < 0):
                        x2 = 0
                    elif (x2 >= srcWidth):
                        x2 = srcWidth-1
                    
                    if (y2 < 0):
                        y2 = 0
                    elif (y2 >= srcHeight):
                        y2 = srcHeight-1

                    #Decide se usa pixel da imagem original ou pixel previsto pela RNA
                    if (x % factor == 0 and y % factor == 0):                   
                        dstData[y][x][rgb] = srcData[y2][x2][rgb]
                    else:
                        dstData[y][x][rgb] = int(resultSet[r][0] * 255 + 0.5)

                #itera indices
                j += n - 1
            i += n - 1

    #Imagem destino
    dstImg = Image.fromarray(dstData)

    return dstImg

#Artificial Neural Network for Zoom class
class artificialNeuralNetwork:

    def __init__(self):

        self.counter = 0
        self.classifier = Sequential() # Initialising the ANN

        self.classifier.add(Dense(units = 64, kernel_initializer = 'normal', activation = 'relu', input_dim = 2))
        self.classifier.add(Dense(units = 64, kernel_initializer = 'normal', activation = 'relu'))
        self.classifier.add(Dense(units = 64, kernel_initializer = 'normal', activation = 'relu'))
        self.classifier.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'linear'))

        self.classifier.compile(optimizer = 'adam', loss = 'mse', metrics=['mae', 'mse'])     

    def train(self, XtrainingSet, YtrainingSet, batchSize, epochs):
        self.counter += 1
        start = time.time()
        self.classifier.fit(XtrainingSet, YtrainingSet, batch_size = batchSize, epochs = epochs, verbose = 0)
        stop = time.time()
        print('trained #' + str(self.counter) + ' in ' + str(stop - start) + ' secs')        


    def pred(self, predSet, testSet = None):
        start = time.time()
        Ypred = self.classifier.predict(predSet)
        stop = time.time()
        print('predicted #' + str(self.counter) + ' in ' + str(stop - start) + ' secs')

        if testSet is None:
            return Ypred

        #Compara conjunto previsto com conjunto de teste
        else:
            total = 0
            correct = 0
            wrong = 0
            for i in range(Ypred.size):
              total = total + 1

              if(int(testSet[i]*255+0.5) == int(Ypred[i]*255+0.5)):
                correct = correct + 1
              else:
                wrong = wrong + 1

            print("Correct: " + str(correct) + " Wrong: " + str(wrong))
            print("Total: " + str(total) + " Ratio: " + str(wrong/total))
            print()

            return Ypred