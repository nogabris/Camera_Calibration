import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from math import ceil

#Diretorio de Imagens a serem usadas para calibração
PATH_images = r"C:\Users\gabri\OneDrive\Documents\Python Scripts\RoboticaMovel\images\five_plane"
#Diretorio da Imagem com distorção a ser Corrigida
PATH_unidis=r"C:\Users\gabri\OneDrive\Documents\Python Scripts\RoboticaMovel\Reference.png"

#Identificando o tamanho do nosso tabuleiro de xadrez
chessboardSize = (24,17)#Quantidade de quadrados que iremos ter (Largura x Altura)
frameSize = (1440,1080)#Especificacoes da Camera(1280x720)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Set das variaveis para Identificacao de exata de cantos dentro da imagem
#Iremos preparar a parte de pontos para predefinir com base no tabuleiro qual sera
# Qual sera  a posicao dos nossos quadrados

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm


#Separaremos a imagens em pontos 3d e 2d, com representacao nas coordenadas reais e o plano 2d
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



images = glob.glob('*.png',root_dir=PATH_images)
images = [PATH_images+'\\'+image for image in images]
col = ceil(len(images)/2)

fig = plt.figure(figsize=(15, 10))
for index,image in enumerate(images,start=1):
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Procura encontrar os padroes das bordas conforme o tamanho tabuleiro que passamos
    # Retorna se foi encontrado os cantos do quadrados na cariavel Ret
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        #Encontraremos os cantos do tabuleiro
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        fig.add_subplot(2, col, index)
        plt.imshow(img)
plt.show()
# cv.destroyAllWindows()


#Calibração
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# ret : Variavel que verifica se foi possivel fazer a calibracao
# cameraMatrix : Relaciona as imagens opticas com as de ponto matriz 2d com 3d
# dist: retorna os parametros de distorcao
# rvecs:Retorna os valores do vetor de rotacao
# tvecs:Retorna os valores do vetor de tangenciação

print("Camera was Calibrated: ", bool(ret))
print("Camera Matrix: ", cameraMatrix)
print("Distortion Parameters: ", dist)
print("Translation Vectors: ", tvecs)
print("Rotation Vectors: ", rvecs)


############## UNDISTORTION #####################################################
img = cv.imread(PATH_unidis)
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)))

plt.subplots(1,2,figsize=(15, 10))
plt.subplot(121)
plt.imshow(img)
plt.title('Imagem Original Distorção')
plt.subplot(122)
plt.imshow(dst)
plt.title('Imagem Resultante Sem distorção')
plt.show()


