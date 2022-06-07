import random as rd

import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
from pynput.keyboard import Key, Controller
from PIL import Image
import torchvision.transforms as transforms
import pygame, sys, time
from pygame.locals import *
from constants import *
from random import *
# https://data-flair.training/blogs/python-2048-game/

sizeofboard = 4
totalpoints = 0
defaultscore = 2
kb = Controller()
surface = pygame.display.set_mode((400, 500), 0, 32)
pygame.display.set_caption("2048 Game by DataFlair")
pygame.init()
font = pygame.font.SysFont("monospace", 40)
fontofscore = pygame.font.SysFont("monospace", 30)
tileofmatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
undomatrix = []
black = (0, 0, 0)
red = (255, 0, 0)
orange = (255, 152, 0)
deeporange = (255, 87, 34)
brown = (121, 85, 72)
green = (0, 128, 0)
lgreen = (139, 195, 74)
teal = (0, 150, 136)
blue = (33, 150, 136)
purple = (156, 39, 176)
pink = (234, 30, 99)
deepurple = (103, 58, 183)
deepblue = (7, 42, 108)
pointsPerMove=0
notDone = True
colordict = {
    0: black,
    2: red,
    4: green,
    8: purple,
    16: deepurple,
    32: deeporange,
    64: teal,
    128: lgreen,
    256: pink,
    512: orange,
    1024: deepblue,
    2048: brown
}


def main():
    mainInput=input("Please input what you want to do. 0 to Train a Model and Play a Round. Anything else to compare a model to a random agent and see them play. ")
    intInput=int(mainInput)
    if not intInput:
        print("Training New Model")
        trainGame()
    else:
        print("Starting Random Agent")
        randomAgent()
        print("Done With Random Agent")
        reset()
        playGame()
        reset()
    return 0


def trainGame():
    dataSet = []
    file1 = open("data.txt", "r")
    notEnd = True
    print("Reading Data")
    while notEnd:
        tempRow = []
        file_line = file1.readline()
        if not file_line:
            # print("End Of File")
            notEnd = False
        else:
            for nbr in file_line.split(','):
                tempRow.append(float(nbr))
            if len(tempRow) == 17:
                dataSet.append(tempRow)
    print("Done Reading Data")
    print("Dividing data into testing and training")
    file1.close()
    testingValues = []
    trainingValues = []
    trainingSet = []
    testingSet = []
    np.random.shuffle(dataSet)
    for i in range(0, 32000):
        trainingSet.append(dataSet[i])
    for i in range(32001, len(dataSet)):
        testingSet.append(dataSet[i])
    for i in trainingSet:
        trainingValues.append(i[16])
        del i[16]
    for i in testingSet:
        testingValues.append(i[16])
        del i[16]
    testingSet = tf.constant(testingSet)
    testingValues = tf.constant(testingValues)
    trainingSet = tf.constant(trainingSet)
    trainingValues = tf.constant(trainingValues)
    print("Done dividing data")
    print("Preparing to Train Neural Network")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(16,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(trainingSet, trainingValues, epochs=100)
    print("Evaluating Model on Testing Set")
    model.evaluate(testingSet, testingValues)
    model.save('sequentialModel2')
    playGame()


def getcolor(i):
    return colordict[i]

def randomAgent():
    global notDone
    placerandomtile()
    placerandomtile()
    while notDone:
        action=rd.randint(0,4)
        step(action)
def playGame():
    global pointsPerMove
    global notDone
    model = tf.keras.models.load_model("improvedSequentialModel")
    placerandomtile()
    placerandomtile()
    while notDone:
        normalData = 0
        sum=0
        normalArray = []
        tileArray = np.array(tileofmatrix)
        flatArray = tileArray.flatten()
        for i in range(0, flatArray.size):
            if flatArray[i] == 0:
                normalData = 0
            else:
                normalData = (math.log2((flatArray[i]))) / 12
            normalArray.append(normalData)
            sum+=normalData
        normalTensor = tf.constant(np.reshape(normalArray, (1, 16)))
        modelPred = model.predict(normalTensor)
        same = True
        reshaped = np.reshape(modelPred, (4,))
        predProbs = {
            0: reshaped[0],
            1: reshaped[1],
            2: reshaped[2],
            3: reshaped[3]
        }
        previousPoints=pointsPerMove
        while same:
            # if not notDone:
            #     return
            if predProbs:
                action = max(predProbs, key=predProbs.get)
            else:
                action=rd.randint(0,4)
            step(action)
            if previousPoints-pointsPerMove!=0:
                same=False
            if predProbs:
                del predProbs[action]

    return 0


def step(action):
    global notDone
    time.sleep(.2)
    if action == 0:
        kb.press(Key.up)  # Presses "up" key
        kb.release(Key.up)  # Releases "up" key
    elif action == 1:
        kb.press(Key.down)  # Presses "up" key
        kb.release(Key.down)  # Releases "up" key
    elif action == 2:
        kb.press(Key.left)  # Presses "up" key
        kb.release(Key.left)  # Releases "up" key
    elif action == 3:
        kb.press(Key.right)  # Presses "up" key
        kb.release(Key.right)  # Releases "up" key
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if checkIfCanGo():
            if event.type == KEYDOWN:
                if isArrow(event.key):
                    rotations = getrotations(event.key)
                    addToUndo()
                    for i in range(0, rotations):
                        rotatematrixclockwise()

                    if canmove():
                        movetiles()
                        mergetiles()
                        placerandomtile()

                    for j in range(0, (4 - rotations) % 4):
                        rotatematrixclockwise()
                    printmatrix()

        else:
            gameover()
            notDone = False
            return 

    pygame.display.update()


def canmove():
    for i in range(0, sizeofboard):
        for j in range(1, sizeofboard):
            if tileofmatrix[i][j - 1] == 0 and tileofmatrix[i][j] > 0:
                return True
            elif (tileofmatrix[i][j - 1] == tileofmatrix[i][j]) and tileofmatrix[i][j - 1] != 0:
                return True
    return False


def movetiles():
    for i in range(0, sizeofboard):
        for j in range(0, sizeofboard - 1):

            while tileofmatrix[i][j] == 0 and sum(tileofmatrix[i][j:]) > 0:
                for k in range(j, sizeofboard - 1):
                    tileofmatrix[i][k] = tileofmatrix[i][k + 1]
                tileofmatrix[i][sizeofboard - 1] = 0


def mergetiles():
    global totalpoints
    global pointsPerMove
    pointsPerMove=0
    for i in range(0, sizeofboard):
        for k in range(0, sizeofboard - 1):
            if tileofmatrix[i][k] == tileofmatrix[i][k + 1] and tileofmatrix[i][k] != 0:
                tileofmatrix[i][k] = tileofmatrix[i][k] * 2
                tileofmatrix[i][k + 1] = 0
                totalpoints += tileofmatrix[i][k]
                totalpoints += tileofmatrix[i][k]
                pointsPerMove += tileofmatrix[i][k]
                pointsPerMove += tileofmatrix[i][k]
                movetiles()

def getTotalPoints():
    return totalpoints


def getHighestTile():
    highestTile = 2
    for i in range(0, sizeofboard - 1):
        for k in range(0, sizeofboard - 1):
            if (tileofmatrix[i][k] > highestTile):
                highestTile = tileofmatrix[i][k]
    return highestTile


def getTileMatrix():
    return tileofmatrix


def placerandomtile():
    global sameGrid
    sameGrid=False
    c = 0
    for i in range(0, sizeofboard):
        for j in range(0, sizeofboard):
            if tileofmatrix[i][j] == 0:
                c += 1

    k = floor(random() * sizeofboard * sizeofboard)

    while tileofmatrix[floor(k / sizeofboard)][k % sizeofboard] != 0:
        k = floor(random() * sizeofboard * sizeofboard)
    randNum = np.random.randint(0, 1)
    if randNum == 0:
        tileofmatrix[floor(k / sizeofboard)][k % sizeofboard] = 2
    else:
        tileofmatrix[floor(k / sizeofboard)][k % sizeofboard] = 4


def floor(n):
    return int(n - (n % 1))


def printmatrix():
    surface.fill(black)
    global sizeofboard
    global totalpoints

    for i in range(0, sizeofboard):
        for j in range(0, sizeofboard):
            pygame.draw.rect(surface, getcolor(tileofmatrix[i][j]), (
                i * (400 / sizeofboard), j * (400 / sizeofboard) + 100, 400 / sizeofboard, 400 / sizeofboard))
            label = font.render(str(tileofmatrix[i][j]), 1, (255, 255, 255))
            label2 = fontofscore.render("YourScore:" + str(totalpoints), 1, (255, 255, 255))
            surface.blit(label, (i * (400 / sizeofboard) + 30, j * (400 / sizeofboard) + 130))
            surface.blit(label2, (10, 20))


def checkIfCanGo():
    for i in range(0, sizeofboard ** 2):
        if tileofmatrix[floor(i / sizeofboard)][i % sizeofboard] == 0:
            return True

    for i in range(0, sizeofboard):
        for j in range(0, sizeofboard - 1):
            if tileofmatrix[i][j] == tileofmatrix[i][j + 1]:
                return True
            elif tileofmatrix[j][i] == tileofmatrix[j + 1][i]:
                return True
    return False


def convertToLinearMatrix():
    mat = []
    for i in range(0, sizeofboard ** 2):
        mat.append(tileofmatrix[floor(i / sizeofboard)][i % sizeofboard])

    mat.append(totalpoints)
    return mat


def addToUndo():
    undomatrix.append(convertToLinearMatrix())


def rotatematrixclockwise():
    for i in range(0, int(sizeofboard / 2)):
        for k in range(i, sizeofboard - i - 1):
            temp1 = tileofmatrix[i][k]
            temp2 = tileofmatrix[sizeofboard - 1 - k][i]
            temp3 = tileofmatrix[sizeofboard - 1 - i][sizeofboard - 1 - k]
            temp4 = tileofmatrix[k][sizeofboard - 1 - i]

            tileofmatrix[sizeofboard - 1 - k][i] = temp1
            tileofmatrix[sizeofboard - 1 - i][sizeofboard - 1 - k] = temp2
            tileofmatrix[k][sizeofboard - 1 - i] = temp3
            tileofmatrix[i][k] = temp4


def gameover():
    global totalpoints
    surface.fill(black)

    label = font.render("gameover", 1, (255, 255, 255))
    label2 = font.render("score : " + str(totalpoints), 1, (255, 255, 255))
    print(totalpoints)
    surface.blit(label, (50, 100))
    surface.blit(label2, (50, 200))


def reset():
    global totalpoints
    global tileofmatrix
    global notDone
    notDone= True
    totalpoints = 0
    surface.fill(black)
    tileofmatrix = [[0 for i in range(0, sizeofboard)] for j in range(0, sizeofboard)]


def isArrow(k):
    return (k == pygame.K_UP or k == pygame.K_DOWN or k == pygame.K_LEFT or k == pygame.K_RIGHT)


def getrotations(k):
    if k == pygame.K_UP:
        return 0
    elif k == pygame.K_DOWN:
        return 2
    elif k == pygame.K_LEFT:
        return 1
    elif k == pygame.K_RIGHT:
        return 3


main()
