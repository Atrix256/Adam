import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import glob
import os
from PIL import Image

def MakeGif(csvName):
    frameNameBase = os.path.splitext(csvName)[0]
    frameNames = []
    i = 0
    while os.path.isfile(frameNameBase + "_" + str(i) + ".png"):
        frameNames.append(frameNameBase + "_" + str(i) + ".png")
        i = i + 1
    frames = [Image.open(image) for image in frameNames]
    frame_one = frames[0]
    frame_one.save(frameNameBase + ".gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    

fileNames = glob.glob("out/*.csv")

color1 = (0.949020, 0.474510, 0.474510)
color2 = (0.474510, 0.611765, 0.949020)
color3 = (0.749020, 0.949020, 0.474510)
color4 = (0.949020, 0.474510, 0.890196)
color5 = (0.474510, 0.949020, 0.866667)
color6 = (0.949020, 0.729412, 0.474510)

'''
palette = itertools.cycle(sns.color_palette())
color1 = next(palette)
color2 = next(palette)
color3 = next(palette)
color4 = next(palette)
color5 = next(palette)
color6 = next(palette)
'''

for fileName in fileNames:
    print(fileName)
    MakeGif(fileName)
    df = pd.read_csv(fileName).drop(['Step'], axis=1)

    fig, (ax1,ax2,ax3) = plt.subplots(1, 3)

    fig.set_figwidth(6.4*3)
    fig.set_figheight(4.8)

    ax1.plot(df["GD LR 0.010000"], label="GD LR 0.010000", color=color1)
    ax1.plot(df["GD LR 0.001000"], label="GD LR 0.001000", color=color2)
    ax1.plot(df["GD LR 0.000100"], label="GD LR 0.000100", color=color3)
    ax1.plot(df["Adam Alpha 0.100000"], label="Adam Alpha 0.100000", color=color4)
    ax1.plot(df["Adam Alpha 0.010000"], label="Adam Alpha 0.010000", color=color5)
    ax1.plot(df["Adam Alpha 0.001000"], label="Adam Alpha 0.001000", color=color6)

    ax2.plot(df["GD LR 0.010000"], label="GD LR 0.010000", color=color1)
    ax2.plot(df["GD LR 0.001000"], label="GD LR 0.001000", color=color2)
    ax2.plot(df["GD LR 0.000100"], label="GD LR 0.000100", color=color3)

    ax3.plot(df["Adam Alpha 0.100000"], label="Adam Alpha 0.100000", color=color4)
    ax3.plot(df["Adam Alpha 0.010000"], label="Adam Alpha 0.010000", color=color5)
    ax3.plot(df["Adam Alpha 0.001000"], label="Adam Alpha 0.001000", color=color6)

    themin = df.min().min()
    themax = df.max().max()

    ax1.set_ylim(themin, themax)
    ax2.set_ylim(themin, themax)
    ax3.set_ylim(themin, themax)

    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)
    ax3.set_xlim(0, 100)

    ax1.legend()
    ax2.legend()
    ax3.legend()

    fig.suptitle('20 Points Each, Average Height : ' + fileName)

    ax1.set_ylabel('Height')
    ax1.set_xlabel('Step')

    ax2.set_ylabel('Height')
    ax2.set_xlabel('Step')

    ax3.set_ylabel('Height')
    ax3.set_xlabel('Step')

    '''
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)

    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log', base=2)
    '''

    fig.tight_layout()
    fig.savefig(fileName + ".png", bbox_inches='tight')
