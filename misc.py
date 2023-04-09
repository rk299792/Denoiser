import torch

import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_psnr(testset,model):
    model.eval()
    train_psnr=[]
    for i in range(len(testset)):
        noise,gt=testset[i]
        train_psnr.append(cv2.PSNR(model(noise.view(-1,1,224,224)).detach().numpy().reshape(1,224,224),gt.numpy()))
    print(f"PSNR after model: {sum(train_psnr)/len(train_psnr)}")

    return train_psnr

def gt_psnr(testset):
    gt_psnr=[]
    for i in range(len(testset)):
        noise,gt=testset[i]
        gt_psnr.append(cv2.PSNR(noise.numpy(),gt.numpy()))
    print(f"PSNR test set: {sum(gt_psnr)/len(gt_psnr)}")
    return gt_psnr


def check_images(trainset_image,testset_image,model):

    input_image,target_image=trainset_image

    input_image=input_image.view(-1,1,224,224)
    target_image=target_image.view(-1,1,224,224)

    input_image2,target_image2=testset_image
    input_image2=input_image2.view(-1,1,224,224)
    target_image2=target_image2.view(-1,1,224,224)
    with torch.no_grad():
        netoutput_image=model(input_image)
        netoutput_image2=model(input_image2)



    fig, ax = plt.subplots(2, 3,figsize=(12, 8))


    ax[0][0].imshow(np.transpose(input_image.view(1,224,224).to(torch.uint8),(1,2,0)),cmap="gray")
    ax[0][1].imshow(np.transpose(target_image.view(1,224,224).to(torch.uint8),(1,2,0)),cmap="gray")
    ax[0][2].imshow(np.transpose(netoutput_image.view(1,224,224).to(torch.uint8),(1,2,0)),cmap="gray")

    ax[1][0].imshow(np.transpose(input_image2.view(1,224,224).to(torch.uint8),(1,2,0)),cmap="gray")
    ax[1][1].imshow(np.transpose(target_image2.view(1,224,224).to(torch.uint8),(1,2,0)),cmap="gray")
    ax[1][2].imshow(np.transpose(netoutput_image2.view(1,224,224).to(torch.uint8),(1,2,0)),cmap="gray")

    # Add titles to each subplot
    ax[0][0].set_title('Noisy Image')
    ax[0][1].set_title('Ground Truth Image')
    ax[0][2].set_title('Denoised Image')

    # # Remove the x and y ticks
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])

    # Show the plot
    plt.show()