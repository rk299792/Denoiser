# Denoiser

This project is divided into three parts. First, I denoised the first mode of noisy images (task 1) using the REDCNN architecture. Secondly, I applied continual learning to denoise the second mode of images (task 2). Finally, I incorporated EWC regularization to make the continual learning process robust and prevent catastrophic forgetting.


### Making Noisy Images: 
I obtained the cat and dog images from Kaggle and cropped them to a size of $224*224$ before converting them to grayscale. For the mode I noisy images, I added Gaussian noise with a mean of 0 and standard deviation of 20 to each pixel. For the mode II noisy images, I added Gaussian noise with a mean of 10 and standard deviation of 60 to each pixel.

### Part 1 Goal: 
The goal of Part 1 is to train a neural network denoising framework that maps a noisy image $(X)$ to its corresponding ground truth image $(Y)$. In this context, $X$ represents the noisy images, while $Y$ represents the images before adding the noise.

### Task1:
To denoise the Type I images, I employed the REDCNN architecture. However, due to computational constraints, I used only 1000 images and trained the model for 2 epochs.

### Part 2 Goal: 
The goal of Part 2 is to demonstrate catastrophic forgetting during continual learning and implement EWC regularization to prevent it.

### Task 2 and Continual Learning:

Continual Learning is a concept that involves learning a model for multiple tasks sequentially without forgetting knowledge obtained from the preceding tasks, even when data from previous tasks are no longer available during training. In our case, Task 1 involved denoising mode I images, while Task 2 involved denoising mode II images. After training the model with Task 1, I used the weights of that model to initialize training for Task 2. However, when testing this model (task II) with Type I test images, its performance significantly decreased, which is known as catastrophic forgetting.

### EWC implementation: 
To prevent catastrophic forgetting, I implemented EWC regularization. EWC encourages the parameters with high Fisher information value to remain close to the parameters from the previous task.  Which means EWC retains the important parameters from the previous task and encourages other parameters to be optimized for Task 2. The lambda parameter was chosen based on the importance given to the previous task.

$$EWC=\lambda \sum_{i=1}^{i=p}F_i(\theta^*-\theta)^2$$

In this context, p represents the number of parameters in the model, $\theta^*$ represents the parameters obtained from the previous task, and $F_i$ represents the Fisher information(diagonal value of the Fisher information matrix) from the previous task. I obtained the Fisher information matrix by squaring the gradient of the loss.
