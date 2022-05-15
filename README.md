# HPML
<h1>Distributed training of GANs</h1>

<h3> <u> OVERVIEW: </u> </h3>
We implement different training strategies for creating a Generator to build fake abstract art images. <br>
We profile different training methods to understand the bottleneck in our training pipeline. <br>
We implement training strategies to help improve these bottlenecks to improve the training speed while maintaining the quality of our GANs. <br>

<h3> <u> ARCHITECTURE: </u> </h3>

![image](https://user-images.githubusercontent.com/47019139/168489567-050c0a44-8253-4208-8270-1f178e87d20c.png)

<h3> <u> REPOSITORY: </u> </h3>
GDLoss.png - Graph plot of the Generator Loss and the Discriminator Loss of a GAN.
animation.gif - A GIF file to show the progress of images generated over 200 epochs.
dcgan.py - Main Python file to run in sbatch which contains the training logic.
slowed_down_looped_once.gif - animation.gif slowed down and looped once for better visual understanding.
test.out - Output file generated using running of sbatch. 
