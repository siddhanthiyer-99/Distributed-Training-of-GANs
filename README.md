# HPML
<h1>Distributed training of GANs</h1>

![slowed_down_looped_once](https://user-images.githubusercontent.com/47019139/168666158-5f1450e6-6c74-476f-8637-1dfc7f256ace.gif) <br>

<h3> <u> OVERVIEW: </u> </h3>
We implement different training strategies for creating a Generator to build fake abstract art images. <br>
We profile different training methods to understand the bottleneck in our training pipeline. <br>
We implement training strategies to help improve these bottlenecks to improve the training speed while maintaining the quality of our GANs. <br>

<h3> <u> ARCHITECTURE: </u> </h3>

![image](https://user-images.githubusercontent.com/47019139/168489567-050c0a44-8253-4208-8270-1f178e87d20c.png)

<h3> <u> REPOSITORY: </u> </h3>
GDLoss.png - Graph plot of the Generator Loss and the Discriminator Loss of a GAN. <br><br>
animation.gif - A GIF file to show the progress of images generated over 200 epochs. <br><br>
cpu.out - Output file generated using running of cpu.sbatch. <br><br>
cpu.sbatch - Sbatch file to run on CPU. <br><br>
dcgan.py - Main Python file to run in sbatch which contains the training logic. <br><br>
rtx.out - Output file generated using running of rtx.sbatch. <br><br>
rtx.sbatch - Sbatch file to run on rtx8000 GPU. <br><br>
slowed_down_looped_once.gif - animation.gif slowed down and looped once for better visual understanding. <br><br>
v100.out - Output file generated using running of b100.sbatch. <br><br>
v100.sbatch - Sbatch file to run on v100 GPU. <br><br>
v100_GDLoss.png - Graph plot of the Generator Loss and the Discriminator Loss of a GAN on v100 GPU. <br><br>
v100animation.gif - A GIF file to show the progress of images generated over 200 epochs on v100 GPU. <br><br>

<h3> <u> HOW TO RUN: </u> </h3>
1. Run the following command for GPU - sbatch training.sbatch <br><br>
2. Run the following command for CPU - python dcgan.py --device cpu <br><br>

<h3> <u> RESULTS: </u> </h3>

![image](https://user-images.githubusercontent.com/47019139/168666005-45aef600-f980-4e4b-8ba6-d560f416bb94.png)



