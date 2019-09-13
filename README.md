# Scale-Invariant-Feature-Transform

### Objective:

The objective of this task is to detect key points in an image which are the first three steps of
Scale-Invariant Feature Transform (SIFT).

### Approach:

1. Generate 4 octaves of the images:

      a. We take the original image, and generate progressively blurred out images.
Then, you resize the original image to half size. And you generate blurred out
images again. And you keep repeating.

      b. Images of the same size form an octave. We create total four octaves. Each
octave has 5 images. The individual images are formed because of the
increasing amount of blur.

      c. Total of 20 images are generated in this step.

2. Generate Difference of Gaussians:

      a. We calculate the differences between two consecutive images in each octave.

      b. Total of 16 images are generated in this step.

3. Compute key points (Maxima and Minima points):

      a. We iterate through each pixel and check all it's neighbours. The check is done
within the current image, and also the one above and below it.

      b. The maximum and minimum points are located and marked on the original
images.

4. The generated image at the end of step 3, gives us the desired result.me 
