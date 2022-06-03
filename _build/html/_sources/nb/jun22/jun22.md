Summary
===============================

Summary of progress in May 2022

- **Ideas for setting up training procedure**
    1. Introduce a **third label**. The responses for the two classes (appr/ retr) oscillate. *We can define a segment as time points between the intersections of the response curves.* This is because at intersection influence of both the stimuli is similar, and at other time points one of them dominates. Thus such a segment would have least *contamination* from response for the other stimulus. Since it may be difficult to determine which class the fMRI responses at the intersection time points belong to, we can introduce a third target label: **Don't care, or indistinguishable, or unidentifiable**. So the time points between intersections can belong to either of the two classes and the points around intersections can belong to the new third class. 

    2. Use a **Temporal Convolutional Network (TCN)** and classify a segment into the two classes. TCN takes as input an image, does temporal convolution, and predicts a label for the image. In our case, input image can be the time series of all rois within a segment: a time x roi matrix. TCN will do temporal convolution and may find patterns of approach and retreat in the data. By computing importance scores, we can build intuition as to which rois at which time points are important for classification. 

    3. **Soft labels**. Why associate a time segment with a hard label? We can assign for each time point a vector as a label. Each element of this vector label will represent the degree of belongingness to each class. We can then provide a longer segment as input and let the model learn temopral dependencies. This may not be a classification set up, it may become a regression set up. We can use the canonical responses of each stimulus, [as in here](./02-understanding_emoprox2_stimulus.ipynb) as the soft labels. We will neither need the third *don't care* label, nor any alternative simpler model. The only issue will be of finding saliency methods for regression set ups.  


- **Progress**
    1. 
