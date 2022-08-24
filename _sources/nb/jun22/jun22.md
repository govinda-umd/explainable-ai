Summary
===============================

Summary of progress in June 2022

- **MAX Paradigm**
    1. This paradigm was designed by previous lab members and fMRI scans of 109 participants were collected. This is the main paper: [@murty_distributed_2022]. It found that responses to threat and safety in the early periods differed from those in the late periods in many ROIs. Some ROIs displayed sustained responses to the two conditions, and some others displayed transient profiles. Overall, effect of anxious anticipation is **distributed** across the entire brain, including the *subcortical* regions.

- **Progress**
    1. [**Data**](./00-max_dataset.ipynb) I extracted the fMRI time series for the MAX paradigm and grouped the subjects into two equal halves: *training*/*exploratory* set and an *unseen* set. I do all analyses on the training set and plan to report the final results on the unseen set to avoid circularity in the analysis. To test for generalizability of the models, I split the training set into two groups: training and validation, and track model's performances on the validation set. For each subject, I extracted the time series for all the *fake* stimulus blocks.

    2. [**Models**](./01-max_data_pytorch_models.ipynb) I used two classes of models; one, a feed forward neural network (FFNN) model to classify data based on time averaged responses, and second, a recurrent neural network (RNN) model to utilise temporal patterns in the data for classification. All models are written using PyTorch. 
        1. The FFNN is a two layer multilayer perceptron (MLP) that takes in time averaged ROI responses within a stimulus block and predicts the corresponding stimulus, threat/safe. Training loss keeps decreasing while validation loss stabilizes after certain epochs. But as anticipated, training accuracy increases beyond 0.8 with negligible rate, and validation accuracy plateaus ~0.7. This indicates that we need a more sophisticated model that can process time series data. 

        2. In order to do that, I used RNN. It is a gated recurrent unit (GRU) developed previously by me and Joyneel that takes in the time series of all ROIs within a block as input and predicts the stimulus present for that block. The model has a single hidden layer with $32$ hidden units and a fully connected layer to transform hidden state of the input to stimulus prediction. Very surprisingly, the validation accuracy is ~0.65. Moreover, training accuracy and validation accuracy curves diverge indicating that the model is overfitting. 
    
    3. **Simulating data** But since the real data is very noisy, how do I make sure whether the model learns the relevant patterns in the data and not fit noise? The goal is to see how well the NN can do with data with very little noise. Is that a trivial problem to learn? Or very difficult even with just a little bit of noise. Real data have a ton of noise.
        1. [White noise](01-simulated_data_white_noise.ipynb) One way Luiz suggested was to simulate the time series by controlling for structure of noise. I can increase complexity of noise and converge the statistical properties of the simulated data to the real data. To start, I added an **i.i.d. distributed Gaussian** noise over the mean time series of each stimulus and created input samples. This is called *White noise*. I set the mean and standard deviation of the White noise to be that of the real data. I also scaled the standard deviation with a coefficient $\text{noise\_level} \gte 0$ to control for amount of noise w.r.t. the mean signal. $\text{noise\_level} = 1.0$ means that the std. of the noise is same as that of the real data, and $\text{noise\_level} = 0.5$ means the std. of the noise is half of that of the real data. I varied the noise level from 0.0 through 5.0 and evaluated GRU model's performance. At noise level of 1.0, validation accuracy is 0.78, meaning this is like a soft ceiling for model's performance. Additionally, I found that reducing noise level from 1.0 to 0.5 increases validation accuracy from 0.78 to 0.91 and also increasing noise level from 1.0 to 2.0 validation accuracy drops from 0.78 to 0.60, which are significant. Since the simulated data is one realization of the noise distribution, I repeated the procedure for $1000$ times, plotted the validation accuracy curves, and verified that the accuracy is around $0.78$.

        2. [Wide sense stationary](01-simulated_data_wss_noise.ipynb) Need to do this.

    4. [**Interpreting the model**](02-interpretations_simulated_data_white_noise.ipynb) Now the main part, explaining model's predictions in terms of parts of the input. This is typically done by attributing importance scores to each input feature; the more the score, the more the importance. It is important to note that for multi-label-prediction model, each label will have its own explanations in the input. I started with the standrad interpretation methods: Saliency and Integrated gradients. 
    Saliency is a simple approach for computing input attributions, returning the gradient of the output w.r.t. the input. This approach can be understood as writing the total differential of the output in terms of the differentials of input features. The partial derivatives of the output w.r.t. the input features become input attributions. 
    Integrated gradients represents the integral of gradients w.r.t. inputs along the path from a given baseline to input. This approach satisfies the axioms of sensitivity and implementation invariance. More details can be found in the [original paper](https://doi.org/10.48550/arXiv.1703.01365).
    Luiz and I expected that attributions should reflect *visible* patterns in the inputs, i.e. 
        1. attributions for all input features follow a very similar pattern,
        2. if a part of time series goes down and is not important, attributions are still high. 
    *The results were not consistent with our understanding.*

    5. [**Verification on MNIST data**](03-pytorch_quickstart.ipynb) As a final way to verify whether I was doing everything right, I trained a simple MLP model on a benchmark images dataset, famously known as MNIST digits dataset, and interpreted its predictions using Saliency and Integrated Gradients. MNIST digits dataset is a large collection of handwritten digits that is used as a benchmark for training any novel machine learning/deep learning model. Attributions using both methods met our expectations, i.e. attributions reflected the important parts of the input image essential for prediction.  
    
    Luiz wanted to see *what does the model see for a class of input*, e.g. in MNIST a class is a digit (say $7$). However, all the methods I know of compute attributions for *a* particular input, not for a class of inputs. So I did the following:
        1. took average of all the images of each class and computed attributions for these (averaged) inputs, 
        2. computed attributions for a white noise image.
    Surprisingly, the attributions for these two types of images were qualitatively the same as those for a particular image. This meant that we could observe *what the model see for a class of input*.

But since results on fMRI data were not promising, we thought of shifting our attention to some other computational tools and come back here in future.
