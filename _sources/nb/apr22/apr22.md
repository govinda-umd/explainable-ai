Summary
===============================

Summary of progress in April 2022

- **Emoprox2 full dataset**:
    I got the full data of emoprox2 paradigm for the 300 ROI Schaefer parcellation, the MAX paradigm’s 85 ROIs parcellation. 
    The paradigm stimulus was given as follows: each participant goes through 6 runs, each with 2 blocks of stimuli. A block may have shock events. So the data is naturally partitioned into continuous segments /windows between start of block to shock onset, shock offset to next shock onset, and shock offset to the end of the block. 
    I used these segments as my data samples. The data contains time series from all the ROIs - 300 or 85 depending on the parcellation.
    In order to incorporate hemodynamic lag in the fMRI responses w.r.t. to the stimulus, I selected time series by shifting each continuous segment by 2TRs and 3TRs.
    With these datasets I trained the GRU model, and checked whether their performances are above chance.

    1. Although model's accuracy is ~0.7, this single value is not representative of how the model *sees* the input time series. i.e. information from which time points does the model use more(less) to classify the segment correctly?

    2. Since we have the stimulus timing files, we should understand how the stimulus and its following fMRI response vary with time. Can we clearly separate approach segments from retreat segments? Ans. No! The paradigm was designed to study how threat anticipation (approach) and relaxation (retreat) phases interact, and how does one phase transition into the other. What are the dynamic patterns of roi activities that encode such stimuli?

- **Observations from emoprox2 stimuli**. [](./02-understanding_emoprox2_stimulus.ipynb)
    1. The plots show that the (canonical) fMRI response peaks at around 8.75TRs. And the information about the stimulus is present from around 5(or 6)TRs. So we should take segments of the fMRI signal shifted by 5(6)TRs from the onset of stimulus. 

    2. We also observe that the responses for the two classes (appr/ retr) oscillate. We can define a segment as time points between the intersections of the response curves. This is because at intersection influence of both the stimuli is similar, and at other time points one of them dominates. Thus such a segment would have least *contamination* from response for the other stimulus.

    3. Since it may be difficult to determine which class the fMRI responses at the intersection time points belong to, we can introduce a third target label: **Don't care, or indistinguishable, or unidentifiable**. So the time points between intersections can belong to either of the two classes and the points around intersections can belong to the new third class. 

- 


- 

- ...
