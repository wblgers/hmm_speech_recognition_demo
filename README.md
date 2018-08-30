# hmm_speech_recognition_demo

### 0. Setup Environment
This demo project is running on python2.x, please install the following required packages as well:
- scikits.talkbox: Calculation of MFCC features on audio 
- hmmlearn: Hidden Markov Models in Python, with scikit-learn like API	
- scipy: Fundamental library for scientific computing

All the three python packages can be installed via `pip install`, on Python3.x, the package `scikits.talkbox` can't be installed correctly for me.


### 1. Description
#### 1.1 Problem
By utilizing the `GMMHMM` in `hmmlearn`, we try to model the audio files in 10 categories. `GMMHMM` model provides easy interface to train a HMM model and to evaluate the score on test set.

Please more details in the [doc](https://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn-hmm) of `hmmlearn`.
#### 1.2 Dataset
It's a demo project for simple isolated speech word recognition. There are only 100 audio files with extention of `.wav` for training, and 10 audio files for testing. To be more specified:

- Train: For digits 0-9, each with 10 sampels with Chinese pronunciation
- Test:  For digits 0-9, each with 1 sample with Chinese pronunciation

#### 1.3 Demo running results:
In python2.x, run the script `demo.py`, get the result below:
```
Finish prepare the training data
Finish training of the GMM_HMM models for digits 0-9
Test on true label  10 : predict result label is  9
Test on true label  1 : predict result label is  1
Test on true label  3 : predict result label is  3
Test on true label  2 : predict result label is  2
Test on true label  5 : predict result label is  5
Test on true label  4 : predict result label is  4
Test on true label  7 : predict result label is  7
Test on true label  6 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  8 : predict result label is  9
Final recognition rate is 80.00 %
```

### 2. Tricky bug workaround in hmmlearn
In the training of HMM model, there may lead to negative value of `startprob_`. By checking the code of hmmlearn, I found the following code in `hmmlearn/base.py` is suspicious:
```
def _do_mstep(self, stats):
    """Performs the M-step of EM algorithm.
    Parameters
    ----------
    stats : dict
        Sufficient statistics updated from all available samples.
    """
    # The ``np.where`` calls guard against updating forbidden states
    # or transitions in e.g. a left-right HMM.
    if 's' in self.params:
        startprob_ = self.startprob_prior - 1.0 + stats['start']
        self.startprob_ = np.where(self.startprob_ == 0.0,
                                   self.startprob_, startprob_)
        normalize(self.startprob_)
    if 't' in self.params:
        transmat_ = self.transmat_prior - 1.0 + stats['trans']
        self.transmat_ = np.where(self.transmat_ == 0.0,
                                  self.transmat_, transmat_)
        normalize(self.transmat_, axis=1)
```
When updating the `self.startprob_` and `self.transmat_` in every step, it will subtract 1.0 from the old value, in this situation, it will be very likely to lead to a negative value for both `self.startprob_` and `self.transmat_`.

Meanwhile, this issue is also submitted in the issue list of `hmmlearn`, but no response from the maintainer of `hmmlearn`:

[startprob_ of ghmm is negative or nan #276](https://github.com/hmmlearn/hmmlearn/issues/276)

For a temporary workaround, I modify this part of code to get rid of the subtraction, since there is a `normalize` after the update.
```
def _do_mstep(self, stats):
    """Performs the M-step of EM algorithm.

    Parameters
    ----------
    stats : dict
        Sufficient statistics updated from all available samples.
    """
    # The ``np.where`` calls guard against updating forbidden states
    # or transitions in e.g. a left-right HMM.
    if 's' in self.params:
        # startprob_ = self.startprob_prior - 1.0 + stats['start']
        startprob_ = self.startprob_prior + stats['start']
        self.startprob_ = np.where(self.startprob_ == 0.0,
                                   self.startprob_, startprob_)
        normalize(self.startprob_)
    if 't' in self.params:
        # transmat_ = self.transmat_prior - 1.0 + stats['trans']
        transmat_ = self.transmat_prior + stats['trans']
        self.transmat_ = np.where(self.transmat_ == 0.0,
                                  self.transmat_, transmat_)
        normalize(self.transmat_, axis=1)
```

Please note: This modification is just a wordaround, official solution will be followed up if there is any update.  