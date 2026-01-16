# Audio-to-Text-Translation-System
ASR model is developed using a Convolutional Neural Network (CNN) followed by a Bidirectional Long Short-Term Memory (BiLSTM) network and trained using Connectionist Temporal Classification (CTC) loss.
DATASET LINK:https://www.openslr.org/resources/12/dev-clean.tar.gz

1. Introduction

Automatic Speech Recognition (ASR) is a fundamental problem in speech and language processing that involves converting an acoustic speech signal into its corresponding textual representation. The task is inherently challenging due to variability in speaking rate, pronunciation(accent), speaker characteristics, background conditions, and alignment between audio frames with their respective characters.

In this work, an end-to-end character-level ASR model is developed using a Convolutional Neural Network (CNN) followed by a Bidirectional Long Short-Term Memory (BiLSTM) network and trained using Connectionist Temporal Classification (CTC) loss.

The model operates on Mel-spectrogram features extracted from raw speech signals sampled at 16 kHz, which provide a meaningful time–frequency representation of audio. A convolutional module employed to learn robust local acoustic patterns, such as formant structures and short-term spectral variations, while reducing spectral dimensionality. This is followed by a BiLSTM module that captures long-range temporal dependencies in both forward and backward directions, which is essential for modeling contextual information in continuous speech.

Unlike frame-level classification approaches, this system does not require explicit alignment between acoustic frames and character labels. Instead, CTC loss is used to enable alignment-free sequence learning, allowing the model to learn optimal alignments internally during training. This makes the approach suitable for real-world speech datasets where manual alignment is impractical.

To improve generalization and robustness, the training pipeline incorporates data augmentation techniques, including speed perturbation at the waveform level and SpecAugment (time and frequency masking) at the feature level. Furthermore, the model is evaluated using k-fold cross-validation, ensuring that performance is not biased by a single train–validation split and providing a more reliable estimate of generalization ability.


2. Problem Formulation
Given an input speech signal x represented as a sequence of acoustic frames, the objective is to predict the corresponding transcription y which is a sequence of characters:

x->y i.e.(y1,y2,...,yT)

Since the input speech sequence typically contains a much larger number of acoustic frames than the target transcription, and explicit frame-level alignments between acoustic frames and output characters are not available, the ASR task is formulated as an alignment-free sequence learning problem.

3. Dataset Description
The model is trained on a LibriSpeech dataset consisting of:

i)Audio files in .flac format

ii)Corresponding transcription files (.trans.txt)
Each transcription file maps an audio file identifier to its textual content.

iii)Sampling rate: 16 kHz

4. Audio Preprocessing
Audio preprocessing plays a crucial role in improving the stability, convergence, and generalization of speech recognition models. In this work, preprocessing is performed at both the waveform level and the feature level, ensuring that the input data is consistent and robust to real-world variability.

4.1) Signal Normalization
Each input audio signal undergoes a sequence of normalization steps before feature extraction.

i)Audio is converted to mono (if multi-channel):
Speech recordings may contain multiple channels. To ensure uniform processing and avoid channel-dependent variations, all multi-channel audio signals are converted to mono by averaging across channels.

ii)Resampled to 16 kHz:
All audio signals are resampled to a fixed sampling rate of 16 kHz. This sampling rate is widely used in speech recognition as it captures the frequency range relevant to human speech while maintaining computational efficiency.

iii)Amplitude normalization is applied:
To reduce variations caused by different recording volumes and microphone gains, amplitude normalization is applied.

4.2) Data Augmentation
To improve robustness and generalization, the following augmentations are applied during training only:

i)Speed Perturbation:
Speed perturbation modifies the temporal characteristics of the speech signal by slightly altering the playback speed. In this work, the audio waveform is randomly resampled to simulate speaking rate variations of 0.9× or 1.1×.Helps the model generalize to unseen speaking styles.
Importantly, pitch characteristics are preserved, allowing the content to remain unchanged.

ii)SpecAugment:
SpecAugment is applied at the Mel-spectrogram level and introduces structured noise directly in the time–frequency domain. Two types of masking are used:
#Frequency masking:
A contiguous range of Mel frequency channels is randomly masked (set to zero). This encourages the model to:
Avoid reliance on narrow frequency bands,Learn robust spectral representations and Remain invariant to frequency-specific distortions such as channel noise.
The Frequency mask width is selected as a proportion(20%) of the total number of Mel bins.

#Time masking:
A contiguous segment along the time axis is randomly masked. This simulates:
Missing or corrupted audio segments and Short pauses in speech.
Time masking forces the model to leverage contextual information from surrounding frames, improving temporal robustness

5. Feature Extraction
Feature extraction converts raw speech waveforms into a structured representation that is suitable for neural network–based acoustic modeling. In this work, Mel-spectrogram features are used, as they closely approximate the human auditory perception of sound.

5.1) Time–Frequency Analysis:
Speech is a non-stationary signal whose frequency content changes over time. To capture these temporal variations, the input waveform is divided into short overlapping frames, and a Short-Time Fourier Transform (STFT) is applied to each frame, describes how the spectral content of speech evolves over time.

5.2) Mel Filterbank Representation:
The linear-frequency spectrum obtained from the STFT is projected onto the Mel scale, which is a perceptual scale designed to mimic the frequency resolution of the human auditory system.The resulting Mel-spectrogram represents the distribution of energy across Mel frequency bands over time.

5.3) Feature Extraction Parameters:
The following parameters are used for Mel-spectrogram computation:
i)Number of Mel filters: 80
Provides a detailed yet compact spectral representation suitable for deep learning models.
ii)FFT size: 512
Determines the frequency resolution of the STFT.
iii)Window length: 400 samples (~25 ms)
Captures sufficient phonetic information while maintaining temporal locality.
iv)Hop length: 160 samples (~10 ms)
Controls frame overlap and temporal resolution.

These values are commonly used in state-of-the-art ASR systems and represent a balance between time resolution, frequency resolution, and computational efficiency.

5.4) Log-Scaled Spectral Representation:
The Mel-spectrogram implicitly emphasizes lower-energy components that are important for speech perception. This representation is robust to small variations in amplitude and background noise, making it suitable for acoustic modeling.

5.5) Feature Normalization:
To stabilize training and reduce variability across utterances, global mean and variance normalization is applied to each Mel-spectrogram:

X^=X−mu/(sigma+epsilon)
where:
X is the Mel-spectrogram,
mu is the mean,
sigma is the standard deviation,
epsilon is a small constant for numerical stability.

6. Dataset Pipeline and Data Handling
6.1) ASRDataset Class
The ASRDataset class is responsible for data loading, preprocessing, and augmentation.
#Audio Loading
Loads .flac audio files, Converts multi-channel audio to mono, Resamples audio to 16 kHz and Applies amplitude normalization

#Feature Extraction
Converts waveform to Mel-spectrogram, Applies global mean–variance normalization.

#Data Augmentation (Training Only)
Speed perturbation and SpecAugment(Frequency masking,Time masking)

#Text Processing
Converts transcript text into character-level label indices and Uses a predefined character vocabulary

The dataset returns-->(MelSpectrogram,LabelSequence)

6.2) Collate Function:
Due to variable-length speech sequences, a custom collate_fn is used to prepare mini-batches.
Functions of collate_fn:
#Sorting by Sequence Length
Batch samples are sorted by decreasing time length.
Improves efficiency and stability of sequence modeling.

#Padding
Mel-spectrograms are padded along the time dimension.
Enables batch processing.

#Input Length Computation
Original time lengths are recorded.
Adjusted to account for time reduction caused by CNN pooling.

#Target Concatenation
All label sequences are concatenated into a single tensor.
Corresponding target lengths are stored.

Output of collate_fn
Padded Mel-spectrogram tensor-->(B,1,F,Tmax)

7. Model Architecture:
The proposed ASR model follows a CNN--BiLSTM--CTC architecture.
It is designed to first extract robust local acoustic features and then model long-range temporal dependencies.

7.1) CNN Feature Extractor:
The convolutional 2layers operates on 2D Mel-spectrograms and learns local time–frequency patterns.
Each convolutional block follows the sequence:
Conv2D->BatchNorm->ReLU->MaxPooling

i)First Convolutional Block:(1->32 channels) and kernal size(3*3) with padding 1.
The input to the first CNN layer is a Mel-spectrogram tensor of shape:(B,1,80,T) 
where:B->Batch size, 1->no. of channels,80->Mel frequency bins, T-> no. of time frames.
#Batch Normalization:
Batch normalization is applied channel-wise to stabilize the distribution of activations during training.
#ReLU Activation:
A Rectified Linear Unit (ReLU) non-linearity is applied element-wise:
#Max Pooling (2*2, Stride 2×1)
A max-pooling operation with kernel size 2*2 and stride 2*1 is applied:
Convolutional Block Structure
#Output shape:
(B,32,F/2,T) because we do stride by 2 at freq. domain

ii)Second Convolutional Block(32->64 channels) and kernal size(3*3) with padding 1.
##Batch Normalization and #ReLU Activation is same as first layer
#Max Pooling (2*2, Stride 2×1)
A second max-pooling operation with kernel size 2*2 and stride 2*2 is applied:
#Output shape:
(B,64,F/2,T/2) because we do stride by 2 at both domain

7.2) Bidirectional LSTM Encoder:
The reshaped CNN features(64*20) are passed to a 2-layer Bidirectional LSTM.
Hidden size: 256.
Bidirectional processing captures both past and future context.
Batch-first configuration is used.
The BiLSTM effectively models long-term temporal dependencies in speech signals.

7.3) Output Projection Layer
The BiLSTM output is passed through a fully connected linear layer that projects each time step to:
29 character classes(Zt)
These include:English letters (a–z)+Space+Apostrophe (')+CTC blank symbol
#Probability Estimation (Training Phase)
i)During training, the raw logits are converted into log-probabilities using a log-softmax function:
P(c|t)=log Softmax(Zt)
These log-probabilities are provided to the CTC loss function,
ii)During testing, the objective is to obtain a single character sequence from the frame-level predictions.
A greedy decoding strategy is employed:
At each time step, select the character with the maximum posterior probability:
C=argmax P(c∣t)

8. Training Objective:
The model is trained using CTC Loss, which allows sequence-to-sequence learning without explicit frame-level alignment.
CTC marginalizes over all valid alignments between input frames and target labels, enabling end-to-end training.

8.1) Optimization Strategy:
Optimizer: AdamW
Learning rate: 0.001
Learning rate scheduler: ReduceLROnPlateau
Gradient clipping: 5.0 //to prevent exploding gradients by limiting the maximum value.
Batch size: 8
Patience:3 //number of consecutive epochs the model is allowed to show no improvement in a monitored metric
Factor:0.5 //multiplicative value by which the learning rate is reduced when the monitored metric does not improve.

8.2) Training Monitoring and Loss Visualization
The training dynamics are visualized by plotting the CTC training loss against epochs, which provides insight into convergence behavior and optimization stability of the proposed model.

8.3) Training is done using Google_Colab GPU(T4).


9. cross-Validation Strategy:
To ensure a robust and unbiased evaluation of the proposed speech recognition model, 4-fold cross-validation is employed. This strategy allows the model’s performance to be assessed across multiple data splits, reducing dependence on a single train–validation partition.

The complete dataset is divided into four approximately equal-sized folds, each maintaining a similar distribution of speech samples and transcription lengths. During training, three folds are used for model training, while the remaining fold is reserved exclusively for validation. This process is repeated four times, such that each fold serves as the validation set exactly once.

For every fold, the model is trained independently from scratch, and performance is monitored using the validation loss computed with the CTC objective. The best-performing model for each fold is selected and saved based on the lowest validation loss achieved during training, ensuring that the retained model generalizes well to unseen data.

This cross-validation approach provides:
A more reliable estimate of model performance.
Reduced risk of overfitting to a specific split.
Improved confidence in the stability and generalization capability of the model.

10. Decoding Method
During inference(testing), the model outputs frame-level character probabilities.
A greedy CTC decoding strategy is applied:
Select the most probable character at each time step.
Remove repeated characters.
Remove blank tokens.

This produces the final predicted transcription.

11. Evaluation Metric:
Model performance is evaluated using Word Error Rate (WER):
WER=(S+D+I)/N
Where:
S = substitutions
D = deletions
I = insertions
N = total number of words in reference transcription

12. Experiment Results

The training process shows consistent convergence across all folds.
The use of CNN feature extraction, BiLSTM temporal modeling, and CTC loss enables the model to learn accurate character-level transcriptions from raw speech.

Qualitative testing on unseen audio samples demonstrates intelligible and coherent transcriptions.

13. Conclusion

This work presents a complete end-to-end Automatic Speech Recognition (ASR) system based on a Convolutional Neural Network (CNN) combined with Bidirectional Long Short-Term Memory (BiLSTM) networks and trained using Connectionist Temporal Classification (CTC) loss. The proposed architecture effectively bridges low-level acoustic modeling and high-level temporal sequence learning, enabling direct transcription of speech signals without requiring explicit frame-level alignments.

By operating on Mel-spectrogram features, the convolutional front-end learns robust local time–frequency patterns, while the BiLSTM layers capture long-range contextual dependencies essential for accurate character-level prediction. The use of CTC loss allows the model to internally learn optimal alignments between input frames and output symbols, simplifying the training pipeline and making the system suitable for real-world speech data.

The integration of data augmentation techniques, including speed perturbation and SpecAugment, improves model robustness and generalization to variations in speaking rate and acoustic conditions. Additionally, the use of k-fold cross-validation provides a reliable evaluation framework and reduces the risk of performance bias due to a single train–validation split.

________________________________________________________________________________________________

#just checked pretrained model
fine-tunes NVIDIA NeMo’s pretrained English Conformer CTC ASR model (stt_en_conformer_ctc_small) on a custom FLAC-based speech dataset with corresponding transcripts.

Workflow:
Load a pretrained Conformer-CTC ASR model from NeMo.

Prepare training and validation manifests (training.json, val.json) from FLAC audio and transcript files.

Configure training, validation, and optimizer settings.

Fine-tune the model using PyTorch Lightning.

Save the fine-tuned ASR model as a .nemo file.


