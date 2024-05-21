# Tensor Fusion Network (TFN) for Multimodal Sentiment Analysis

This repository contains the implementation of the Tensor Fusion Network (TFN) for multimodal sentiment analysis using the CMU-MOSI dataset. The TFN architecture incorporates language, visual, and acoustic modalities to predict sentiment intensity.

## Dataset: CMU-MOSI

The CMU-MOSI dataset is an annotated dataset of video opinions from YouTube movie reviews. It includes sentiment annotations on a seven-step Likert scale from very negative to very positive. The dataset comprises 2,199 opinion utterances from 93 distinct speakers, with an average length of 4.2 seconds per video.

### Dataset Features

- **Language Modality**: Uses GloVe word vectors for spoken words.
- **Visual Modality**: Extracts facial expressions and action units using the FACET framework and OpenFace.
- **Acoustic Modality**: Extracts acoustic features using the COVAREP framework.

### Sentiment Prediction Tasks

1. **Binary Sentiment Classification**
2. **Five-Class Sentiment Classification**
3. **Sentiment Regression**

## Tensor Fusion Network (TFN)

TFN consists of three main components:

1. **Modality Embedding Subnetworks**: Extracts features from language, visual, and acoustic modalities.
2. **Tensor Fusion Layer**: Explicitly models unimodal, bimodal, and trimodal interactions.
3. **Sentiment Inference Subnetwork**: Performs sentiment inference based on the fused multimodal tensor.

### Modality Embedding Subnetworks

- **Language Embedding Subnetwork**: Uses LSTM to learn time-dependent representations of spoken words.
- **Visual Embedding Subnetwork**: Uses a deep neural network to process visual features extracted from facial expressions.
- **Acoustic Embedding Subnetwork**: Uses a deep neural network to process acoustic features extracted from audio signals.

### Tensor Fusion Layer

The Tensor Fusion Layer models the interactions between different modalities using a three-fold Cartesian product, generating a multimodal tensor that captures unimodal, bimodal, and trimodal dynamics.

### Sentiment Inference Subnetwork

A fully connected deep neural network that takes the multimodal tensor as input and performs sentiment classification or regression.

## Experiments

Three sets of experiments were conducted:

1. **Multimodal Sentiment Analysis**: Compared TFN with state-of-the-art multimodal sentiment analysis models.
2. **Tensor Fusion Evaluation**: Analyzed the importance of subtensors and the impact of each modality.
3. **Modality Embedding Subnetworks Evaluation**: Compared TFN's modality-specific networks with state-of-the-art unimodal sentiment analysis models.

## Results

TFN outperformed state-of-the-art approaches in binary sentiment classification, five-class sentiment classification, and sentiment regression. The ablation study showed the importance of modeling trimodal dynamics for improved performance.

## How to Use

### Prerequisites

- Python 3.x
- TensorFlow or PyTorch (depending on the implementation)
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TFN-multimodal-sentiment.git
   cd TFN-multimodal-sentiment
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

1. Download the CMU-MOSI dataset from the official source.
2. Extract the dataset and place it in the `data` directory.

### Training the Model

1. Preprocess the dataset:
   ```bash
   python preprocess.py --data_dir data/CMU-MOSI
   ```

2. Train the TFN model:
   ```bash
   python train.py --config configs/tfn_config.json
   ```

### Evaluation

Evaluate the trained model on the test set:
```bash
python evaluate.py --model_dir models/tfn --data_dir data/CMU-MOSI
```

### Configuration

Modify the configuration file `configs/tfn_config.json` to change hyperparameters, model settings, and dataset paths.

## Citation

If you use this code or dataset in your research, please cite the original paper:

```bibtex
@inproceedings{zadeh2016mosi,
  title={Multimodal Sentiment Intensity Analysis in Videos: Facial Gestures and Verbal Messages},
  author={Zadeh, Amir and Chen, Minghai and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe},
  booktitle={IEEE Intelligent Systems},
  year={2016}
}
```

## License

This project is licensed under the MIT License.

---

Feel free to open an issue if you have any questions or need further assistance. Happy researching!
