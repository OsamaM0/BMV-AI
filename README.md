# Be My Voice (BMV)
## Cutting-Edge Video ArSL Recognition

<p align="center">
  <img src="https://github.com/OsamaM0/BMV-AI/blob/main/image/Rectangle%201042.svg" alt="ArSL Recognition">
</p>

## Overview 📹

This repository contains state-of-the-art models and techniques for Arabic Sign Language (ArSL) recognition from video inputs. Our goal is to leverage deep learning and computer vision technologies to build accurate and efficient models capable of interpreting ArSL in real-time.

## Features ✨

- **End-to-End ArSL Recognition**: From raw video inputs to ArSL word predictions.
- **Real-Time Inference**: Optimized models for low latency, suitable for real-world applications.
- **Custom Deep Learning Models**: Novel convolutional and transformer-based architectures specifically designed for ArSL recognition.
- **Data Augmentation**: Extensive data augmentation techniques to handle variations in lighting, background, and signer differences.
- **Custom Datasets**: Incorporates multiple ArSL datasets, including custom-built datasets for specialized gestures.
- **Visualization Tools**: Tools for visualizing model predictions, keypoints, and attention maps.

## Table of Contents 📚

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation 💻

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/BMV-AI.git
cd ArSL-Recognition
pip install -r requirements.txt
```

## Usage 🚀

### Training and Visualization
To train a model from scratch, execute:

For Training and Visualization Model You can Ask Me

## Datasets 📦

This project builds upon our collected ArSL dataset:

- **Custom Dataset**: We created and use our custom datasets with the aid of the Faculty of Disabilities and Rehabilitation Sciences.

## Models 🧠

The repository includes a custom model architecture optimized for ArSL recognition:

### **Custom Convolutional and Transformer-based Model**

Our model leverages the strengths of both Convolutional Neural Networks (CNNs) and Transformer-based architectures to efficiently recognize ArSL from video sequences.

- **Efficient Channel Attention (ECA)**: Enhances feature representations by recalibrating the importance of different channels in the input tensor, improving the model's ability to focus on significant features.
- **Causal Dilated Depthwise Convolution (CausalDWConv1D)**: Captures temporal dependencies in sequential data, ensuring that future information is not used in the prediction of the current step.
- **Conv1D Blocks**: Efficiently process temporal sequences by combining depthwise convolutions with attention mechanisms, expanding and contracting the feature space.
- **Multi-Head Self-Attention**: Implements self-attention to dynamically weigh the importance of different parts of the input sequence, capturing long-range dependencies.
- **Transformer Blocks**: Further refine the sequential data by stacking multi-head self-attention layers and fully connected layers, with residual connections and layer normalization for stability.

Each component is designed and fine-tuned specifically for ArSL data to maximize accuracy and efficiency in real-time applications.

## Training 🎓

The training process is highly configurable:

- **Config Files**: Use YAML configuration files for flexible experiment setups.
- **Hyperparameter Tuning**: Easily adjust learning rates, batch sizes, and other hyperparameters.
- **Custom Architectures**: Train our state-of-the-art custom models specifically designed for ArSL.
- **Multi-GPU Support**: Train models using multiple GPUs for faster results.

## Evaluation 📊

Evaluate the performance of the trained models:

```bash
python evaluate.py --dataset_path path/to/dataset --model_path path/to/model.pth
```

- **Metrics**: Accuracy, precision, recall, F1-score, and more.
- **Confusion Matrix**: Generate a confusion matrix to analyze model performance.

## Issues 🤝

**Stay connected**: Feel free to [open an issue](https://github.com/OsamaM0/BMV-AI/issues) if you encounter any problems or have questions.

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact 📱

<p align="center">
  <a href="https://www.facebook.com/profile.php?id=100010073048538&mibextid=ZbWKwL"><img src="https://img.icons8.com/color/48/000000/facebook.png" alt="Facebook"/></a>
  <a href="https://t.me/Osama_Mo7"><img src="https://img.icons8.com/color/48/000000/telegram-app.png" alt="Telegram"/></a>
  <a href="https://www.linkedin.com/in/osama-mohammed-456502205"><img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn"/></a>
</p>
