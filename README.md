# 🖼️ Image Analysis using Hypergraph-Based Manifold Ranking

**The program implements an advanced image retrieval system based on the methodology described in the paper *"[Multimedia Retrieval Through Unsupervised Hypergraph-Based Manifold Ranking](https://ieeexplore.ieee.org/document/8733193)"*. The goal is to extract features from images and compute similarity scores to identify the most visually related images within a dataset.**

> ℹ️ This project is not open source and does not grant any usage rights.
> For usage terms and legal information, see [Code Ownership & Usage Terms](#-code-ownership--usage-terms).

## 🚀 Overview

The system processes a given dataset of 300 images and performs the following steps:

1. 🧠 **Feature Extraction.**
   Images are transformed into tensors using PyTorch and normalized for use with a pre-trained ResNet50 model. The model's flattening layer is used to extract feature vectors for each image.

3. 📏 **Distance Calculation.**
   Euclidean distances are computed between all image pairs to quantify dissimilarity. These distances are then used to rank images by similarity.

4. 🥇 **Initial Ranking.**
   For each image, the top L most similar images are selected to form a ranking list. This reduces computational cost and identifies nearest neighbors.
   
6. 🔁 **Manifold Ranking.** 
   A hypergraph-based manifold ranking algorithm refines similarity scores through multiple iterations. This includes:
   - Rank normalization
   - Hypergraph construction
   - Hyperedge similarity computation
   - Cartesian product of hyperedge elements
   - Final similarity matrix generation

7. 🧪 **Evaluation.** 
   A subset of four target images is randomly selected from the dataset, and for each one, the top 4 visually similar matches are retrieved. The system then calculates the average similarity score across these results to evaluate retrieval accuracy.

## 🧠 Technologies Used

- Python
- Pytorch & torchvision (ResNet50)
- NumPy

## 🎯 Purpose

This project was created to explore how we can retrieve visually similar images from a dataset without using labeled data. By combining deep feature extraction with a hypergraph-based ranking algorithm, the system identifies and ranks images based on visual similarity. **It is developed solely for academic and research purposes.**

## 🧰 Prerequisites

Before running the application, make sure your environment is properly configured.

### Python Version

- Python **3.9** is recommended

### Required Libraries

- torch (**2.8.0**)
- torchvision (**0.23.0**)

## 🧪 How to Run

1. **Clone the repository (or download and decompress the ZIP file)**
   ```bash
   git clone https://github.com/kpavlis/image-analysis.git
   cd image-analysis

2. **Specify** the path to your image dataset (containing 300 `.jpg` images from at least 5 different categories) by editing line 19 in `main.py`:
   ```python
   image_folder = datasets.ImageFolder(root='your_images_folder', transform=transform)

3. **Confirm** that you have installed the required libraries

4. **Run** `main.py`

5. **View the output rankings and similarity scores** in the console

## 🛠️ Running a Demo

Below is a manually created visualization that demonstrates how the system retrieves similar images from a dataset.

Each row shows:
- 🎯 A target image on the left
- 🥇 The top 4 most similar images retrieved by the system on the right


<img width="2050" height="1116" alt="Ima_Ana_1" src="https://github.com/user-attachments/assets/2bc70966-69cf-489c-ab9c-f21ea4b79af3" />


### 📌 Notes

The above collage was manually created to illustrate the retrieval results.  
The application itself does **not generate graphical output** — it returns similarity scores and image rankings via the console.

> 📁 **This run was performed using a dataset of 300 images**, covering various categories such as pyramids, animals, people, and winter sports.

# 🔒 Code Ownership & Usage Terms

This project was created and maintained by:

- Konstantinos Pavlis (@kpavlis)
- Theofanis Tzoumakas (@theofanistzoumakas)
- Michael-Panagiotis Kapetanios (@KapetaniosMP)

🚫 **Unauthorized use is strictly prohibited.**  
No part of this codebase may be copied, reproduced, modified, distributed, or used in any form without **explicit written permission** from the owners.

Any attempt to use, republish, or incorporate this code into other projects—whether commercial or non-commercial—without prior consent may result in legal action.

For licensing inquiries or collaboration requests, please contact via email: konstantinos1125 _at_ gmail.com .

© 2025 Konstantinos Pavlis, Theofanis Tzoumakas, Michael-Panagiotis Kapetanios. All rights reserved.

