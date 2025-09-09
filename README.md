# 🖼️ Image Analysis via Hypergraph-Based Manifold Ranking

This program implements an advanced image retrieval system based on the methodology described in the paper *"Multimedia Retrieval through Unsupervised Hypergraph-based Manifold Ranking"*. The goal is to extract features from images and compute similarity scores to identify the most visually related images within a dataset.

> ℹ️ This project is not open source and does not grant any usage rights.
> For usage terms and legal information, see [Code Ownership & Usage Terms](#-code-ownership--usage-terms).

## 🚀 Overview

The system processes a given dataset of images and performs the following steps:

1. 🧠 **Feature Extraction** 
   Images are transformed into tensors using PyTorch and normalized for use with a pre-trained ResNet50 model. The model's flattening layer is used to extract feature vectors for each image.

2. 📏 **Distance Calculation**
   Euclidean distances are computed between all image pairs to quantify dissimilarity. These distances are then used to rank images by similarity.

3. 🥇 **Initial Ranking**  
   For each image, the top L most similar images are selected to form a ranking list. This reduces computational cost and identifies nearest neighbors.

4. 🔁 **Manifold Ranking** 
   A hypergraph-based manifold ranking algorithm refines similarity scores through multiple iterations. This includes:
   - Rank normalization
   - Hypergraph construction
   - Hyperedge similarity computation
   - Cartesian product of hyperedge elements
   - Final similarity matrix generation

5. 🧪 **Evaluation** 
   A subset of target images is randomly selected, and the top 4 matches are retrieved. The system calculates the average similarity score to evaluate retrieval accuracy.

## 📊 Dataset

The dataset includes 300 images from various categories:
- Egyptian pyramids
- Cats
- Dogs
- People in activities
- Bicycles
- Vehicles

## 📷 Output

The application outputs visual comparisons between target images and their top matches. Each result includes:
- Target image
- Top 4 most similar images
- Similarity scores

Example categories retrieved include:
- Pyramids with different angles and environments
- Snowboarding scenes matched with similar winter sports images

## 🧠 Technologies Used

- Python & PyTorch
- torchvision (ResNet50)
- NumPy
- COCO image dataset

## 📈 Accuracy Metric

The system computes the average similarity score across all target images using the formula:

**Mean Similarity Score** =  
$$\frac{1}{n} \sum_{i=1}^{n} \text{Similarity}(i)$$  
Where *n* is the number of top matches per target image.

Higher scores indicate better retrieval performance.


## 🧪 How to Run

1. Place your images in `data/coco_images/`
2. Run `main_prerelease.py`
3. View the output rankings and similarity scores in the console and image collage

## 📌 Notes

- The ranking process is repeated 7 times to refine similarity scores.
- The final output includes both numerical scores and visual comparisons.

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

