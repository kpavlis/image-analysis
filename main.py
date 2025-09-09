import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import copy
import math
import numpy as np
import itertools
import random

# Definitions of transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for use with ResNet
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])


image_folder = datasets.ImageFolder(root='your_images_folder', transform=transform)
filenames = [image_folder.imgs[i][0] for i in range(len(image_folder))]# List of file paths

limited_image_folder = torch.utils.data.Subset(image_folder, range(300))
limited_filenames = filenames[:300]
print("Mapping image numbers and file-paths:")
for i in range(300):
    print(i, " ", limited_filenames[i])

# Extracting images and labels
images, labels = zip(*[(data[0], data[1]) for data in limited_image_folder])

# Convert the images to tensors
images_tensor = torch.stack(images)

# Loading the pre-trained ResNet50 model
resnet_model = torchvision.models.resnet50(pretrained=True)

features = []

def hook(module, input, output):
    flattened = output.view(output.size(0), -1)
    features.append(flattened.clone().detach())


hook_handle = resnet_model.avgpool.register_forward_hook(hook)


with torch.no_grad():
    resnet_model.eval()
    _ = resnet_model(images_tensor)

hook_handle.remove()

extracted_features = torch.cat(features, dim=0)

distances = [[0 for _ in range(len(extracted_features))] for _ in range(len(extracted_features))]

for i,feature_tensor_v1 in enumerate(extracted_features):
    for j,feature_tensor_v2 in enumerate(extracted_features):
        distances[i][j] = [j, torch.norm(feature_tensor_v1 - feature_tensor_v2)]

distance_normalized = copy.deepcopy(distances)

for i in range(len(extracted_features)):
    distances[i].sort(key=lambda x : x[1])




def get_neighbors(extracted_features, distances):
    neighbors = [[0 for _ in range(4)] for _ in range(len(extracted_features))]

    for i in range(len(extracted_features)):
        for j in range(4):
            neighbors[i][j] = distances[i][j][0]

    return neighbors



counter = 0 # Counter for the number of epochs
def repeater(extracted_features, distances, distance_normalized, counter):

    neighbors = get_neighbors(extracted_features, distances)

    # A) --- Rank Normalization ---

    L = len(extracted_features)

    for i in range(L):
        for j in range(L):
            positions_v1 = [k for k, sublist in enumerate(distances[i]) if sublist[0] == j]
            positions_v2 = [l for l, sublist in enumerate(distances[j]) if sublist[0] == i]
            distance_normalized[i][j][1] = 2 * L - (positions_v1[0] + positions_v2[0])

    for i in range(len(extracted_features)):
        distance_normalized[i].sort(key=lambda x: x[1], reverse=True)




    # B) --- Hypergraph Construction ---

    Hb = [[0 for _ in range(len(extracted_features))] for _ in range(len(extracted_features))]
    neighbors_normalized = get_neighbors(extracted_features, distance_normalized)

    for i in range(len(extracted_features)):
        for j in range(4):
            Hb[i][neighbors_normalized[i][j]] = 1





    H = copy.deepcopy(Hb)

    for i in range(L):
        for j in range(L):
            if H[i][j] == 1:
                positions_v1 = [k for k, sublist in enumerate(distance_normalized[i]) if sublist[0] == j]

                H[i][j] = 1 - math.log(positions_v1[0] + 1, 5)


    w = []
    for i in range(L):
        w.append(sum(H[i]))






    # C) --- Hyperedge Similarities ---

    H_array = np.array(H)
    H_array_transpose = H_array.T

    Sh = np.dot(H_array, H_array_transpose)
    Sv = np.dot(H_array_transpose, H_array)

    S = Sh * Sv

    # D) --- Cartesian Product of Hyperedge Elements ---

    C = [[0 for _ in range(len(extracted_features))] for _ in range(len(extracted_features))]

    for i in range(L):
        for j in range(L):
            p = 0
            for k in range(L):
                p += w[k] * H[k][i] * H[k][j]
            C[i][j] = p
    # E) --- Hypergraph-Based Similarity ---
    W_matrix = C * S

    mylist = W_matrix.tolist()

    W_matrix_modified_sorted = [[0 for _ in range(len(extracted_features))] for _ in range(len(extracted_features))]

    for i in range(L):
        for j in range(L):
            W_matrix_modified_sorted[i][j] = [j, mylist[i][j]]

    W_matrix_modified = copy.deepcopy(W_matrix_modified_sorted)

    for i in range(L):
        W_matrix_modified_sorted[i].sort(key=lambda x: x[1], reverse=True)

    print(f" ----------- The repeater ran for {counter} times... : -----------")

    return W_matrix_modified_sorted, W_matrix_modified


result_sorted, result = repeater(extracted_features, distances, distance_normalized, 1)

for i in range(6):
    result_sorted, result = repeater(extracted_features, result_sorted, result, i+2)

times = random.randint(3,8)


target_images = []
target_images_numbers = []
for i in range(times):
    target_images_number = random.randint(0, len(result)-1)
    print("For Target Image the " + str(target_images_number) + " and related images: ")
    target_images.append(result_sorted[target_images_number])
    target_images_numbers.append(target_images_number)
    for i in range(4):
        print(result_sorted[target_images_number][i])
    print("\n")



print("\n")
print("The scores of the target images are: ")
total_avg_sum = 0
for i in range(len(target_images)):
    sum = 0
    for j in range(4):
        sum+=target_images[i][j][1]
    avg_sum=sum/4
    print("For Target Image the " + str(target_images_numbers[i]) + ": " + str(avg_sum))
    total_avg_sum+=avg_sum
print("Total score is: "+str(total_avg_sum/len(target_images)))
