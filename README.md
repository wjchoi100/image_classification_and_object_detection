# Capstone Project - Dog Breed Classification & Object Detection

## Objective
This capstone project encompasses two distinct yet interconnected tasks within the realm of computer vision:

**Task 1 - Image Classification**: The primary objective is to develop a multi-class image classification application capable of accurately predicting various dog breeds. This application provides breed predictions for both static images and real-time webcam captures. The project aims to achieve a high level of classification accuracy, with a target score of 85% or higher, emphasizing the model's proficiency in dog breed recognition under various conditions.

**Task 2 - Object Detection**: In parallel, the project delves into the field of object detection. Specifically, it aims to build an object detection model to identify a specific dog breed, Norwich Terriers, within images and videos. The success criteria for the object detection model include achieving a mean Average Precision at IoU threshold 0.50 (mAP50) score of 0.80 or higher.

---

## Datasets
- **Task 1 - Image Classification**: Labeled images of 93 different dog breeds.
  - (https://www.kaggle.com/datasets/kabilan03/dogbreedclassification)
  - 6,405 training images
  - 766 validation images
  - 891 test images.
- **Task 2 - Object Detection**: Images of Norwich Terriers annotated using the *Roboflow* application.
  - (https://roboflow.com/)
  - Originally 211 images of Norwich Terriers.
  - 507 final number of images after applying various data augmentation techniques
  - 444 training images
  - 42 validation images
  - 21 test images


## Data Preprocessing (Task 1 - Image Classification)
- Converted color of images to RGB format
- Changed size of images to 299 x 299 pixels
- Normalized pixel values to 0~1

  
## EDA (Task 1 - Image Classification)
- The average number of images per breed is around 69 images, with the minimum of 50 and maximum of 100 images
- The distribution and the mean pixel (color) values are remarkably similar among the training, validation, and test images, all having a mean value of approximately 112.
- The mean pixel (color) values by breed fall within the range of approximately 100 to 125.
- Red is the most predominant color channel, closely followed by Green and Blue. However, overall, all three colors are used quite evenly in images.


---

## Modeling

**Task 1 - Image Classification**:
 - Conducted multiple iterations of multi-class image classification model development with varying number of layers
 - Conducted multiple iterations of multi-class image classification model development using the following pre-trained models and custom 'top' layers
   - VGG16
   - ResNet50
   - Xception
 - Employed data augmentation techniques to apply various random transformations to the original images, generating new training samples.
 - Utilized the following regularization techniques:
   - L2 Regularization
   - Drop out
   - Early Stopping
 - Evaluated prediction results using various classification metrics and analyzed misclassified images
 - Chose the top-performing model and utilized it to create a Streamlit application capable of predicting dog breeds from both static images and live webcam feeds

**Task 2 - Object Detection**:
 - Applied the pre-trained YOLOv8s model to assess its ability to accurately detect objects such as dogs and people in both images and videos
 - Trained YOLOv8m and YOLOv8l models using custom images, annotated using the RoboFlow application, to identify Norwich Terriers in both images and videos, displaying bounding boxes and confidence scores
 - Evaluated the model's performance using mAP50 scores, a confusion matrix, and testing it on unseen images and videos.
 
---

## Conclusion

**Task 1 - Image Classification**:
Utilizing pre-trained models to build a custom image classifier resulted in significantly superior performance compared to constructing a model from scratch. Among the three well-known pre-trained models used for this task, Xception outperformed the others. VGG16 failed to converge, and while ResNet50 achieved substantial improvements over VGG16, it still showed signs of underfitting.

Summary of model performance trained using Xception:

- train accuracy: 0.87
- validation accuracy: 0.85
- test accuracy: 0.90
- precision: 0.91
- recall: 0.90
- f1 score: 0.90
- total number of parameters: ~21 million

The superior performance of Xception can likely be attributed to its unique architecture, which includes a depthwise separable convolutional layer, a departure from traditional CNN designs. This innovative approach significantly reduces the number of parameters, making Xception more memory and computationally efficient, helping to prevent overfitting and accelerate training. On the other hand, VGG16 is known for its deep architecture with numerous layers, which can hinder convergence due to vanishing gradients. As for ResNet50, its residual connections, while effective in mitigating vanishing gradients, may lead to underfitting when the dataset is not extensive enough, as was the case for this task.

**Task 2 - Object Detection**:
Utilizing YOLOv8 (You Only Look Once), the latest iteration of the YOLO algorithm, for detecting Norwich Terriers has yielded a model capable of effectively identifying them in both images and videos.

In the initial iteration, which was trained using YOLOv8m with images of Norwich Terriers and Beagles, the model failed to detect Norwich Terriers in video footage. However, the second model, trained using the YOLOv8l architecture exclusively on Norwich Terrier images, not only excelled at detecting Norwich Terriers in both images and videos but also successfully distinguished them from other dog breeds, achieving an impressive mAP50 score of 0.947. This enhanced model performance can be attributed to several key factors:

- Utilizing a more advanced and comprehensive model (YOLOv8l) compared to the initial iterations (YOLOv8m).
- Increasing the number of training epochs from 70 to 100.
- Expanding the dataset with a larger number of images.
- Applying more sophisticated data augmentation techniques.
- Focusing the training exclusively on Norwich Terriers without introducing other breeds, which allowed the model to learn more effectively.
