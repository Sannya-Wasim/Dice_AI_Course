# **Assignment: Image Classification and Retrieval Using Deep Learning Models**

**Task 01: Implementing a Basic CNN Model from Scratch**

In this task, I implemented a Convolutional Neural Network (CNN) model using TensorFlow to classify images of digits into three classes: 0, 1, and 2. The dataset was split into training and validation sets, with 80% used for training and 20% for validation. Data augmentation techniques were applied to enhance the model's performance. The LeNet architecture was used as the base model. After training for 10 epochs, the model achieved a validation accuracy of [mention your accuracy here].


**Task 02: Fine-tuning a Pre-trained Model**

For this task, I fine-tuned the MobileNetV2 pre-trained model on the same dataset of digits (0, 1, and 2). Transfer learning techniques were applied to adapt the model to our specific dataset. After fine-tuning, the model achieved a validation accuracy of [mention your accuracy here]. This task aimed to demonstrate the application of transfer learning concepts.


**Task 03: Image Retrieval Using Pre-trained CNN Model**

In this task, I developed a Python program for image retrieval using a pre-trained, lightweight MobileNet model. Users can select a query image from the "query_images" folder, and the program retrieves the top 4 similar images from the "images_database" folder based on the Euclidean distance metric. The program is capable of handling various image formats such as JPG, PNG, and JPEG. Clear instructions on how to run the program are provided in the repository.


**How to Run the Code:**

1. **Task 01:** Run the `task_01_image_classification.py` script. Make sure TensorFlow and other dependencies are installed. Adjust the number of epochs and model architecture in the script if needed.

2. **Task 02:** Run the `task_02_fine_tuning.py` script. Ensure you have a pre-trained MobileNetV2 model and the required dataset. Modify the script to specify the path to the pre-trained model and adjust other hyperparameters as necessary.

3. **Task 03:** Execute the `image_retrieval.py` script. Make sure to have the pre-trained cifar_MobileNet model file and the "query_images" and "images_database" folders in the same directory. Follow the prompts to select a query image and observe the program retrieving similar images based on the MobileNet features and Euclidean distance metric.

**Sample Query Images:**

Include some sample query images in the "query_images" folder to demonstrate the functionality of the image retrieval system.

**Repository Structure:**

- `Assignment_02_Task_01`: Python script for Task 01.
- `Assignment_02_Task_02`: Python script for Task 02.
- `Assignment_02_Task_03`: Python script for Task 03.
- `query_images/`: Folder containing sample query images.
- `images_database/`: Folder containing images for retrieval.
- `README.md`: Detailed explanation of tasks, code, and instructions.

**Dependencies:**

- Python 3.x
- TensorFlow
- NumPy
- Pillow (PIL)

**Important Notes:**

- Ensure the correct paths to the dataset, pre-trained models, and image folders are specified in the scripts.
- Verify that the necessary permissions are set to read/write images in the query and database folders.

**Conclusion:**

This assignment demonstrated the implementation of CNN models for image classification and retrieval tasks. From building a basic CNN model from scratch to fine-tuning a pre-trained model, and finally, creating an image retrieval system, various deep learning concepts were applied. These tasks provided hands-on experience in both fundamental and advanced aspects of deep learning, reinforcing the knowledge acquired during the course.

**Author:**

[Sannya Wasim]
[sannya.wasim01@gmail.com]

**Date:**

[25 October'23]
