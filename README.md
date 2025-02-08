# 🚀 DeepFakeDetection  

DeepFakeDetection is a deep learning-based project that classifies images as **real or AI-generated**. It uses a **CNN model** trained on a dataset containing real and AI-generated images to predict the authenticity of unseen images.  

## 📂 Dataset Structure  
The dataset consists of:  
- **train/** (Contains labeled images)  
  - `real/` - Contains real images  
  - `fake/` - Contains AI-generated images  
- **test/** (Contains unlabeled images to be predicted)
## 📂 Dataset  
The dataset is too large to be included in this repository. Please download it manually from Kaggle:  

🔗 **[DeepFakeDetection Dataset on Kaggle](https://www.kaggle.com/competitions/vista-25)**  

After downloading, extract the dataset inside the project folder:


## 🎯 Objective  
The model predicts whether an image is **real (1.0) or fake (0.0)** and generates a CSV file with the format:  
```
image_id,label  
0,1  
1,0  
2,0
```
...  

## 🛠️ Installation & Usage  
### Clone the Repository  
Clone the repository from GitHub and navigate to the project folder.  

### Install Dependencies  
Ensure you have Python installed, then install the required dependencies using the `requirements.txt` file.  

### Train the Model  
Run the training script to train the deep learning model on the dataset.  

### Make Predictions  
Use the trained model to make predictions on test images and generate the output CSV file.  

## 📊 Model Details  
- **Algorithm:** Convolutional Neural Networks (CNN)  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Output Format:** Probability score (0-1 scale, where 1 = real, 0 = fake)  

## 📄 License  
This project is licensed under the **MIT License**. See the LICENSE file for details.  

## 🤝 Contributing  
Contributions are welcome! Feel free to open issues or submit pull requests.  

## ✨ Author  
Developed by **Arunkarthick.K**  
GitHub: [https://github.com/Arunkarthi123](https://github.com/Arunkarthi123)  
