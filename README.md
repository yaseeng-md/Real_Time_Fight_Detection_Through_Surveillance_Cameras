# Real Time Fight Detection Through Surveillance Cameras

### Project Description
This project focuses on developing a system that detects fights in real-time using surveillance cameras. With the increasing need for public safety, fight detection systems can assist in identifying violent activities in real time, enabling swift responses. The system processes video sequences from surveillance footage, classifies them as "fight" or "no fight," and provides accurate predictions using deep learning models.

### Key Features:
1. Real-time processing of surveillance video streams.
2. Ability to detect fights in sequences of 8 frames.
3. Implemented using state-of-the-art models such as 3D CNN, 2D CNN + LSTM, and 3D CNN + LSTM.
4. Models have been trained and tested on sequences of frames resized to (150, 150, 3).

### Model Descriptions
1. 2D CNN + LSTM: This model extracts spatial features using 2D CNN layers and then passes these features to LSTM layers to capture temporal information across frames.

2. 3D CNN: This model utilizes 3D convolutional layers to capture both spatial and temporal features from the video sequences in a single pass.

3. 3D CNN + LSTM: A hybrid approach combining 3D CNN layers for initial spatial-temporal feature extraction, followed by LSTM layers to capture temporal dependencies.

## How to Use This Repository
Clone the repository using:
```bash
  git clone https://github.com/yaseeng-md/Real_Time_Fight_Detection_Through_Surveillance_Cameras.git
  cd Real_Time_Fight_Detection_Through_Surveillance_Cameras
```
Install the necessary dependencies from the requirements.txt file.
```bash
pip install -r requirements.txt
```
Run the provided Jupyter notebooks to preprocess the data, train models, and make predictions.

After downloading the repository, :
```bash
Fight Detection Project
├── .gitattributes
├── fight
│      ├── fight
│      ├── noFight
│      ├── LICENSE
│      ├── README
│      └── videos
├── 3D_CNN and 3D_CNN_LSTM_Model.ipynb          # Jupyter notebook implementing 3D CNN and 3D CNN + LSTM models
├── CNN_LSTM_Model.ipynb                        # Jupyter notebook implementing CNN + LSTM model
├── Prediction_module.ipynb                     # Jupyter notebook for running predictions on new data
├── Images                                      # Directory containing images used in the project
│   ├── Model Architectures
│   │   ├── 3D_CNN_LSTM_Model.png               # Architecture diagram for the 3D CNN + LSTM model
│   │   ├── 3D_CNN_Model.png                    # Architecture diagram for the 3D CNN model
│   │   └── cnn_lstm_model_arch.png             # Architecture diagram for the CNN + LSTM model
│   │
│   ├── Paper in between images
│   │   ├── frames_plot.png                     # Plot of frames used for model input
│   │   ├── noFight Prediction.png              # Example output showing no fight detected
│   │   ├── output.png                          # General model output image
│   │   └── Prediction.png                      # Prediction results for fight detection
│   │
│   └── Plots
│       ├── 3D_CNN_Accuracy_Plot.jpg            # Accuracy plot for 3D CNN model
│       ├── 3D_CNN_Loss_Plot.jpg                # Loss plot for 3D CNN model
│       ├── 3D_CNN_LSTM_Accuracy_plot.jpg       # Accuracy plot for 3D CNN + LSTM model
│       ├── 3D_CNN_LSTM_Loss_plot.jpg           # Loss plot for 3D CNN + LSTM model
│       ├── cnn_lstm_model_acc.jpg              # Accuracy plot for CNN + LSTM model
│       └── cnn_lstm_model_loss.jpg             # Loss plot for CNN + LSTM model
│
├── Model History                               # Model training history files
│   ├── 3D_CNN_LSTM_History.csv                 # Training history for 3D CNN + LSTM model
│   ├── cnn_lstm_model_history.pkl              # Pickled history for CNN + LSTM model
│   └── ThreeD_model_history.csv                # Training history for 3D CNN model
│
├── .ipynb_checkpoints                          # Jupyter notebook checkpoint files
│   └── ThreeD_model_history-checkpoint.csv
│
└── Saved_Models                                # Directory containing saved models
    ├── 3D_CNN_LSTM_Model.keras                 # Saved 3D CNN + LSTM model in Keras format
    ├── 3D_CNN_Model.keras                      # Saved 3D CNN model in Keras format
    ├── CNN_LSTM.h5                             # Saved CNN + LSTM model in .h5 format
    └── CNN_LSTM.keras                          # Saved CNN + LSTM model in Keras format
```


## The Results:
| Model                     | Training Accuracy        |Validation Accuracy|
|---------------------------|--------------------------|-------------------|
| 2D CNN + LSTM             | 0.99(99%)                | 0.632 (63%)       |
| 3D CNN                    | 1.0(100%)                | 0.784(78%)        |
| 3D CNN + LSTM             | 1.0(100%)                | 0.635(63%)        |

## The plots

### Training Accuracies
## 2D CNN + LSTM

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/yaseeng-md/Real_Time_Fight_Detection_Through_Surveillance_Cameras/blob/main/Images/Plots/cnn_lstm_model_acc.jpg" alt="Accuracy" width="400" height="300" style="margin: 10px;"/>
    <img src="https://github.com/yaseeng-md/Real_Time_Fight_Detection_Through_Surveillance_Cameras/blob/main/Images/Plots/cnn_lstm_model_loss.jpg" alt="Loss" width="400" height="300" style="margin: 10px;"/>
</div>

# 3D CNN
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/yaseeng-md/Real_Time_Fight_Detection_Through_Surveillance_Cameras/blob/main/Images/Plots/3D_CNN_Accuracy_Plot.jpg" alt="Accuracy" width="400" height="300" style="margin: 10px;"/>
    <img src="https://github.com/yaseeng-md/Real_Time_Fight_Detection_Through_Surveillance_Cameras/blob/main/Images/Plots/3D_CNN_Loss_Plot.jpg" alt="Loss" width="400" height="300" style="margin: 10px;"/>
</div>




## Authors

- [@Karthikkosuri](https://github.com/Karthikkosuri)
- [@yaseeng-md](https://github.com/yaseeng-md)
