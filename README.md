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

After downloading the datasets, you may extract them under `data/real` and `data/fake` respectively. In the end, the `data` directory should look like this:


## The Results:
| Model                     | Training Accuracy        |Validation Accuracy|
|---------------------------|--------------------------|-------------------|
| 2D CNN + LSTM             | 0.99(99%)                | 0.632 (63%)       |
| 3D CNN                    | 1.0(100%)                | 0.784(78%)        |
| 3D CNN + LSTM             | 1.0(100%)                | 0.635(63%)        |

## The plots





## Authors

- [@Karthikkosuri](https://www.github.com/octokatherine)
- [@yaseeng-md]("https://github.com/yaseeng-md")
