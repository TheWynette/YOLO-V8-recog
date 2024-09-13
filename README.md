# YOLO-V8-recog
YOLO V8 recog
YOLO-V8 Real-Time Digit Recognition is a project aimed at recognizing digits on semicircular colored paper in real-time using the YOLO-V8 model. The project is designed to run on a Raspberry Pi installed on a model aircraft, allowing it to recognize digits and output corresponding addresses in real-time.

Table of Contents

[Introduction](#introduction)
[Installation](#installation)
[Usage](#usage)
[Configuration](#configuration)
[Examples](#examples)
[Contributing](#contributing)
[License](#license)
[Acknowledgements](#acknowledgements)

Introduction
This project uses the YOLO-V8 model for automatic digit recognition from images of semicircular colored paper. The digits can be recognized even when the paper is moving, which is ideal for real-time applications like identifying addresses during flight.

Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/TheWynette/YOLO-V8-recog.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd YOLO-V8-recog
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the model and dataset**: Follow the instructions in the `model_training.md` file to train and prepare the YOLO-V8 model for digit recognition.
2. **Run the digit recognition script**:
   ```bash
   python detect.py --input path/to/image.jpg --output path/to/output.jpg
   ```
   Use the `--input` flag to specify the path to the input image and the `--output` flag to specify the path for saving the output image.

## Configuration
Remote Repository Setup
1. **Check your remote repository URL**:
   ```bash
   git remote -v
   ```
2. **Set or update the remote URL**:
   ```bash
   git remote set-url origin git@github.com:TheWynette/YOLO-V8-recog.git
   ```
3. Push changes to the remote repository:
   ```bash
   git push -u origin main
   ```

## Handling Branches
If you encounter issues with branches or need to set up a new branch:
1. **Create and switch to a new branch**:
   ```bash
   git checkout -b main
   ```
2. **Push the branch to the remote repository**:
   ```bash
   git push -u origin main
   ```

## Examples
Here are some examples of what the digit recognition output looks like:

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Make sure to follow the contribution guidelines in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Special thanks to the YOLO community for their contributions to the object detection field.
Thanks to **the great myself** for my assistance with integrating YOLO-V8 into this project.

