import os
from IPython import display
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image, clear_output
from roboflow import Roboflow

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HOME = os.getcwd()
print(HOME)
clear_output()
ultralytics.checks()

rf = Roboflow(api_key="E52RkDukgBOXgdielAOV")
project = rf.workspace("newtask0801").project("recog0801")
version = project.version(1)
dataset = version.download("yolov8")

dataset_location = r"C:\Users\marsh\Desktop\recognition\yolo\board"

home_dir = os.path.expanduser("~")
os.chdir(home_dir)

# train
os.system(
    f"yolo task=detect mode=train model=yolov8s.pt data={dataset_location}/data.yaml epochs=120 imgsz=80 plots=True"
)
