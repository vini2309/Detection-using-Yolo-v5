import subprocess

batch_size = 16
no_epochs = 1
weight_name = "yolov5s.pt"
subprocess.call("cd yolov5/ && python train.py --img 416 --batch {} --epochs {} --data ../data.yaml --cfg ./models/custom_yolov5s.yaml --weights {} --name yolov5s_results  --cache".format(batch_size, no_epochs, weight_name))