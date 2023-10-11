import os,sys, subprocess
import yaml
from detection.utils.main_utils import read_yaml_file
from detection.logger import logging
from detection.exception import AppException
from detection.entity.config_entity import ModelTrainerConfig
from detection.entity.artifacts_entity import ModelTrainerArtifact



class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config


    

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            subprocess.call("unzip data.zip")
            subprocess.call("rm data.zip")

            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)


            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            batch_size = self.model_trainer_config.batch_size
            no_epochs = self.model_trainer_config.no_epochs
            weight_name = self.model_trainer_config.weight_name
            subprocess.call("cd yolov5/ && python train.py --img 416 --batch {} --epochs {} --data ../data.yaml --cfg ./models/custom_yolov5s.yaml --weights {} --name yolov5s_results  --cache".format(batch_size, no_epochs, weight_name))
            os.system("cp yolov5/runs/train/yolov5s_results/weights/best.pt yolov5/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp yolov5/runs/train/yolov5s_results/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")
           
            subprocess.call("rm -rf yolov5/runs")
            subprocess.call("rm -rf train")
            subprocess.call("rm -rf valid")
            subprocess.call("rm -rf data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise AppException(e, sys)
