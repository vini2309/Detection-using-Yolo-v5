import os,sys,shutil
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
            #os.system("unzip data.zip")
            shutil.unpack_archive('data.zip')
            #os.system("rm data.zip")
            os.remove('data.zip')

            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)


            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            os.system(f"cd yolov5/ && python train.py --img 416 --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --data ../data.yaml --cfg ./models/custom_yolov5s.yaml --weights {self.model_trainer_config.weight_name} --name yolov5s_results  --cache")
            #os.system("cp yolov5/runs/train/yolov5s_results/weights/best.pt yolov5/")
            shutil.copy('yolov5/runs/train/yolov5s_results/weights/best.pt', 'yolov5/') 
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy("yolov5/runs/train/yolov5s_results/weights/best.pt", "artifacts/model_trainer/")
           
            #os.system("rm -rf yolov5/runs")
            #os.system("rm -rf train")
            #os.system("rm -rf valid")
            #os.system("rm -rf data.yaml")
            shutil.rmtree('yolov5/runs')
            shutil.rmtree('train')
            shutil.rmtree('valid')
            os.remove('data.yaml')

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise AppException(e, sys)
