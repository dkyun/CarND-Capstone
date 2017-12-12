# Source for tf object detection API
git clone https://github.com/tensorflow/models.git 

# Necessary in order to execute train.py
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 

# Source for detection models
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

# Example command to train the network

python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=rfcn_resnet101_coco.config

# Example command to generate a frozen_inference_graph.pb. 

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./rfcn_resnet101_coco.config --trained_checkpoint_prefix ./models/train/model.ckpt-5000 --output_directory ./fine_tuned_model
