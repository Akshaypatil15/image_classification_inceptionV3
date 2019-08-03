# image_classification_inceptionV3
We are going to use inception v3 for mobile manufacture image based classification.

## Generating Record files
Here we have sample record files in data directory, if you have your own data is should be splitted in train-val folder.
For generating your own record files, run following command.

$ TRAIN_DIR=PATH_TO_YOUR_TRAIN_FOLDER
$ VALIDATION_DIR=PATH_TO_YOUR_VALIDATION_FOLDER
$ OUTPUT_DIRECTORY=PATH_TO_YOUR_OUTPUT_FOLDER
$ LABELS_FILE=TXT_FILE_PATH

$ python src/build_image_data.py \
	--train_directory=$TRAIN_DIR \
	--validation_directory=$VALIDATION_DIR \
	--output_directory=$OUTPUT_DIRECTORY \
	--labels_file=$LABELS_FILE

## Download Pre-Trained Inception-v3 checkpoint

$ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

## Train Model

$ TRAIN_DIR=PATH_TO_YOUR_DATA_FOLDER
$ CHECKPOINT_PATH=PATH_TO_PRETRAINED_INCEPTION_MODEL_FOLDER/inception_v3.ckpt
$ TRAINED_MODEL_DIR=PATH_TO_SAVE_TRAINED_MODEL_DIR

$ python src/slim/train_image_classifier.py \
	--train_dir=$TRAINED_MODEL_DIR \
	--dataset_dir=$TRAIN_DIR \
	--dataset_name=cell_phone_data \
	--dataset_split_name=train \
	--model_name=inception_v3 \
	--batch_size=32 \
	--checkpoint_path=$CHECKPOINT_PATH \
	--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
	--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
