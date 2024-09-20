# Description: Prepare the directory structure for the seven scene dataset

if [ ! -d "train_test_datasets" ]; then
  mkdir train_test_datasets
fi

if [ ! -d "train_test_datasets/gt_3Dmodels" ]; then
  mkdir train_test_datasets/gt_3Dmodels
fi

if [ ! -d "train_test_datasets/imgs_datasets" ]; then
  mkdir train_test_datasets/imgs_datasets
fi

TARGET_FOLDER="train_test_datasets/gt_3Dmodels"
OUTPUT_FILE="Cambridge.zip"
FILE_ID="19LRQ5j9I4YdrUykkoavcRTR6ekygU5iU"

# Download the file from Google Drive using gdown and save it in the target folder
gdown --id $FILE_ID -O $TARGET_FOLDER/$OUTPUT_FILE

# Unzip the downloaded file in the target folder
unzip $TARGET_FOLDER/$OUTPUT_FILE -d $TARGET_FOLDER

# Remove the zip file after extraction
rm $TARGET_FOLDER/$OUTPUT_FILE

echo "Download, extraction, and cleanup completed in $TARGET_FOLDER."

TARGET_FOLDER="train_test_datasets/imgs_datasets"
FILE_ID="1MZyLPu9Z7tKCeuM4DchseoX4STIhKyi7"

# Download the file from Google Drive using gdown and save it in the target folder
gdown --id $FILE_ID -O $TARGET_FOLDER/$OUTPUT_FILE

# Unzip the downloaded file in the target folder
unzip $TARGET_FOLDER/$OUTPUT_FILE -d $TARGET_FOLDER

# Remove the zip file after extraction
rm $TARGET_FOLDER/$OUTPUT_FILE

echo "Download, extraction, and cleanup completed in $TARGET_FOLDER."