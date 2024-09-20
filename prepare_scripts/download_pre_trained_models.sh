OUTPUT_FILE="logs.zip"
FILE_ID="1iH8PfqgPPQod0q_I8T_ZSO_mSj5XRUuO"

# Download the file from Google Drive using gdown and save it in the target folder
gdown --id $FILE_ID -O $OUTPUT_FILE

# Unzip the downloaded file in the target folder
unzip $OUTPUT_FILE

# Remove the zip file after extraction
rm $OUTPUT_FILE

echo "Download, extraction, and cleanup completed."