#!/bin/bash

# Check if pysen is installed
if ! command -v pysen &> /dev/null
then
    echo "Error: pysen is not installed."
    exit 1
fi

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null
then
    echo "Error: jupyter is not installed."
    exit 1
fi

# Directory where the .ipynb files are
DIR="$1"

# If no directory is provided, exit the script
if [ -z "$DIR" ]
then
    echo "Error: No directory provided."
    exit 1
fi

# Find all .ipynb files in the directory and its subdirectories
find "$DIR" -name "*.ipynb" -type f | while read FILE
do
    # Convert the .ipynb file to .py
    jupyter nbconvert --to python "$FILE"
    
    # Check if the conversion was successful
    if [ $? -eq 0 ]
    then
        # If the conversion was successful, remove the original .ipynb file
        rm "$FILE"

        # Remove the .ipynb file extension from the file name
        PY_FILE=${FILE%.ipynb}.py
        
        # Remove lines starting with '#' from the .py file
        sed -i '/^#/d' "$PY_FILE"

        # Run pysen format command on the .py file
        pysen run_files format "$PY_FILE"
    else
        # If the conversion failed, print an error message
        echo "Conversion of $FILE failed."
    fi
done
