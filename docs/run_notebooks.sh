for FILE in docs/*/*.ipynb; do 
    papermill $FILE $FILE; 
    python docs/remove_metadata.py $FILE;
done
