for FILE in docs/source/*.ipynb; do 
    papermill $FILE $FILE; 
    python docs/remove_metadata.py $FILE;
done
