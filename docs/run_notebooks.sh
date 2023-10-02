for FILE in docs/(source|videos)/*.ipynb; do 
    papermill $FILE $FILE; 
    python docs/remove_metadata.py $FILE;
done
