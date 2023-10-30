for FILE in docs/*/*.ipynb; do
    echo "Running $FILE"
    DIR=$(dirname $FILE)
    BASE=$(basename $FILE)
    mv $FILE .

    jupyter nbconvert --to notebook $BASE --execute --inplace
    python docs/remove_metadata.py $BASE;

    mv $BASE $DIR
done
