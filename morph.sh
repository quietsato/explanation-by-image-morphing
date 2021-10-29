#!/bin/sh

DIFFMORPH_DIR="./DiffMorph"

IMAGE_DIR="../out/main/20211029155602_test0_morphing_idcvae_only"
MORPH_NUM=10
CONCAT_LIST="./morph/concat_list.txt"

EPOCHS=5

cd $DIFFMORPH_DIR

rm -f $CONCAT_LIST

for i in $(seq -f "%03g" $MORPH_NUM); do
    j=`echo "$i - 1" | bc | xargs printf "%03d"`
    MORPH_START="$IMAGE_DIR/$j.png"
    MORPH_END="$IMAGE_DIR/$i.png"
    echo $MORPH_START $MORPH_END

    python3 ./morph.py -s $MORPH_START -t $MORPH_END -e $EPOCHS
    mv ./morph/morph.mp4 ./morph/$i.mp4
    echo "file './$i.mp4'" >> $CONCAT_LIST
done

ffmpeg -safe 0 -f concat -i $CONCAT_LIST -c copy ./morph/concatenated.mp4
