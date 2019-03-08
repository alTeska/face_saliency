for i in {1..10}
do
   echo "Run $i time"
   python coco_test_mixin.py
   pkill maltab
   rm -rf model_caches/*
done
