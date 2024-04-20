#python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=GoogLeNet --target_class=753 --defense='_FRU'
#python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=ResNet152
#python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=VGG16
#python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=GoogLeNet --target_class=753 --defense='_FRU'
#python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=ResNet152 --defense='_FRU'

for TARGET_CLASS in {573,807,541,240,475,753,762,505}
do
    python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=GoogLeNet --target_class=$TARGET_CLASS
    python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=GoogLeNet --target_class=$TARGET_CLASS --defense='_FRU'
done