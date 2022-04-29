# python tools/demo.py image -n yolox_s -c yolox_s.pth --path inference/ --conf 0.1 --nms 0.5 --tsize 640 --save_result --device [cpu/gpu]
# python tools/demo.py image -f exps/yolox_x_mix_mot20_ch.py -c bytetrack_x_mot20.tar  --path inference/ --conf 0.5 --nms 0.5 --tsize 640 --save_result --device [cpu/gpu]
# python tools/demo.py image -f exps/example/custom/yolox_s.py -c YOLOX_outputs/yolox_s/latest_ckpt.pth  --path inference/ --conf 0.25 --nms 0.5 --tsize 640 --save_result --device [cpu/gpu] 
python tools/demo.py image -f exps/crowd.py -c latest_ckpt.pth  --path inference/ --conf 0.25 --nms 0.5 --tsize 640 --save_result --device [cpu/gpu] 
