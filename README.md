Build and run demo on device side:
- apt-get update
- apt install libcairo2-dev
- apt install cmake
- cmake -DBUILD_AARCH64=1 ..
- make -j4

After build successfully:
- Push all libs under libs/opencv-4.9.0/aarch64/lib/ to /usr/lib/
- Push all libs under libs/SNPE2.19/aarch64-ubuntu-gcc9.4/ to /usr/lib/
- Push all libs under libs/SNPE2.19/hexagon-v68/unsigned/ to /usr/lib/rfsa/adsp/
- Push "aidemo" app to device
	Note: Users can modify rtspLists, videoLists anh NUM_THREADS in main.cpp to test with specific configurations. In videoLists, the path must be the absolute path of video on device.
- To run the demo application:
	+ export XDG_RUNTIME_DIR=/run/user/root
	+ ./aidemo --add
	+ ./aidemo

Prepare .dlc models:
- cd ${SNPE_ROOT}/examples/Models/
- mkdir scrfd && cd scrfd
- Download SCRFD model from: https://github.com/deepinsight/insightface/tree/master/python-package
- Push the model to ${SNPE_ROOT}/examples/Models/scrfd/onnx/
- Convert from onnx to dlc: snpe-onnx-to-dlc --input_network onnx/buffalo_s/det_500m.onnx --input_dim input.1 "1,3,640,640" --output_path $SNPE_ROOT/examples/Models/scrfd/dlc/det_500m.dlc
- Prepare raw image list:
	+ mkdir -p data/cropped
	+ python scripts/create_scrfd_raws.py -i data/ -d data/cropped/
	+ python scripts/create_file_list.py -i data/cropped/ -o data/cropped/raw_list.txt -e *.raw
	+ python scripts/create_file_list.py -i data/cropped/ -o data/target_raw_list.txt -e *.raw -r
- Quantize the dlc model: snpe-dlc-quantize --input_dlc dlc/det_500m.dlc --input_list data/cropped/raw_list.txt --output_dlc dlc/det_500m_quantized.dlc
