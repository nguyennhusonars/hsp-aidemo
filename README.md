* Build and run demo on host side:
  
  sudo apt-get update
  
  sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  
  sudo apt install clang
  
  sudo apt-get install libc++1 libc++abi1
  
  sudo apt install cmake
  
  git clone https://github.com/nguyennhusonars/hsp-aidemo
  
  cd <path-to-project>
  
  mkdir build && cd build
  
  cmake -DBUILD_X86=1 ..
  
  make -j4

* Build and run demo on device side:
  
  apt-get update
  
  apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  
  apt install cmake
  
  cmake -DBUILD_AARCH64=1 ..
  
  make -j4
