# Ubuntu16.04下安装fftw，png，zlib
    
    
     最近在研究图像降噪，学习到了经典的降噪算法BM3D，其中用到了fftw png zlib 这几个库
 ## fftw 安装
 1.[download](ftp://ftp.fftw.org/pub/fftw/) 
 
  
      下载比较新的版本安装.
      
 2. install commond
 
     
      tar -zxvf fftw<version>.tar
      cd fftw<version>
      ./configure --prefix=/usr/local/fftw --enable-float --enable_shared --disable-fortran
      sudo make
      sudo make install
      
 3. 在clion下写cmakelist.txt
 
     
      set(INC_DIR
        /usr/local/fftw/include
        )
      set(LINK_DIR
        /usr/local/fftw/lib
        )
       include_directories(${INC_DIR})
       link_directories(${LINK_DIR})
       link_libraries(fftw3)
       link_libraries(fftw3f)  # n多文章说如果你用fftwf 添加 float的lib 不过都没有明说如何添加lib，我花了一天时间才弄好。
 
 ## png 安装
 1 [download](https://sourceforge.net/projects/libpng/files/)
 
 2 install commond
 
 
      tar -zxvf png<version>.tar
      cd png<version>
      ./config --prefix=/usr/local/libpng # 这个是安装路径 你可以放在任何位置,若放在/home下，下面执行的时候不需要加sudo;
      sudo make
      sudo make install
 3.在clion下写cmakelist.txt
 
 
        set(INC_DIR
           /usr/local/libpng/include
        )
        set(LINK_DIR
            /usr/local/libpng/lib
            )
        include_directories(${INC_DIR})
        link_directories(${LINK_DIR})
        link_libraries(png)
 
 
      