# install opencv in liunx
1. download
   [sources](https://opencv.org/releases.html)  下载 3.4版本的Sources
2. 编译

  
     tar -zxvf opencv-3.4.5.zip
     cd opencv-3.4.5
     mkdir build
     cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
     sudo make 
     sudo make install
     
3. 配置 


     sudo gedit /etc/ld.so.conf.d/opencv.conf 此时是空的，写进入 /usr/local/lib  //将opencv库添加到路径，从而可以让系统找到
     sudo ldconfig  //使上述配置命令生效
     sudo gedit /etc/bash.bashrc //配置bash 在末尾添加如下命令
      
          PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig 
          export PKG_CONFIG_PATH 
     
     source /etc/bash.bashrc  //使bash配置生效
     sudo updatedab           //更新
     
4. guide：
   
   
        cd opencv-3.4.0/smaples/cpp/example_cmake 
        sudo gedit CMakeLists.txt //可以模仿改文件写 CMakeList.txt
        
5. openMP [cmakelist](file/CMakeLists.txt)
6. openCL [envir](file/main.cpp)
7. cfc [code](https://pan.baidu.com/s/1CgO-R5d0sF4I2Nlq4MTzRA)



     
 
     