# 温馨提示：
1. 一定不要处女党，喜欢删除自己不知道功能的文件（除了/home 之外的所有文件）
2. 一旦系统崩溃，立马用个启动U盘识别系统，然后用另一个U盘把重要文件copy出来。

# 常见问题

1. pip question:
  
       ImportError: No module named 'pip._internal'
    执行pip3：
    
        import: not authorized `sys' @ error/constitute.c/WriteImage/1028.
        from: can't read /var/mail/pip
        /usr/bin/pip3: 行 11: 未预期的符号 `main.main' 附近有语法错误
        /usr/bin/pip3: 行 11: `    sys.exit(main.main())'
   answer1:
       
       curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
       python3.5 get-pip.py --force-reinstall`
2. Ubuntun16.04下安装python3.4

      a: [download](https://www.python.org/ftp/python/3.4.2/Python-3.4.2.tgz)
      b: install
      
          tar -xzvf Python-3.4.2.tgz
          cd Python-3.4.2
          ./configure
          make
          sudo make install
      c: config python3.4 (may need)
      
         sudo rm /usr/bin/python
         sudo ln -s /usr/local/Python/Python3.4.2/python /usr/bin/python   
      d: 错误 E: Sub-process /usr/bin/dpkg returned an error code (1
      
        cd /var/lib/dpkg
        sudo mv info info.bak
        sudo mkdir info