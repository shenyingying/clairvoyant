# 导入常见问题
1. You can move the version from the manifest to the defaultConfig in the build
![question1](pic/ans1.png)
answer:
![answer1](pic/ans11.png)
2. Could not find com.android.tools.build:aapt2:3.2.1-4818971.
![question2](pic/ans2.png)
answer:
![answer2](pic/ans22.png)
3. 错误: 找不到符号private final Fill<T> fill;

     ![question3](pic/ans3.png)
     
answer:

     `set to 'bazel', 'cmake', 'makefile', 'none'
      def nativeBuildSystem = 'none'`
    [FYI](https://github.com/tensorflow/tensorflow/issues/21431)