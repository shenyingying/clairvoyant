#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  // openCL GPU 并行计算;
#include <omp.h> // openMP CPU 并行计算;
#include "src/CMN_MSEnhance.h"
#include <time.h>

int main() {
    // launch OpenCL environment...
    std::vector<cv::ocl::PlatformInfo> plats;
    cv::ocl::getPlatfomsInfo(plats);
    const cv::ocl::PlatformInfo *platform = &plats[0];
    std::cout << "Platform name: " << platform->name().c_str() << std::endl;
    std::cout << "OpenCL CL_PLATFORM_VERSION: " << platform->version().c_str() << std::endl;
    std::cout << "OpenCL CL_PLATFORM_VENDOR: " << platform->vendor().c_str() << std::endl;
    std::cout << "OpenCL deviceNumber: " << platform->deviceNumber() << std::endl;
    cv::ocl::Device current_device;
    platform->getDevice(current_device, 0);
    std::cout << "Device name: " << current_device.name().c_str() << std::endl;
    current_device.set(0);
    std::string opencl_version = platform->version();
    std::string opencl_vendor = platform->vendor();
    std::string opencl_device_name = current_device.name();

    bool is_have_opencl = cv::ocl::haveOpenCL();
    bool is_have_svm = cv::ocl::haveSVM();
    bool is_use_opencl = cv::ocl::useOpenCL();
    bool is_have_amd_blas = cv::ocl::haveAmdBlas();
    bool is_have_amd_fft = cv::ocl::haveAmdFft();
    std::cout << "is_have_opencl:" << is_have_opencl << std::endl;
    std::cout << "is_have_amd_blas:" << is_have_amd_blas << std::endl;
    std::cout << "is_have_amd_fft:" << is_have_amd_fft << std::endl;
    std::cout << "is_have_svm:" << is_have_svm << std::endl;
    std::cout << "is_use_opencl:" << is_use_opencl << std::endl;
    cv::ocl::setUseOpenCL(true);



    cv::Mat src=cv::imread("/home/sy/data/work/eye/image_jpg/im0001.jpg");
//    cv::UMat src = cv::imread("/home/sy/data/work/eye/image_jpg/im0001.jpg", 1).getUMat(cv::ACCESS_RW);
    cv::Mat dst_mse,dst_opencv;
//    cv::UMat dst_opencv;
    uchar *Src,*Dst;
//
    CMN_MSEnhance cmn_msEnhance(src);
//    CMN_MSEnhance cmn_msEnhance;
//    cmn_msEnhance.UCMN_MSEnhance(src);

    double start_refine,end_refine;
    start_refine=cv::getTickCount();
    Src=cmn_msEnhance._mat_Convert_Array();
    Dst=(uchar*)malloc(src.rows*src.cols*src.channels()* sizeof(uchar));
    int status=cmn_msEnhance.IM_CMN_MSEnhance_Refine(Src,Dst,src.cols,src.rows,3,3*src.cols,12);
    if(!status){
        dst_mse=cmn_msEnhance._array_Convert_Mat(Dst);
    }
    end_refine=cv::getTickCount();
    std::cout << "the exe time in cmn_msEnhance refine: " << (end_refine - start_refine) / (cv::getTickFrequency())<<" s"<< std::endl;


    double start_opencv,end_opencv;
    start_opencv=cv::getTickCount();
    dst_opencv=cmn_msEnhance.multiScaleShape(13,src);
//    dst_opencv=cmn_msEnhance.UmultiScaleShape(13,src);
//    cmn_msEnhance.UmultiScaleShape(13,src);
    end_opencv=cv::getTickCount();
    std::cout << "the exe time in opencv gaussion blur: " << (end_opencv - start_opencv) / (cv::getTickFrequency())<<" s" << std::endl;

    cv::imshow("src",src);
    cv::imshow("cmn_msEnhance",dst_mse);
    cv::imshow("opencv_gaussion",dst_opencv);

    cv::imwrite("cmn_msEnhance.jpg",dst_mse);
    cv::imwrite("opencv_gaussion.jpg",dst_opencv);
    cv::waitKey();
    return 0;
}