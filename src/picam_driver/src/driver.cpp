#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


static const std::string OPENCV_WINDOW = "Image window";


std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


class PiCamDriver {
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher image_pub_;
    cv_bridge::CvImagePtr cv_bridge;
    
    
    int capture_width = 1280 ;
    int capture_height = 720 ;
    int display_width = 1280 ;
    int display_height = 720 ;
    int framerate = 60 ;
    int flip_method = 2 ;
    cv::VideoCapture cap;

    std::string pipeline = gstreamer_pipeline(capture_width,
        capture_height,
        display_width,
        display_height,
        framerate,
        flip_method);
        
    public:
        PiCamDriver() : it_(nh_) {
            image_pub_ = it_.advertise("/picam/image", 1);
            cv::namedWindow(OPENCV_WINDOW);
            cap = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
        }

        ~PiCamDriver() {
            cv::destroyWindow(OPENCV_WINDOW);
        }

        void run() {
            cv::Mat img;
            while (nh_.ok()) {
                if (!cap.read(img)) break;
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
                cv::imshow(OPENCV_WINDOW, img);
                cv::waitKey(3);
                image_pub_.publish(msg);
                }
        }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "picam_driver_node");
    PiCamDriver driver_;
    driver_.run();
    return 0;
}