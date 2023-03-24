#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    VideoCapture cap("video.mp4"); // Video dosyasýný yükle

    CascadeClassifier detector("cascade.xml"); // Nesne algýlayýcý sýnýfýný oluþtur

    Ptr<Tracker> tracker = TrackerKCF::create(); // Nesne takip algoritmasýný oluþtur

    Size frame_size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)); // Video karesi boyutunu al

    Scalar box_color(0, 255, 0); // Nesnelerin çevresine çizilecek kutu rengi, Yeþil

    // Videoyu aç
    if (!cap.isOpened()) {
        std::cerr << "Video açýlamadý." << std::endl;
        return -1;
    }

    // Ýlk kareyi oku
    Mat frame;
    cap.read(frame);

    Rect2d bbox = selectROI(frame, false); // Takip edilecek nesne alanýný seç

    tracker->init(frame, bbox); // Nesneyi takip etmeye baþla

    // Videoyu döngüye sok
    while (cap.isOpened())
    {
        cap.read(frame); // Bir sonraki kareyi oku

        // Nesneleri algýla
        std::vector<Rect> detections;
        detector.detectMultiScale(frame, detections);

        // Algýlanan nesnelerin kutularýný çiz
        for (auto&& box : detections) {
            rectangle(frame, box, box_color, 2);
        }

        bool success = tracker->update(frame, bbox); // Nesneyi takip et

        // Takip baþarýlýysa, nesnenin kutusunu çiz
        if (success) {
            rectangle(frame, bbox, box_color, 2);
        }
        
        imshow("Nesne Takibi", frame); // Video karesini göster

        // ESC tuþuna basýldýðýnda döngüyü sonlandýr 27 == ESC kodu
        if (waitKey(1) == 27) {
            break;
        }
    }

    cap.release(); // Videoyu serbest býrak

    return 0;



