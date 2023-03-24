#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    VideoCapture cap("video.mp4"); // Video dosyas�n� y�kle

    CascadeClassifier detector("cascade.xml"); // Nesne alg�lay�c� s�n�f�n� olu�tur

    Ptr<Tracker> tracker = TrackerKCF::create(); // Nesne takip algoritmas�n� olu�tur

    Size frame_size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)); // Video karesi boyutunu al

    Scalar box_color(0, 255, 0); // Nesnelerin �evresine �izilecek kutu rengi, Ye�il

    // Videoyu a�
    if (!cap.isOpened()) {
        std::cerr << "Video a��lamad�." << std::endl;
        return -1;
    }

    // �lk kareyi oku
    Mat frame;
    cap.read(frame);

    Rect2d bbox = selectROI(frame, false); // Takip edilecek nesne alan�n� se�

    tracker->init(frame, bbox); // Nesneyi takip etmeye ba�la

    // Videoyu d�ng�ye sok
    while (cap.isOpened())
    {
        cap.read(frame); // Bir sonraki kareyi oku

        // Nesneleri alg�la
        std::vector<Rect> detections;
        detector.detectMultiScale(frame, detections);

        // Alg�lanan nesnelerin kutular�n� �iz
        for (auto&& box : detections) {
            rectangle(frame, box, box_color, 2);
        }

        bool success = tracker->update(frame, bbox); // Nesneyi takip et

        // Takip ba�ar�l�ysa, nesnenin kutusunu �iz
        if (success) {
            rectangle(frame, bbox, box_color, 2);
        }
        
        imshow("Nesne Takibi", frame); // Video karesini g�ster

        // ESC tu�una bas�ld���nda d�ng�y� sonland�r 27 == ESC kodu
        if (waitKey(1) == 27) {
            break;
        }
    }

    cap.release(); // Videoyu serbest b�rak

    return 0;



