#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
              "The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
    
    gflags::SetUsageMessage("Convert jpg to proto binary file\n"
                            "Usage:\n"
                            "    compute_image_mean INPUT_FILE OUTPUT_FILE\n");
    
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    std::cout << argc << std::endl;
    
    if (argc < 3 || argc > 3) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
        return 1;
    }
    
    std::string input_file_name (argv[1]);
    std::string output_file_name (argv[2]);
    
    std::cout << input_file_name << std::endl;
    std::cout << output_file_name << std::endl;
    
    Mat cv_image = imread(input_file_name);
    
    BlobProto sum_blob;
    Datum datum;
    
    int h = cv_image.size().height;
    int w = cv_image.size().width;
    
    int             resize_height   = h;
    int             resize_width    = w;
    bool            is_color        = true;
    std::string     enc ("jpg");
    
    ReadImageToDatum(input_file_name, 0, resize_height, resize_width, is_color,
                     enc, &datum);
    
    Mat cv_test = DecodeDatumToCVMat(datum, is_color);
    
    imwrite(input_file_name  + ".check.jpg", cv_test);
    
    sum_blob.set_num(1);
    sum_blob.set_channels(datum.channels());
    sum_blob.set_height(datum.height());
    sum_blob.set_width(datum.width());
    
    std::cout << datum.data().size() << std::endl;
    
    int size_in_datum = datum.data().size();
    
    
    for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.add_data(0.);
    }
    for (int i = 0; i < size_in_datum; ++i) {
      sum_blob.set_data(i, datum.data()[i]);
    }
    
    // Write to disk
    if (argc == 3) {
        LOG(INFO) << "Write to " << argv[2];
        WriteProtoToBinaryFile(sum_blob, argv[2]);
    }
    
    return 0;
}
