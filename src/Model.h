#ifndef SUPER_VIDEO_RES_MODEL_H
#define SUPER_VIDEO_RES_MODEL_H

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include <iostream>
#include <string>
#include <vector>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/superres/superres.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/opencv_modules.hpp>

using namespace std;
using namespace tensorflow;
using namespace cv;


class Model
{
public:
    Session *session;
    GraphDef graph_def;
    SessionOptions opts;
    Tensor inputTensor;
    vector<Tensor> outputs;

    /**
     * Initialize tensorflow session.
     * @param modelGraphFile string the path to the model graph file.
     */
    void initSession(string modelGraphFile = "")
    {
        if (modelGraphFile.empty()) {
            modelGraphFile = "model.pb";
        }

        // Read the model graph.
        TF_CHECK_OK(ReadBinaryProto(Env::Default(), modelGraphFile, &graph_def));

        // Set GPU options.
//        opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
//        opts.config.mutable_gpu_options()->set_allow_growth(true);

        // Create a new session.
        TF_CHECK_OK(NewSession(opts, &session));

        // Load graph into session
        TF_CHECK_OK(session->Create(graph_def));

        // Debugging
        cout << "Layer names in the model" << endl;
        int node_count = graph_def.node_size();
        for (int i = 0; i < node_count; i++) {
            auto n = graph_def.node(i);
            cout << "Names : " << n.name() << endl;

        }

    }

    void createInputTensor(int batch_size = 1, int height = 640, int width = 460, int channels = 1)
    {
        inputTensor = Tensor(DT_FLOAT, TensorShape({batch_size, height, width, channels}));
    }

    void feedForward()
    {
        Status run_status = session->Run({{"conv2d_1_input", inputTensor}},
                                         {"srcnn_output/Identity"}, {}, &outputs);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
        }
    }
};

#endif //SUPER_VIDEO_RES_MODEL_H
