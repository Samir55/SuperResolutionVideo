#pragma once

#include "wrappers.h"
#include "utils/inference.h"
#include "utils/constants.h"
#include <unistd.h>

using namespace tensorflow_cc_inference;

class Decoder {
public:

    /**
     * Sets up the FFmpeg context for specific video.
     *
     * @param cxt       The FFmpeg context to set up.
     * @param fileName  The video path.
     * @param enableSR  Enable super resolution for this video.
     * @param srPtr     Pointer to the superresolution function.
     *
     * @return          0 if success, -1 otherwise.
     */
    int setupFFmpegCxt(FFmpegContext* cxt, const string& fileName,
                       bool enableSR = false, void (* srPtr)(AVFrame*) = nullptr) {
        cxt->fileName = fileName;

        cxt->formatCxt = nullptr;

        // Open video file
        if (avformat_open_input(&cxt->formatCxt, cxt->fileName.c_str(), nullptr, nullptr) != 0)
            return -1; // Couldn't open file

        // Retrieve stream information
        if (avformat_find_stream_info(cxt->formatCxt, nullptr) < 0)
            return -1; // Couldn't find stream information

        // Dump information about file onto standard error
        av_dump_format(cxt->formatCxt, 0, cxt->fileName.c_str(), 0);

        // Find the first video stream
        for (int i = 0; i < cxt->formatCxt->nb_streams; i++) {
            if (cxt->formatCxt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                cxt->videoStream = i;
                break;
            }
        }

        if (cxt->videoStream == -1)
            return -1; // Didn't find a video stream

        // Get a pointer to the codec context for the video stream
        AVCodec* codec = avcodec_find_decoder(cxt->formatCxt->streams[cxt->videoStream]->codecpar->codec_id);

        if (codec == nullptr) {
            cerr << "Unsupported codec!" << endl;
            return -1; // Codec not found
        }

        cxt->codecCtx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(cxt->codecCtx, cxt->formatCxt->streams[cxt->videoStream]->codecpar);
        cxt->codecCtx->sr_enabled = 0;
        cxt->fps = av_q2d(cxt->formatCxt->streams[cxt->videoStream]->r_frame_rate);

        // Open codec
        if (avcodec_open2(cxt->codecCtx, codec, nullptr) < 0)
            return -1; // Could not open codec

        // Allocate the frames
        cxt->frame = av_frame_alloc();
        cxt->frameYUV = av_frame_alloc();

        if (cxt->frameYUV == nullptr || cxt->frame == nullptr)
            return -1;

        // Determine required buffer size and allocate buffer
        cxt->numBytes = avpicture_get_size(AV_PIX_FMT_RGB24, cxt->codecCtx->width, cxt->codecCtx->height);
        cxt->buffer = (uint8_t*) av_malloc(cxt->numBytes * sizeof(uint8_t));

        avpicture_fill((AVPicture*) cxt->frameYUV, cxt->buffer, AV_PIX_FMT_YUV420P,
                       cxt->codecCtx->width, cxt->codecCtx->height);

        // Setup scale context
        cxt->swsCtx = sws_getContext(
                cxt->codecCtx->width,
                cxt->codecCtx->height,
                cxt->codecCtx->pix_fmt,
                cxt->codecCtx->width,
                cxt->codecCtx->height,
                AV_PIX_FMT_YUV420P,
                SWS_BICUBIC,
                nullptr,
                nullptr,
                nullptr);

        // Super resolution
        cxt->enableSR = enableSR;

        if (enableSR) {
            cxt->codecCtx->sr_enabled = 1;
            cxt->codecCtx->sr_ptr = &Decoder::enhanceFrame;
        }

        return 0;
    }

    /**
     * Overlaps the frames from two different videos (LR & SR).
     * The SR frame is the one on the right of the separator.
     *
     * @param frameLR   The low-res frame.
     * @param frameSR   The high-res frame.
     * @param sepX      The separator x.
     * @param width     The frame width.
     * @param height    The frame height.
     *
     * @return          The new mixed frame.
     */
    AVFrame mixFrames(AVFrame* frameLR, AVFrame* frameSR, int width, int height, int sepX) {
        int total = width * height;

        AVFrame ret;

        ret.linesize[0] = frameLR->linesize[0];
        ret.data[0] = new uint8_t[int(total * 1.5)];

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                uint8_t* data1 = frameLR->data[0];
                uint8_t* data2 = frameSR->data[0];

                uint8_t& ny = ret.data[0][i * width + j];
                uint8_t& nu = ret.data[0][(i / 2) * (width / 2) + (j / 2) + total];
                uint8_t& nv = ret.data[0][(i / 2) * (width / 2) + (j / 2) + total + (total / 4)];

                if (j >= sepX) { // Modified
                    ny = data2[i * width + j];
                    nu = data2[(i / 2) * (width / 2) + (j / 2) + total];
                    nv = data2[(i / 2) * (width / 2) + (j / 2) + total + (total / 4)];
                } else if (j < sepX) { // Original
                    ny = data1[i * width + j];
                    nu = data1[(i / 2) * (width / 2) + (j / 2) + total];
                    nv = data1[(i / 2) * (width / 2) + (j / 2) + total + (total / 4)];
                }
            }
        }

        return ret;
    }

    /**
     * Decodes a frame from the given FFmpeg context video.
     * Static to be executed in the thread functions.
     *
     * @param cxt      The FFmpeg context that holds the video.
     *
     * @return          -1 if eof, 1 frame decoding failed, 0 otherwise.
     */
    static int decodeFrame(FFmpegContext* cxt) {
        int frameFinished;
        AVPacket packet;

        while (1) {
            if (av_read_frame(cxt->formatCxt, &packet) < 0)
                return -1;

            if (packet.stream_index == cxt->videoStream)
                break;
        }

        int ret = avcodec_decode_video2(cxt->codecCtx, cxt->frame, &frameFinished, &packet);

        if (ret < 0)
            return 1;

        if (frameFinished) {
            sws_scale(cxt->swsCtx, (const unsigned char* const*) cxt->frame->data,
                      cxt->frame->linesize, 0,
                      cxt->codecCtx->height, cxt->frameYUV->data,
                      cxt->frameYUV->linesize);
        } else {
            return 1;
        }

        av_packet_unref(&packet);

        return 0;
    }

    /**
     * Applys the super res to the given frame.
     * Static to be executed in the thread functions.
     *
     * @param f The to-be-enhanced frame pointer.
     */
    static void enhanceFrame(AVFrame* f) {
        if (!f->buf[0] || !f->buf[1] || !f->buf[2] || f->pict_type != AV_PICTURE_TYPE_I)
            return;

        vector<vector<ColorPoint>> image(f->height, vector<ColorPoint>(f->width));
        vector<vector<float>> x(f->height, vector<float>(f->width));

        // Create input tensor.
        int64_t dims[] = {1, f->height, f->width, 1};
        TF_Tensor* in = TF_AllocateTensor(TF_FLOAT, dims, 4, 1 * f->height * f->width * sizeof(float));
        float* normData = (float*) (TF_TensorData(in));

        // Fill the image
        for (int i = 0; i < f->height; ++i) {
            for (int j = 0; j < f->width; ++j) {
                uint8_t y = f->buf[0]->data[i * f->linesize[0] + j];
                uint8_t u = f->buf[1]->data[i / 2 * f->linesize[1] + j / 2];
                uint8_t v = f->buf[2]->data[i / 2 * f->linesize[2] + j / 2];

                // Fill the input tensor by the y channel normalized values.
                normData[i * f->width + j] = float(y / 255.0);

                image[i][j] = ColorPoint(y, u, v);
                x[i][j] = y;
            }
        }

        // Feed forward input tensor to the model.
        TF_Tensor* out = model(in);
        float* newVals = (float*) (TF_TensorData(out));

        // Fill the frame again
        for (int i = 0; i < f->height; ++i) {
            for (int j = 0; j < f->width; ++j) {

                int val = static_cast<int>(newVals[i * f->width + j] * 255.0);

                val = val < 0 ? 0 : val;
                val = val > 255 ? 255 : val;

                f->buf[0]->data[i * f->linesize[0] + j] = uint8_t(val);
                f->buf[1]->data[i / 2 * f->linesize[1] + j / 2] = image[i][j].u;
                f->buf[2]->data[i / 2 * f->linesize[2] + j / 2] = image[i][j].v;
            }
        }
    }

    /**
     * Clears the FFmpeg context.
     *
     * @param cxt    The FFmpeg context to set up.
     */
    void clearFFmpegContext(FFmpegContext* cxt) {
        // Free the frames
        av_free(cxt->buffer);
        av_free(cxt->frameYUV);
        av_free(cxt->frame);

        avcodec_close(cxt->codecCtx);
        avformat_close_input(&cxt->formatCxt);
    }

private:

    // The model to be used for enhancements
    static Inference model;
};

// Initialize static member
Inference Decoder::model("../src/models/srcnn.pb", "conv2d_1_input", "conv2d_3/BiasAdd");