#include <iostream>
#include <string>
#include <vector>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/superres/superres.hpp>
//#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/opencv_modules.hpp>

#include "GraphInference/Inference.h"

using namespace tensorflow_cc_inference;
using namespace std;
using namespace cv;
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_thread.h>
}

#define SFM_REFRESH_EVENT  (SDL_USEREVENT + 1)
#define SFM_BREAK_EVENT  (SDL_USEREVENT + 2)
#define SEP_WIDTH 4
#define WINDOW_TITLE "Hive"

int exitFlag = 0;

int pauseFlag = 0;

int sep_x, sep_y; // Separator between old and new frame

struct FFmpegContext
{
    string filename;
    AVFormatContext *pFormatCtx;
    AVCodecContext *pCodecCtx;
    AVCodec *pCodec;

    AVFrame *pFrame;
    AVFrame *pFrameYUV;

    int videoStream;
    uint8_t *buffer;
    int numBytes;

};

struct FrameContainer
{
    AVFrame *frame;
    int64_t pts;

    bool operator<(const FrameContainer &rhs) const
    {
        return pts > rhs.pts;
    }
};

struct FrameThreadDataContainer
{
    priority_queue<FrameContainer> *queue;
    FFmpegContext *cxt;
    SwsContext *sws_ctx;
};

struct ColorPoint
{
    ColorPoint()
    {}

    ColorPoint(uint8_t y, uint8_t u, uint8_t v)
        : y(y), u(u), v(v)
    {}

    uint8_t y, u, v;
};

priority_queue<FrameContainer> framesQueue1, framesQueue2;

int refreshThread(void *data);

int fetchFramesThread(void *data);

void startThreads(FFmpegContext *ffmpegContext1, FFmpegContext *ffmpegContext2, SwsContext *sws_ctx1,
                  SwsContext *sws_ctx2);

int setupFFmpegCxt(FFmpegContext *cxt, const string &filename);

void clearFFmpegContext(FFmpegContext *cxt);

int setupSDL(SDL_Window *&screen, SDL_Renderer *&sdlRenderer, SDL_Texture *&sdlTexture, SwsContext *&sws_ctx1,
             SwsContext *&sws_ctx2, AVCodecContext *pCodecCtx);

int getFrame(FFmpegContext *cxt, SwsContext *sws_ctx);

AVFrame mixFrames(AVFrame *frame1, AVFrame *frame2, int height, int width);

void enhanceFrame(AVFrame *f);

//Model *model = new Model();
auto srModel = Inference("../graphs/model.pb", "conv2d_1_input", "conv2d_3/BiasAdd");

int main()
{
    string filename = "titanic.ts";

    FFmpegContext ffmpegContext1;
    FFmpegContext ffmpegContext2;

    if (setupFFmpegCxt(&ffmpegContext1, filename) != 0 || setupFFmpegCxt(&ffmpegContext2, filename) != 0)
        return -1;

    // Super resolution
    ffmpegContext2.pCodecCtx->sr_enabled = 1;
    ffmpegContext2.pCodecCtx->sr_ptr = &enhanceFrame;

    // SDL
    SDL_Window *screen;
    SDL_Renderer *sdlRenderer;
    SDL_Texture *sdlTexture;
    SwsContext *sws_ctx1;
    SwsContext *sws_ctx2;

    if (setupSDL(screen, sdlRenderer, sdlTexture, sws_ctx1, sws_ctx2, ffmpegContext1.pCodecCtx) != 0)
        return -1;

    // Execution threads
    startThreads(&ffmpegContext1, &ffmpegContext2, sws_ctx1, sws_ctx2);

    SDL_Event event;

    while (1) {
        SDL_WaitEvent(&event);
        if (event.type == SFM_REFRESH_EVENT) {
            if (framesQueue1.empty() || framesQueue2.empty() || framesQueue1.top().pts != framesQueue2.top().pts)
                continue;

            AVFrame *frame1 = framesQueue1.top().frame;
            AVFrame *frame2 = framesQueue2.top().frame;
            framesQueue1.pop();
            framesQueue2.pop();

            AVFrame mixedFrame = mixFrames(frame1, frame2, ffmpegContext1.pCodecCtx->height,
                                           ffmpegContext1.pCodecCtx->width);

            SDL_UpdateTexture(sdlTexture, nullptr, mixedFrame.data[0],
                              mixedFrame.linesize[0]);
            SDL_RenderClear(sdlRenderer);
            SDL_RenderCopy(sdlRenderer, sdlTexture, nullptr, nullptr);
            SDL_RenderPresent(sdlRenderer);


            av_frame_free(&frame1);
            av_frame_free(&frame2);
        }
        else if (event.type == SDL_KEYDOWN) {
            if (event.key.keysym.sym == SDLK_SPACE) {
                pauseFlag = !pauseFlag;
            }
            else if (event.key.keysym.sym == SDLK_ESCAPE) {
                exitFlag = 1;
                break;
            }
        }
        else if (event.type == SDL_MOUSEBUTTONDOWN) {
            SDL_GetMouseState(&sep_x, &sep_y);
        }
        else if (event.type == SDL_QUIT) {
            exitFlag = 1;
        }
        else if (event.type == SFM_BREAK_EVENT) {
            break;
        }
    }

    SDL_Quit();

    clearFFmpegContext(&ffmpegContext1);
    clearFFmpegContext(&ffmpegContext2);

    return 0;
}

int refreshThread(void *opaque)
{
    exitFlag = 0;
    pauseFlag = 0;

    while (!exitFlag) {
        if (!pauseFlag) {
            SDL_Event event;
            event.type = SFM_REFRESH_EVENT;
            SDL_PushEvent(&event);
        }

        SDL_Delay(40);
    }

    exitFlag = 0;
    pauseFlag = 0;

    SDL_Event event;
    event.type = SFM_BREAK_EVENT;
    SDL_PushEvent(&event);

    return 0;
}

int fetchFramesThread(void *data)
{
    FrameThreadDataContainer *cont = (FrameThreadDataContainer *) data;
    int n = cont->cxt->pCodecCtx->width * cont->cxt->pCodecCtx->height;

    while (!exitFlag) {
        int status = getFrame(cont->cxt, cont->sws_ctx);
        if (status == 1)
            continue;
        else if (status == -1)
            break;

        FrameContainer frameContainer;
        frameContainer.pts = av_frame_get_best_effort_timestamp(cont->cxt->pFrame);

        frameContainer.frame = av_frame_alloc();
        frameContainer.frame->linesize[0] = cont->cxt->pFrameYUV->linesize[0];
        frameContainer.frame->data[0] = new uint8_t[int(n * 1.5)];

        memcpy(frameContainer.frame->data[0], cont->cxt->pFrameYUV->data[0], n * 1.5 * sizeof(uint8_t));

        cont->queue->push(frameContainer);
    }

    return 0;
}

void startThreads(FFmpegContext *ffmpegContext1, FFmpegContext *ffmpegContext2, SwsContext *sws_ctx1,
                  SwsContext *sws_ctx2)
{
    FrameThreadDataContainer *t1 = new FrameThreadDataContainer;
    FrameThreadDataContainer *t2 = new FrameThreadDataContainer;

    t1->cxt = ffmpegContext1;
    t1->queue = &framesQueue1;
    t1->sws_ctx = sws_ctx1;

    t2->cxt = ffmpegContext2;
    t2->queue = &framesQueue2;
    t2->sws_ctx = sws_ctx2;

    SDL_CreateThread(fetchFramesThread, nullptr, (void *) t1);
    SDL_CreateThread(fetchFramesThread, nullptr, (void *) t2);
    SDL_CreateThread(refreshThread, nullptr, nullptr);
}

int setupFFmpegCxt(FFmpegContext *cxt, const string &filename)
{
    cxt->filename = filename;

    cxt->pFormatCtx = nullptr;

    // Open video file
    if (avformat_open_input(&cxt->pFormatCtx, filename.c_str(), nullptr, nullptr) != 0)
        return -1; // Couldn't open file

    // Retrieve stream information
    if (avformat_find_stream_info(cxt->pFormatCtx, nullptr) < 0)
        return -1; // Couldn't find stream information

    // Dump information about file onto standard error
    av_dump_format(cxt->pFormatCtx, 0, cxt->filename.c_str(), 0);

    // Find the first video stream
    for (int i = 0; i < cxt->pFormatCtx->nb_streams; i++) {
        if (cxt->pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            cxt->videoStream = i;
            break;
        }
    }

    if (cxt->videoStream == -1)
        return -1; // Didn't find a video stream

    // Get a pointer to the codec context for the video stream
    AVCodec *pCodec1 = avcodec_find_decoder(cxt->pFormatCtx->streams[cxt->videoStream]->codecpar->codec_id);

    if (pCodec1 == nullptr) {
        fprintf(stderr, "Unsupported codec!\n");
        return -1; // Codec not found
    }

    cxt->pCodecCtx = avcodec_alloc_context3(pCodec1);
    avcodec_parameters_to_context(cxt->pCodecCtx, cxt->pFormatCtx->streams[cxt->videoStream]->codecpar);
    cxt->pCodecCtx->sr_enabled = 0;

    // Open codec
    if (avcodec_open2(cxt->pCodecCtx, pCodec1, nullptr) < 0)
        return -1; // Could not open codec

    // Allocate the frames
    cxt->pFrame = av_frame_alloc();
    cxt->pFrameYUV = av_frame_alloc();

    if (cxt->pFrameYUV == nullptr || cxt->pFrame == nullptr)
        return -1;

    // Determine required buffer size and allocate buffer
    cxt->numBytes = avpicture_get_size(AV_PIX_FMT_RGB24, cxt->pCodecCtx->width, cxt->pCodecCtx->height);
    cxt->buffer = (uint8_t *) av_malloc(cxt->numBytes * sizeof(uint8_t));

    avpicture_fill((AVPicture *) cxt->pFrameYUV, cxt->buffer, AV_PIX_FMT_YUV420P,
                   cxt->pCodecCtx->width, cxt->pCodecCtx->height);

    return 0;
}

void clearFFmpegContext(FFmpegContext *cxt)
{
    // Free the frames
    av_free(cxt->buffer);
    av_free(cxt->pFrameYUV);
    av_free(cxt->pFrame);

    avcodec_close(cxt->pCodecCtx);
    avformat_close_input(&cxt->pFormatCtx);
}

int setupSDL(SDL_Window *&screen, SDL_Renderer *&sdlRenderer, SDL_Texture *&sdlTexture, SwsContext *&sws_ctx1,
             SwsContext *&sws_ctx2, AVCodecContext *pCodecCtx)
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
        fprintf(stderr, "Could not initialize SDL - %s\n", SDL_GetError());
        return -1;
    }

    screen = SDL_CreateWindow(WINDOW_TITLE, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                              pCodecCtx->width, pCodecCtx->height, SDL_WINDOW_OPENGL);
    if (!screen) {
        printf("SDL: could not create window - exiting:%s\n", SDL_GetError());
        return -1;
    }

    sdlRenderer = SDL_CreateRenderer(screen, -1, 0);

    sdlTexture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_IYUV, SDL_TEXTUREACCESS_STREAMING,
                                   pCodecCtx->width,
                                   pCodecCtx->height);

    // initialize SWS context for software scaling
    sws_ctx1 = sws_getContext(
        pCodecCtx->width,
        pCodecCtx->height,
        pCodecCtx->pix_fmt,
        pCodecCtx->width,
        pCodecCtx->height,
        AV_PIX_FMT_YUV420P,
        SWS_BICUBIC,
        nullptr,
        nullptr,
        nullptr);

    sws_ctx2 = sws_getContext(
        pCodecCtx->width,
        pCodecCtx->height,
        pCodecCtx->pix_fmt,
        pCodecCtx->width,
        pCodecCtx->height,
        AV_PIX_FMT_YUV420P,
        SWS_BICUBIC,
        nullptr,
        nullptr,
        nullptr);

    sep_x = pCodecCtx->width / 2; // Initial separator location

    return 0;
}

int getFrame(FFmpegContext *cxt, SwsContext *sws_ctx)
{
    int frameFinished;
    AVPacket packet;

    while (1) {
        if (av_read_frame(cxt->pFormatCtx, &packet) < 0)
            return -1;

        if (packet.stream_index == cxt->videoStream)
            break;
    }

    int ret = avcodec_decode_video2(cxt->pCodecCtx, cxt->pFrame, &frameFinished, &packet);

    if (ret < 0)
        return 1;

    if (frameFinished) {
        sws_scale(sws_ctx, (const unsigned char *const *) cxt->pFrame->data,
                  cxt->pFrame->linesize, 0,
                  cxt->pCodecCtx->height, cxt->pFrameYUV->data,
                  cxt->pFrameYUV->linesize);
    }
    else {
        return 1;
    }

    av_packet_unref(&packet);

    return 0;
}

AVFrame mixFrames(AVFrame *frame1, AVFrame *frame2, int height, int width)
{
    int total = width * height;

    AVFrame ret;

    ret.linesize[0] = frame1->linesize[0];
    ret.data[0] = new uint8_t[int(total * 1.5)];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            uint8_t *data1 = frame1->data[0];
            uint8_t *data2 = frame2->data[0];

            uint8_t &ny = ret.data[0][i * width + j];
            uint8_t &nu = ret.data[0][(i / 2) * (width / 2) + (j / 2) + total];
            uint8_t &nv = ret.data[0][(i / 2) * (width / 2) + (j / 2) + total + (total / 4)];

            if (j > sep_x - SEP_WIDTH / 2 && j < sep_x + SEP_WIDTH / 2) { // Red separator
                ny = 76;
                nu = 84;
                nv = 255;
            }
            else if (j >= sep_x + SEP_WIDTH / 2) { // Modified
                ny = data2[i * width + j];
                nu = data2[(i / 2) * (width / 2) + (j / 2) + total];
                nv = data2[(i / 2) * (width / 2) + (j / 2) + total + (total / 4)];
            }
            else if (j <= sep_x - SEP_WIDTH / 2) { // Original
                ny = data1[i * width + j];
                nu = data1[(i / 2) * (width / 2) + (j / 2) + total];
                nv = data1[(i / 2) * (width / 2) + (j / 2) + total + (total / 4)];
            }
        }
    }

    return ret;
}

void enhanceFrame(AVFrame *f)
{
    if (!f->buf[0] || !f->buf[1] || !f->buf[2] || f->pict_type != AV_PICTURE_TYPE_I)
        return;


    vector<vector<ColorPoint>> image(f->height, vector<ColorPoint>(f->width));
    vector<vector<float>> x(f->height, vector<float>(f->width));

    // Create input tensor.
    int64_t dims[] = {1, f->height, f->width, 1};
    TF_Tensor *in = TF_AllocateTensor(TF_FLOAT, dims, 4, 1 * f->height * f->width * sizeof(float));
    float *f_data = (float *) (TF_TensorData(in));

    // Fill the image
    for (int i = 0; i < f->height; ++i) {
        for (int j = 0; j < f->width; ++j) {
            uint8_t y = f->buf[0]->data[i * f->linesize[0] + j];
            uint8_t u = f->buf[1]->data[i / 2 * f->linesize[1] + j / 2];
            uint8_t v = f->buf[2]->data[i / 2 * f->linesize[2] + j / 2];

            // Fill the input tensor by the y channel normalized values.
            f_data[i * f->width + j] = float(y / 255.0);

            image[i][j] = ColorPoint(y, u, v);
            x[i][j] = y;
        }
    }

    // Feed forward input tensor to the model.
    TF_Tensor *out = srModel(in);
    float *data2_ = (float *) (TF_TensorData(out));

    // Fill the frame again
    for (int i = 0; i < f->height; ++i) {
        for (int j = 0; j < f->width; ++j) {

            int val = static_cast<int>(data2_[i * f->width + j] * 255.0);

            val = val < 0 ? 0 : val;
            val = val > 255 ? 255 : val;

            f->buf[0]->data[i * f->linesize[0] + j] = uchar(val);
            f->buf[1]->data[i / 2 * f->linesize[1] + j / 2] = image[i][j].u;
            f->buf[2]->data[i / 2 * f->linesize[2] + j / 2] = image[i][j].v;
        }
    }
}