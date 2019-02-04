#pragma once

#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <SDL2/SDL_thread.h>
#include <SDL2/SDL.h>
}

#include "utils/concurrent_priority_queue.h"
#include "utils/constants.h"

using namespace std;

struct FrameWrapper {
    AVFrame* frame;
    int64_t pts;

    bool operator<(const FrameWrapper& rhs) const {
        return pts > rhs.pts;
    }
};

struct FFmpegContext {
    string fileName;
    AVFormatContext* formatCxt;
    AVCodecContext* codecCtx;

    AVFrame* frame;
    AVFrame* frameYUV;

    SwsContext* swsCtx;

    int videoStream;
    int numBytes;
    uint8_t* buffer;

    double fps;

    bool enableSR;
};

struct SDLContext {
    SDL_Window* window;
    SDL_Renderer* sdlRenderer;
    SDL_Texture* sdlTexture;
    SwsContext* swsMagCtx;
    SDL_Event event;

    int mouseX, mouseY, globalMouseX, globalMouseY, windowX, windowY, windowWidth, windowHeight;
    int sepX, sepY;
    int sepColorY = RED_Y, sepColorU = RED_U, sepColorV = RED_V;
};

struct PlayerContext {
    string fileName;

    FFmpegContext ffmCxtLR;
    FFmpegContext ffmCxtSR;

    SDLContext sdlCxt;

    concurrent_priority_queue<FrameWrapper> framesQueueLR, framesQueueSR;

    bool exitFlag = 0;
    bool pauseFlag = 0;
    bool magnifierFlag = 0;
    bool dragSeparatorFlag = 0;
};

struct FrameThreadDataWrapper {
    concurrent_priority_queue<FrameWrapper>* queue;

    PlayerContext* plyrCxt;
    FFmpegContext* ffmCxt;
};

struct ColorPoint {
    ColorPoint() {}

    ColorPoint(uint8_t y, uint8_t u, uint8_t v) : y(y), u(u), v(v) {}

    uint8_t y, u, v;
};