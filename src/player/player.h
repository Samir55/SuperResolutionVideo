#pragma once

#include "decoder.h"
#include "viewer.h"
#include "utils/constants.h"

class Player {
public:

    /**
     * Starts playing the given video.
     *
     * @param fileName  The video path.
     *
     * @return  0 if sucess, -1 otherwise.
     */
    int play(string fileName) {
        plyrCxt.fileName = fileName;

        if (decoder.setupFFmpegCxt(&plyrCxt.ffmCxtLR, plyrCxt.fileName) != 0 ||
            decoder.setupFFmpegCxt(&plyrCxt.ffmCxtSR, plyrCxt.fileName, true) != 0)
            return -1;

        if (viewer.setupSDL(plyrCxt.sdlCxt, plyrCxt.ffmCxtLR.codecCtx->height, plyrCxt.ffmCxtLR.codecCtx->width) != 0)
            return -1;

        startThreads(&plyrCxt);

        AVFrame* prevFrameLR = nullptr, * prevFrameSR = nullptr;

        while (1) {
            SDL_Event* e = &plyrCxt.sdlCxt.event;
            SDL_WaitEvent(e);

            viewer.updateState(&plyrCxt.sdlCxt, plyrCxt.dragSeparatorFlag);
            if (eventHandler(e, prevFrameLR, prevFrameSR) != 0)
                break;
        }

        SDL_Quit();

        decoder.clearFFmpegContext(&plyrCxt.ffmCxtLR);
        decoder.clearFFmpegContext(&plyrCxt.ffmCxtSR);

        return 0;
    }

private:

    PlayerContext plyrCxt;

    Decoder decoder;
    Viewer viewer;

    /**
     * Handles the different events while looping.
     *
     * @param e             The event pointer.
     * @param prevFrameLR   The previous low-res frame.
     * @param prevFrameSR   The previous high-res frame.
     *
     * @return              -1 to break the event loop, 0 otherwise.
     */
    int eventHandler(SDL_Event* e, AVFrame*& prevFrameLR, AVFrame*& prevFrameSR) {
        if (e->type == REFRESH_EVENT) {
            if (plyrCxt.framesQueueLR.empty() || plyrCxt.framesQueueSR.empty() ||
                plyrCxt.framesQueueLR.top().pts != plyrCxt.framesQueueSR.top().pts)
                return 0;

            AVFrame* frameLR = plyrCxt.framesQueueLR.top().frame;
            AVFrame* frameSR = plyrCxt.framesQueueSR.top().frame;
            plyrCxt.framesQueueLR.pop();
            plyrCxt.framesQueueSR.pop();

            viewer.render(&plyrCxt.sdlCxt,
                          decoder.mixFrames(frameLR, frameSR,
                                            plyrCxt.sdlCxt.windowWidth, plyrCxt.sdlCxt.windowHeight,
                                            plyrCxt.sdlCxt.sepX),
                          plyrCxt.magnifierFlag);

            av_frame_free(&prevFrameLR);
            av_frame_free(&prevFrameSR);

            prevFrameLR = frameLR;
            prevFrameSR = frameSR;
        } else if (e->type == SDL_KEYDOWN) {
            switch (e->key.keysym.sym) {
                case SDLK_m:
                    plyrCxt.magnifierFlag = !plyrCxt.magnifierFlag;

                    if (prevFrameLR && prevFrameSR) {
                        viewer.render(&plyrCxt.sdlCxt,
                                      decoder.mixFrames(prevFrameLR, prevFrameSR,
                                                        plyrCxt.sdlCxt.windowWidth, plyrCxt.sdlCxt.windowHeight,
                                                        plyrCxt.sdlCxt.sepX),
                                      plyrCxt.magnifierFlag);
                    }

                    break;
                case SDLK_SPACE:
                    plyrCxt.pauseFlag = !plyrCxt.pauseFlag;
                    break;
                case SDLK_ESCAPE:
                    plyrCxt.exitFlag = true;
                    return -1;
            }
        } else if (e->type == SDL_MOUSEBUTTONDOWN) {
            plyrCxt.dragSeparatorFlag = 1;

            if (prevFrameLR && prevFrameSR) {
                viewer.render(&plyrCxt.sdlCxt,
                              decoder.mixFrames(prevFrameLR, prevFrameSR,
                                                plyrCxt.sdlCxt.windowWidth, plyrCxt.sdlCxt.windowHeight,
                                                plyrCxt.sdlCxt.sepX),
                              plyrCxt.magnifierFlag);
            }
        } else if (e->type == SDL_MOUSEBUTTONUP) {
            plyrCxt.dragSeparatorFlag = 0;
        } else if (e->type == SDL_MOUSEMOTION) {
            viewer.updateSeparatorColor(&plyrCxt.sdlCxt);

            if (prevFrameLR && prevFrameSR) {
                viewer.render(&plyrCxt.sdlCxt,
                              decoder.mixFrames(prevFrameLR, prevFrameSR,
                                                plyrCxt.sdlCxt.windowWidth, plyrCxt.sdlCxt.windowHeight,
                                                plyrCxt.sdlCxt.sepX),
                              plyrCxt.magnifierFlag);
            }
        } else if (e->type == SDL_QUIT) {
            plyrCxt.exitFlag = true;
        } else if (e->type == BREAK_EVENT) {
            return -1;
        }

        return 0;
    }

    //
    // Threads
    //

    /**
     * Starts the threads used for decoding the lr and sr videos, as well as the refreshing thread.
     *
     * @param plyrCxt   The player context.
     */
    void startThreads(PlayerContext* plyrCxt) {
        FrameThreadDataWrapper* tLR = new FrameThreadDataWrapper;
        FrameThreadDataWrapper* tSR = new FrameThreadDataWrapper;

        tLR->plyrCxt = plyrCxt;
        tLR->ffmCxt = &plyrCxt->ffmCxtLR;
        tLR->queue = &plyrCxt->framesQueueLR;

        tSR->plyrCxt = plyrCxt;
        tSR->ffmCxt = &plyrCxt->ffmCxtSR;
        tSR->queue = &plyrCxt->framesQueueSR;

        SDL_CreateThread(fetchFramesThread, nullptr, (void*) tLR);
        SDL_CreateThread(fetchFramesThread, nullptr, (void*) tSR);

        SDL_Delay(INITIAL_BUFFERING_TIME);

        SDL_CreateThread(refreshThread, nullptr, (void*) plyrCxt);
    }

    /**
     * Refreshes the whole player by pushing a refresh event each frame.
     *
     * @param data  Pointer to the player context.
     *
     * @return      0
     */
    static int refreshThread(void* data) {
        PlayerContext* cont = (PlayerContext*) data;

        cont->exitFlag = 0;
        cont->pauseFlag = 0;

        while (!cont->exitFlag) {
            if (!cont->pauseFlag) {
                SDL_Event event;
                event.type = REFRESH_EVENT;
                SDL_PushEvent(&event);
            }

            SDL_Delay(1000 / cont->ffmCxtLR.fps);
        }

        SDL_Event event;
        event.type = BREAK_EVENT;
        SDL_PushEvent(&event);

        return 0;
    }

    /**
     * Fetches the frames from the FFmpeg lib, then inserts the frames to the provided queues.
     *
     * @param data  Pointer to FrameThreadDataWrapper that holds the required parameters.
     *
     * @return      0
     */
    static int fetchFramesThread(void* data) {
        FrameThreadDataWrapper* cont = (FrameThreadDataWrapper*) data;
        int n = cont->ffmCxt->codecCtx->width * cont->ffmCxt->codecCtx->height;

        int i = 0;
        while (!cont->plyrCxt->exitFlag) {
            int status = Decoder::decodeFrame(cont->ffmCxt);

            if (status == 1)
                continue;
            else if (status == -1)
                break;

            FrameWrapper frameContainer;
            frameContainer.pts = av_frame_get_best_effort_timestamp(cont->ffmCxt->frame);

            frameContainer.frame = av_frame_alloc();
            frameContainer.frame->linesize[0] = cont->ffmCxt->frameYUV->linesize[0];
            frameContainer.frame->data[0] = new uint8_t[int(n * 1.5)];

            memcpy(frameContainer.frame->data[0], cont->ffmCxt->frameYUV->data[0], n * 1.5 * sizeof(uint8_t));

            cont->queue->push(frameContainer);
        }

        return 0;
    }
};