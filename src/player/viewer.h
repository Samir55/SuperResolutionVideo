#pragma once

#include "wrappers.h"
#include "utils/constants.h"

class Viewer {
public:

    /**
     * Sets up the SDL env.
     *
     * @param sdlCxt    The SDL context.
     * @param height    The video height.
     * @param width     The video width.
     *
     * @return          0 if success, -1 otherwise.
     */
    int setupSDL(SDLContext& sdlCxt, int height, int width) {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
            cerr << "Could not initialize SDL - " << SDL_GetError() << endl;
            return -1;
        }

        sdlCxt.window = SDL_CreateWindow(WINDOW_TITLE,
                                         SDL_WINDOWPOS_UNDEFINED,
                                         SDL_WINDOWPOS_UNDEFINED,
                                         width, height,
                                         SDL_WINDOW_OPENGL);
        if (!sdlCxt.window) {
            cerr << "SDL: could not create window - exiting:" << SDL_GetError() << endl;
            return -1;
        }

        sdlCxt.sdlRenderer = SDL_CreateRenderer(sdlCxt.window, -1, 0);

        sdlCxt.sdlTexture = SDL_CreateTexture(sdlCxt.sdlRenderer,
                                              SDL_PIXELFORMAT_IYUV,
                                              SDL_TEXTUREACCESS_STREAMING,
                                              width, height);

        sdlCxt.swsMagCtx = sws_getContext(
                MAGNIFIER_WINDOW_RADIUS / MAGNIFIER_SCALE_FACTOR,
                MAGNIFIER_WINDOW_RADIUS / MAGNIFIER_SCALE_FACTOR,
                AV_PIX_FMT_YUV420P,
                MAGNIFIER_WINDOW_RADIUS,
                MAGNIFIER_WINDOW_RADIUS,
                AV_PIX_FMT_YUV420P,
                SWS_BICUBIC,
                nullptr,
                nullptr,
                nullptr);

        sdlCxt.sepX = width / 2; // Initial separator location

        if (ENABLE_KEYMAP_DIALOG)
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION,
                                     "Hive",
                                     "Press Space to Play/Pause.\nPress Esc to Exit.\nPress M to Show/Hide the Magnifier.",
                                     sdlCxt.window);

        return 0;
    }

    /**
     * Updates the state variables of the viewer.
     *
     * @param sdlCxt                The SDL context.
     * @param dragSeparatorFlag     Flag indicates if the user is dragging the separator.
     */
    void updateState(SDLContext* sdlCxt, bool dragSeparatorFlag) {
        SDL_GetMouseState(&sdlCxt->mouseX, &sdlCxt->mouseY);
        SDL_GetGlobalMouseState(&sdlCxt->globalMouseX, &sdlCxt->globalMouseY);
        SDL_GetWindowPosition(sdlCxt->window, &sdlCxt->windowX, &sdlCxt->windowY);
        SDL_GetWindowSize(sdlCxt->window, &sdlCxt->windowWidth, &sdlCxt->windowHeight);
        if (dragSeparatorFlag) SDL_GetMouseState(&sdlCxt->sepX, &sdlCxt->sepY);
    }

    /**
     * Renders the given frame.
     *
     * @param sdlCxt            The SDL context.
     * @param frame             The frame to be rendered.
     * @param magnifierFlag     Flags the magnifier enabling state.
     */
    void render(SDLContext* sdlCxt, AVFrame frame, bool magnifierFlag) {
        magnify(sdlCxt, &frame, magnifierFlag);

        drawSeparator(sdlCxt, &frame, magnifierFlag);

        SDL_UpdateTexture(sdlCxt->sdlTexture, nullptr, frame.data[0], frame.linesize[0]);
        SDL_RenderClear(sdlCxt->sdlRenderer);
        SDL_RenderCopy(sdlCxt->sdlRenderer, sdlCxt->sdlTexture, nullptr, nullptr);

        drawMagnifier(sdlCxt, magnifierFlag);

        SDL_RenderPresent(sdlCxt->sdlRenderer);
    }

    /**
     * Magnifies the pointed-to spot and adds the new magnified spot the given frame.
     *
     * @param sdlCxt            The player context.
     * @param frame             The frame to magnify.
     * @param magnifierFlag     Flags the magnifier enabling state.
     */
    void magnify(SDLContext* sdlCxt, AVFrame* frame, bool magnifierFlag) {
        if (!magnifierFlag) return;

        int width = sdlCxt->windowWidth;
        int height = sdlCxt->windowHeight;
        int total = width * height;

        if (isMouseOutside(sdlCxt))
            return;

        // Small frame to magnify
        AVFrame* tmp = av_frame_alloc();

        int r = MAGNIFIER_WINDOW_RADIUS / MAGNIFIER_SCALE_FACTOR;
        int t = r * r;
        int n = avpicture_get_size(AV_PIX_FMT_YUV420P, r, r);

        uint8_t* tmpBuf = (uint8_t*) av_malloc(n * sizeof(uint8_t));
        avpicture_fill((AVPicture*) tmp, tmpBuf, AV_PIX_FMT_YUV420P, r, r);

        for (int i = sdlCxt->mouseY - r / 2, k = 0; i < sdlCxt->mouseY + r / 2; ++i, ++k) {
            for (int j = sdlCxt->mouseX - r / 2, l = 0; j < sdlCxt->mouseX + r / 2; ++j, ++l) {
                tmp->data[0][k * r + l] =
                        frame->data[0][i * width + j];
                tmp->data[0][(k / 2) * (r / 2) + (l / 2) + t] =
                        frame->data[0][(i / 2) * (width / 2) + (j / 2) + total];
                tmp->data[0][(k / 2) * (r / 2) + (l / 2) + t + (t / 4)] =
                        frame->data[0][(i / 2) * (width / 2) + (j / 2) + total + (total / 4)];
            }
        }

        // Scale up
        AVFrame* tmpScaled = av_frame_alloc();

        int rScaled = MAGNIFIER_WINDOW_RADIUS;
        int nScaled = avpicture_get_size(AV_PIX_FMT_YUV420P, rScaled, rScaled);

        uint8_t* tmpBufScaled = (uint8_t*) av_malloc(nScaled * sizeof(uint8_t));
        avpicture_fill((AVPicture*) tmpScaled, tmpBufScaled, AV_PIX_FMT_YUV420P, rScaled, rScaled);

        sws_scale(sdlCxt->swsMagCtx, (const unsigned char* const*) tmp->data,
                  tmp->linesize, 0,
                  MAGNIFIER_WINDOW_RADIUS / MAGNIFIER_SCALE_FACTOR,
                  tmpScaled->data,
                  tmpScaled->linesize);

        // Apply to the original frame
        for (int i = sdlCxt->mouseY - rScaled / 2, k = 0; i < sdlCxt->mouseY + rScaled / 2; ++i, ++k) {
            for (int j = sdlCxt->mouseX - rScaled / 2, l = 0; j < sdlCxt->mouseX + rScaled / 2; ++j, ++l) {
                if ((sdlCxt->mouseY - i) * (sdlCxt->mouseY - i) + (sdlCxt->mouseX - j) * (sdlCxt->mouseX - j)
                    > MAGNIFIER_WINDOW_RADIUS * MAGNIFIER_WINDOW_RADIUS / 4) { // Outside the circle
                    continue;
                }

                frame->data[0][i * width + j] =
                        tmpScaled->data[0][k * tmpScaled->linesize[0] + l];
                frame->data[0][(i / 2) * (width / 2) + (j / 2) + total] =
                        tmpScaled->data[1][(k / 2) * tmpScaled->linesize[1] + (l / 2)];
                frame->data[0][(i / 2) * (width / 2) + (j / 2) + total + (total / 4)] =
                        tmpScaled->data[2][(k / 2) * tmpScaled->linesize[2] + (l / 2)];
            }
        }
    }

    /**
     * Draws the magnifier frame.
     *
     * @param sdlCxt            The SDL context.
     * @param magnifierFlag     Flags the magnifier enabling state.
     */
    void drawMagnifier(SDLContext* sdlCxt, bool magnifierFlag) {
        SDL_Renderer* sdlRenderer = sdlCxt->sdlRenderer;

        if (!magnifierFlag) {
            SDL_ShowCursor(1);
            return;
        } else {
            SDL_ShowCursor(0);
        }

        if (isMouseOutside(sdlCxt))
            return;

        SDL_SetRenderDrawColor(sdlRenderer, BLACK_R, BLACK_G, BLACK_B, MAGNIFIER_BORDER_ALPHA);

        int i;
        for (i = 0; i < MAGNIFIER_BORDER_THICKNESS / 2; ++i) {
            drawCircle(sdlRenderer, sdlCxt->mouseX, sdlCxt->mouseY, MAGNIFIER_WINDOW_RADIUS / 2 - i);
        }

        SDL_SetRenderDrawColor(sdlRenderer, GRAY_R, GRAY_G, GRAY_B, MAGNIFIER_BORDER_ALPHA);

        for (; i < MAGNIFIER_BORDER_THICKNESS; ++i) {
            drawCircle(sdlRenderer, sdlCxt->mouseX, sdlCxt->mouseY, MAGNIFIER_WINDOW_RADIUS / 2 - i);
        }
    }

    /**
     * Draws the separator on the given frame (should be mixed frame).
     *
     * @param sdlCxt            The SDL context.
     * @param frame             The frame to draw the separator on it.
     * @param magnifierFlag     Flags the magnifier enabling state.
     */
    void drawSeparator(SDLContext* sdlCxt, AVFrame* frame, bool magnifierFlag) {
        int width = sdlCxt->windowWidth;
        int height = sdlCxt->windowHeight;
        int total = width * height;

        for (int i = 0; i < height; ++i) {
            for (int j = sdlCxt->sepX - SEP_WIDTH / 2; j <= sdlCxt->sepX + SEP_WIDTH / 2; ++j) {
                if (magnifierFlag && !isMouseOutside(sdlCxt) &&
                    ((sdlCxt->mouseY - i) * (sdlCxt->mouseY - i) + (sdlCxt->mouseX - j) * (sdlCxt->mouseX - j)
                     <= MAGNIFIER_WINDOW_RADIUS * MAGNIFIER_WINDOW_RADIUS / 4)) {
                    continue;
                }

                uint8_t& ny = frame->data[0][i * width + j];
                uint8_t& nu = frame->data[0][(i / 2) * (width / 2) + (j / 2) + total];
                uint8_t& nv = frame->data[0][(i / 2) * (width / 2) + (j / 2) + total + (total / 4)];

                ny = sdlCxt->sepColorY;
                nu = sdlCxt->sepColorU;
                nv = sdlCxt->sepColorV;
            }
        }
    }

    /**
     * Updates the separator color.
     *
     * @param sdlCxt    The SDL context
     */
    void updateSeparatorColor(SDLContext* sdlCxt) {
        if (sdlCxt->mouseX >= sdlCxt->sepX - SEP_WIDTH / 2 &&
            sdlCxt->mouseX <= sdlCxt->sepX + SEP_WIDTH / 2) {
            sdlCxt->sepColorY = GOLD_Y, sdlCxt->sepColorU = GOLD_U, sdlCxt->sepColorV = GOLD_V;
        } else {
            sdlCxt->sepColorY = RED_Y, sdlCxt->sepColorU = RED_U, sdlCxt->sepColorV = RED_V;
        }
    }

    /**
     * Rasterizes a circle.
     * Used for the magnification circle.
     *
     * @param Renderer  The SDL_Renderer ptr.
     * @param _x        The circle center x.
     * @param _y        The circle center y.
     * @param radius    The circle radius.
     */
    void drawCircle(SDL_Renderer* Renderer, int32_t _x, int32_t _y, int32_t radius) {
        int32_t x = radius - 1;
        int32_t y = 0;
        int32_t tx = 1;
        int32_t ty = 1;
        int32_t err = tx - (radius << 1);

        while (x >= y) {
            //  Each of the following renders an octant of the circle
            SDL_RenderDrawPoint(Renderer, _x + x, _y - y);
            SDL_RenderDrawPoint(Renderer, _x + x, _y + y);
            SDL_RenderDrawPoint(Renderer, _x - x, _y - y);
            SDL_RenderDrawPoint(Renderer, _x - x, _y + y);
            SDL_RenderDrawPoint(Renderer, _x + y, _y - x);
            SDL_RenderDrawPoint(Renderer, _x + y, _y + x);
            SDL_RenderDrawPoint(Renderer, _x - y, _y - x);
            SDL_RenderDrawPoint(Renderer, _x - y, _y + x);

            if (err <= 0) {
                y++;
                err += ty;
                ty += 2;
            } else {
                x--;
                tx += 2;
                err += tx - (radius << 1);
            }
        }
    }

    /**
     * Checks if the mouse is outside the window.
     * Actually it's not the whole window, but the area that we can magnify (window size - magnifier size / 2).
     * Futher than that the magnifier window will contain black pixels!
     *
     * @param sdlCxt    The SDL context.
     *
     * @return          true if mouse outsize, false otherwise.
     */
    bool isMouseOutside(SDLContext* sdlCxt) {
        return sdlCxt->globalMouseX < sdlCxt->windowX + MAGNIFIER_WINDOW_RADIUS / 2 ||
               sdlCxt->globalMouseX > sdlCxt->windowX + sdlCxt->windowWidth - MAGNIFIER_WINDOW_RADIUS / 2 ||
               sdlCxt->globalMouseY < sdlCxt->windowY + MAGNIFIER_WINDOW_RADIUS / 2 ||
               sdlCxt->globalMouseY > sdlCxt->windowY + sdlCxt->windowHeight - MAGNIFIER_WINDOW_RADIUS / 2;
    }
};