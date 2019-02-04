#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <queue>
#include <vector>

#include "player.h"

int main() {
    // freopen("output.txt", "w", stdout);

    Player player;
    if (player.play("/Users/ibrahimradwan/Desktop/ffmpeg-player/assets/titanic.ts") != 0) {
        cerr << "Error" << endl;
    }

    return 0;
}
