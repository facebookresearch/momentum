/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/rasterizer/text_rasterizer.h"

#include <algorithm>
#include <cstdint>

namespace momentum::rasterizer {

namespace {

constexpr uint32_t kTextureWidth = 144;
constexpr uint32_t kTextureHeight = 240;
constexpr uint32_t kNumCharsWidth = 16;
constexpr uint32_t kNumCharsHeight = 16;
constexpr uint32_t kPadding = 1;
constexpr uint32_t kCharWidthInImage = kTextureWidth / kNumCharsWidth;
constexpr uint32_t kCharHeightInImage = kTextureHeight / kNumCharsHeight;
constexpr uint32_t kCharWidth = kCharWidthInImage - 2 * kPadding;
constexpr uint32_t kCharHeight = kCharHeightInImage - 2 * kPadding;

// Proggy clean font from https://github.com/bluescan/proggyfonts
// License is MIT license, see https://github.com/bluescan/proggyfonts/blob/master/LICENSE
const struct {
  uint32_t width;
  uint32_t height;
  std::array<uint8_t, 4320> pixelData;
} fontPixmap = {
    144,
    240,
    {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     16,  0,   0,   0,   0,   6,   2,   0,   8,   0,   32,  0,   0,   0,   0,   0,   0,   0,   16,
     0,   192, 64,  129, 5,   2,   0,   8,   48,  32,  0,   0,   0,   0,   0,   0,   0,   16,  128,
     64,  64,  129, 5,   2,   0,   8,   16,  32,  0,   0,   0,   0,   0,   0,   0,   16,  224, 64,
     0,   128, 5,   2,   0,   8,   16,  32,  0,   0,   0,   0,   0,   0,   0,   252, 240, 64,  0,
     128, 133, 31,  63,  14,  16,  224, 241, 193, 3,   0,   0,   0,   0,   16,  240, 64,  0,   0,
     5,   0,   4,   8,   16,  32,  0,   65,  0,   0,   0,   0,   0,   16,  192, 64,  0,   0,   5,
     0,   4,   8,   16,  32,  128, 0,   0,   0,   0,   0,   0,   16,  128, 64,  64,  1,   5,   0,
     4,   8,   16,  32,  0,   0,   0,   0,   0,   0,   0,   16,  0,   192, 0,   0,   5,   0,   4,
     8,   16,  32,  0,   0,   0,   0,   0,   0,   0,   16,  0,   0,   0,   0,   0,   0,   4,   8,
     0,   32,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   160, 0,   0,   0,   0,   0,   8,   32,  16,  0,   0,
     0,   0,   0,   0,   32,  0,   32,  160, 128, 2,   129, 8,   6,   8,   16,  32,  0,   0,   0,
     0,   0,   0,   32,  0,   32,  160, 128, 130, 71,  5,   9,   8,   16,  32,  0,   0,   0,   0,
     0,   0,   16,  0,   32,  0,   224, 71,  65,  5,   9,   0,   8,   64,  64,  128, 0,   0,   0,
     0,   16,  0,   32,  0,   64,  65,  129, 2,   38,  0,   8,   64,  80,  129, 0,   0,   0,   0,
     8,   0,   32,  0,   64,  129, 3,   10,  41,  0,   8,   64,  224, 224, 3,   128, 15,  0,   8,
     0,   32,  0,   240, 3,   5,   21,  17,  0,   8,   64,  80,  129, 0,   0,   0,   0,   4,   0,
     0,   0,   160, 0,   5,   21,  17,  0,   8,   64,  64,  128, 128, 0,   0,   2,   4,   0,   32,
     0,   160, 192, 131, 8,   46,  0,   16,  32,  0,   0,   128, 0,   0,   2,   2,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   16,  32,  0,   0,   128, 0,   0,   0,   2,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   32,  16,  0,   0,   64,  0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   56,  32,  224, 192, 1,   132, 15,  12,  62,  56,
     112, 0,   0,   0,   0,   0,   0,   28,  68,  48,  16,  33,  2,   134, 0,   2,   32,  68,  136,
     0,   0,   0,   0,   0,   0,   34,  68,  40,  0,   1,   2,   133, 0,   1,   16,  68,  136, 64,
     64,  0,   6,   0,   3,   32,  84,  32,  128, 128, 129, 132, 7,   15,  16,  56,  136, 64,  64,
     128, 129, 31,  12,  16,  84,  32,  64,  0,   66,  4,   8,   17,  8,   68,  240, 0,   0,   96,
     0,   0,   48,  8,   68,  32,  32,  0,   194, 15,  8,   17,  8,   68,  128, 0,   0,   128, 129,
     31,  12,  8,   68,  32,  16,  32,  2,   132, 8,   17,  4,   68,  64,  64,  64,  0,   6,   0,
     3,   0,   56,  248, 240, 193, 1,   4,   7,   14,  4,   56,  48,  64,  64,  0,   0,   0,   0,
     8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   64,  0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   32,  0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   56,  96,  240, 128, 195, 131,
     15,  31,  56,  132, 112, 224, 32,  68,  192, 24,  35,  24,  68,  96,  16,  65,  68,  132, 0,
     1,   68,  132, 32,  128, 32,  66,  192, 24,  35,  36,  178, 144, 16,  33,  64,  136, 0,   1,
     2,   132, 32,  128, 32,  65,  64,  21,  37,  66,  170, 144, 240, 33,  64,  136, 7,   15,  2,
     252, 32,  128, 160, 64,  64,  21,  37,  66,  170, 240, 16,  34,  64,  136, 0,   1,   114, 132,
     32,  128, 224, 64,  64,  18,  41,  66,  114, 8,   17,  34,  64,  136, 0,   1,   66,  132, 32,
     128, 32,  65,  64,  18,  41,  66,  4,   8,   17,  66,  68,  132, 0,   1,   68,  132, 32,  128,
     32,  66,  64,  16,  49,  36,  120, 8,   241, 129, 195, 131, 15,  1,   56,  132, 112, 112, 32,
     196, 71,  16,  49,  24,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   192, 65,  0,   7,   4,   0,   60,  96,
     240, 192, 227, 143, 144, 32,  65,  132, 4,   241, 67,  64,  0,   4,   4,   0,   68,  144, 16,
     33,  4,   129, 144, 32,  73,  132, 4,   1,   66,  128, 0,   4,   10,  0,   68,  8,   17,  33,
     0,   129, 16,  17,  73,  72,  136, 0,   65,  128, 0,   4,   10,  0,   68,  8,   17,  193, 0,
     129, 16,  17,  85,  48,  80,  128, 64,  0,   1,   4,   17,  0,   60,  8,   241, 0,   3,   129,
     16,  10,  85,  48,  32,  64,  64,  0,   1,   4,   17,  0,   4,   8,   145, 0,   4,   129, 16,
     10,  54,  72,  32,  32,  64,  0,   2,   4,   0,   0,   4,   144, 16,  33,  4,   129, 16,  4,
     34,  132, 32,  16,  64,  0,   2,   4,   0,   0,   4,   96,  17,  194, 3,   1,   15,  4,   34,
     132, 32,  240, 67,  0,   4,   4,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   64,  0,   4,   4,   0,   127, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   192, 1,   0,   7,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   8,   0,   16,  0,   0,   4,   0,   28,  0,   4,   32,  128, 32,  128, 1,   0,
     0,   0,   16,  0,   16,  0,   0,   4,   0,   2,   0,   4,   0,   0,   32,  0,   1,   0,   0,
     0,   0,   0,   16,  0,   0,   4,   0,   2,   0,   4,   0,   0,   32,  0,   1,   0,   0,   0,
     0,   112, 240, 192, 129, 7,   7,   15,  60,  60,  48,  192, 32,  2,   193, 13,  15,  28,  0,
     128, 16,  33,  66,  132, 8,   2,   34,  68,  32,  128, 32,  1,   65,  18,  17,  34,  0,   240,
     16,  33,  64,  132, 15,  2,   34,  68,  32,  128, 160, 0,   65,  18,  17,  34,  0,   136, 16,
     33,  64,  132, 0,   2,   34,  68,  32,  128, 224, 0,   65,  18,  17,  34,  0,   136, 16,  33,
     66,  132, 8,   2,   34,  68,  32,  128, 32,  1,   65,  18,  17,  34,  0,   240, 240, 192, 129,
     7,   7,   2,   60,  68,  32,  128, 32,  2,   65,  18,  17,  28,  0,   0,   0,   0,   0,   0,
     0,   0,   32,  0,   0,   128, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   32,  0,   0,   112, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     28,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   3,   129, 1,   0,   0,   0,   0,   0,   0,   128, 0,   0,   0,   0,   0,   0,   0,   128,
     0,   1,   2,   0,   0,   0,   0,   0,   0,   128, 0,   0,   0,   0,   0,   0,   0,   128, 0,
     1,   2,   0,   0,   60,  240, 208, 192, 131, 135, 8,   17,  65,  68,  136, 240, 129, 0,   1,
     2,   0,   0,   68,  136, 48,  33,  128, 128, 8,   17,  73,  40,  136, 0,   129, 0,   1,   2,
     39,  0,   68,  136, 16,  192, 128, 128, 8,   10,  73,  16,  136, 128, 96,  0,   1,   140, 28,
     0,   68,  136, 16,  0,   129, 128, 8,   10,  85,  16,  136, 64,  128, 0,   1,   2,   0,   0,
     68,  136, 16,  0,   130, 128, 8,   4,   54,  40,  136, 32,  128, 0,   1,   2,   0,   0,   60,
     240, 16,  224, 1,   7,   15,  4,   34,  68,  240, 240, 129, 0,   1,   2,   0,   0,   4,   128,
     0,   0,   0,   0,   0,   0,   0,   0,   128, 0,   128, 0,   1,   2,   0,   0,   4,   128, 0,
     0,   0,   0,   0,   0,   0,   0,   128, 0,   0,   3,   129, 1,   0,   0,   4,   128, 0,   0,
     0,   0,   0,   0,   0,   0,   112, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   8,
     8,   0,   0,   0,   0,   0,   0,   0,   0,   112, 0,   0,   0,   0,   0,   0,   4,   8,   24,
     0,   0,   0,   0,   0,   0,   0,   0,   136, 0,   0,   0,   0,   0,   0,   4,   8,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   60,  0,   0,   0,   0,   0,   0,   31,  62,  0,   0,   0,
     0,   0,   0,   0,   0,   0,   8,   0,   0,   0,   0,   0,   0,   4,   8,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   60,  0,   0,   0,   0,   0,   0,   4,   8,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   136, 0,   0,   0,   0,   0,   0,   4,   8,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   112, 0,   0,   0,   0,   0,   0,   4,   62,  0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   8,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   4,   8,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   4,   8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   224, 1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   127,
     0,   0,   0,   0,   0,   0,   0,   4,   28,  40,  0,   0,   0,   0,   0,   0,   0,   0,   0,
     32,  0,   128, 1,   64,  16,  4,   34,  0,   248, 96,  0,   0,   0,   0,   31,  0,   0,   0,
     64,  64,  32,  72,  16,  4,   2,   0,   4,   129, 0,   0,   0,   128, 32,  0,   0,   32,  224,
     64,  192, 135, 8,   4,   12,  0,   100, 225, 128, 2,   0,   128, 38,  0,   0,   32,  80,  225,
     65,  4,   5,   4,   20,  0,   20,  145, 64,  1,   0,   128, 42,  0,   0,   32,  80,  64,  64,
     4,   2,   0,   24,  0,   20,  225, 160, 192, 3,   128, 38,  0,   0,   32,  80,  64,  64,  132,
     15,  4,   32,  0,   100, 1,   64,  1,   2,   128, 42,  0,   0,   32,  80,  33,  192, 7,   2,
     4,   34,  0,   4,   1,   128, 2,   2,   128, 32,  0,   0,   32,  224, 224, 35,  8,   2,   4,
     28,  0,   248, 0,   0,   0,   2,   0,   31,  0,   0,   0,   64,  0,   0,   0,   0,   4,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     4,   8,   16,  0,   24,  0,   112, 224, 0,   2,   0,   0,   0,   0,   16,  96,  0,   64,  132,
     136, 17,  0,   36,  0,   128, 0,   1,   1,   0,   62,  0,   0,   24,  144, 0,   64,  130, 4,
     10,  8,   36,  32,  64,  192, 0,   0,   0,   23,  0,   0,   16,  144, 0,   64,  130, 4,   11,
     0,   24,  32,  32,  0,   1,   128, 8,   23,  0,   0,   16,  144, 160, 64,  129, 2,   6,   8,
     0,   248, 240, 224, 0,   128, 8,   22,  28,  0,   56,  96,  64,  1,   1,   142, 5,   8,   0,
     32,  0,   0,   0,   128, 8,   20,  28,  0,   0,   0,   128, 130, 4,   17,  18,  4,   0,   32,
     0,   0,   0,   128, 8,   20,  28,  0,   0,   0,   64,  129, 6,   9,   26,  2,   0,   0,   0,
     0,   0,   128, 9,   20,  0,   0,   0,   0,   160, 64,  143, 4,   61,  34,  0,   248, 0,   0,
     0,   128, 22,  20,  0,   0,   0,   0,   0,   64,  132, 28,  17,  28,  0,   0,   0,   0,   0,
     128, 0,   20,  0,   48,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   64,
     0,   0,   0,   32,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   16,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   16,  64,  192, 128, 2,   0,   6,   0,   0,   8,
     64,  64,  0,   0,   1,   4,   4,   0,   32,  32,  32,  65,  129, 4,   9,   0,   0,   16,  32,
     160, 64,  1,   2,   2,   10,  20,  0,   0,   0,   0,   0,   0,   9,   60,  56,  0,   0,   0,
     0,   0,   0,   0,   0,   0,   48,  96,  192, 128, 1,   3,   6,   10,  68,  124, 248, 240, 225,
     131, 3,   7,   14,  28,  72,  144, 32,  65,  130, 4,   9,   10,  2,   4,   8,   16,  32,  0,
     1,   2,   4,   8,   72,  144, 32,  65,  130, 4,   9,   25,  2,   4,   8,   16,  32,  0,   1,
     2,   4,   8,   120, 240, 224, 193, 131, 7,   15,  15,  2,   60,  120, 240, 224, 1,   1,   2,
     4,   8,   132, 8,   17,  34,  68,  136, 144, 8,   2,   4,   8,   16,  32,  0,   1,   2,   4,
     8,   132, 8,   17,  34,  68,  136, 144, 8,   68,  4,   8,   16,  32,  0,   1,   2,   4,   8,
     132, 8,   17,  34,  68,  136, 144, 56,  56,  124, 248, 240, 225, 131, 3,   7,   14,  28,  0,
     0,   0,   0,   0,   0,   0,   0,   16,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   16,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   160, 64,  0,   1,   3,
     10,  0,   0,   0,   32,  128, 128, 1,   0,   4,   0,   0,   0,   80,  128, 128, 128, 4,   5,
     18,  0,   0,   64,  64,  64,  130, 4,   2,   0,   28,  0,   0,   0,   0,   0,   0,   0,   0,
     0,   176, 0,   0,   0,   0,   0,   0,   1,   34,  60,  24,  193, 128, 1,   3,   6,   12,  0,
     72,  8,   17,  34,  68,  72,  16,  1,   34,  68,  40,  33,  65,  130, 4,   9,   18,  34,  164,
     8,   17,  34,  68,  72,  16,  15,  30,  132, 40,  17,  34,  68,  136, 16,  33,  20,  164, 8,
     17,  34,  68,  136, 8,   17,  34,  158, 72,  17,  34,  68,  136, 16,  33,  8,   148, 8,   17,
     34,  68,  8,   5,   17,  66,  132, 72,  17,  34,  68,  136, 16,  33,  20,  148, 8,   17,  34,
     68,  8,   2,   15,  66,  68,  136, 33,  65,  130, 4,   9,   18,  34,  72,  8,   17,  34,  68,
     8,   2,   1,   66,  60,  136, 193, 128, 1,   3,   6,   12,  0,   52,  240, 224, 193, 131, 7,
     2,   1,   62,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   16,  64,  64,
     128, 2,   0,   5,   0,   0,   8,   32,  64,  0,   128, 0,   2,   4,   20,  32,  32,  160, 64,
     129, 2,   2,   0,   0,   16,  16,  160, 64,  1,   1,   1,   10,  0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   56,  112, 224, 192, 129, 3,
     7,   27,  28,  56,  112, 224, 192, 129, 1,   3,   6,   12,  64,  128, 0,   1,   2,   4,   8,
     36,  34,  68,  136, 16,  33,  2,   1,   2,   4,   8,   120, 240, 224, 193, 131, 7,   15,  63,
     2,   124, 248, 240, 225, 3,   1,   2,   4,   8,   68,  136, 16,  33,  66,  132, 136, 4,   2,
     4,   8,   16,  32,  0,   1,   2,   4,   8,   68,  136, 16,  33,  66,  132, 136, 36,  34,  68,
     136, 16,  33,  2,   1,   2,   4,   8,   120, 240, 224, 193, 131, 7,   15,  31,  28,  56,  112,
     224, 192, 1,   1,   2,   4,   8,   0,   0,   0,   0,   0,   0,   0,   0,   8,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   88,  80,  32,  0,   1,   1,   10,  0,   0,   0,   16,  64,  128, 0,   0,   4,   0,   0,
     32,  40,  64,  128, 128, 2,   5,   10,  0,   0,   32,  32,  64,  129, 2,   2,   1,   20,  80,
     0,   0,   0,   0,   0,   0,   0,   0,   64,  0,   0,   0,   0,   0,   0,   1,   0,   64,  120,
     224, 192, 129, 3,   7,   14,  8,   56,  136, 16,  33,  66,  132, 8,   15,  34,  120, 136, 16,
     33,  66,  132, 8,   17,  0,   100, 136, 16,  33,  66,  132, 8,   17,  34,  68,  136, 16,  33,
     66,  132, 8,   17,  62,  84,  136, 16,  33,  66,  132, 8,   17,  34,  68,  136, 16,  33,  66,
     132, 8,   17,  0,   84,  136, 16,  33,  66,  132, 8,   17,  34,  68,  136, 16,  33,  66,  132,
     8,   17,  8,   76,  136, 16,  33,  66,  132, 8,   17,  34,  56,  136, 224, 192, 129, 3,   7,
     14,  0,   56,  240, 224, 193, 131, 7,   15,  15,  60,  0,   0,   0,   0,   0,   0,   0,   0,
     0,   4,   0,   0,   0,   0,   0,   8,   1,   32,  0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   8,   1,   32,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   7,   0,   28,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0}};

bool getPixelFromBitmap(uint32_t x, uint32_t y) {
  const size_t pixelIndex = y * kTextureWidth + x;
  const size_t byteIndex = pixelIndex / 8;
  const size_t bitIndex = pixelIndex - byteIndex * 8;
  return ((fontPixmap.pixelData[byteIndex] & (static_cast<uint8_t>(1) << bitIndex)) != 0);
}

void renderCharacter(
    char c,
    int imageX,
    int imageY,
    int textScale,
    float depth,
    const Eigen::Vector3f& color,
    Span2f zBuffer,
    Span3f rgbBuffer) {
  const uint32_t charCode = static_cast<uint8_t>(c);
  const uint32_t charRow = charCode / kNumCharsWidth;
  const uint32_t charCol = charCode - (charRow * kNumCharsWidth);

  const uint32_t minTextureX = charCol * kCharWidthInImage + kPadding;
  const uint32_t minTextureY = charRow * kCharHeightInImage + kPadding;

  const int imageHeight = static_cast<int>(zBuffer.extent(0));
  const int imageWidth = static_cast<int>(zBuffer.extent(1));

  for (uint32_t charY = 0; charY < kCharHeight; ++charY) {
    for (uint32_t charX = 0; charX < kCharWidth; ++charX) {
      const uint32_t texX = minTextureX + charX;
      const uint32_t texY = minTextureY + charY;

      if (getPixelFromBitmap(texX, texY)) {
        for (int sy = 0; sy < textScale; ++sy) {
          for (int sx = 0; sx < textScale; ++sx) {
            const int pixelX = imageX + static_cast<int>(charX) * textScale + sx;
            const int pixelY = imageY + static_cast<int>(charY) * textScale + sy;

            if (pixelX >= 0 && pixelX < imageWidth && pixelY >= 0 && pixelY < imageHeight) {
              if (depth <= zBuffer(pixelY, pixelX)) {
                zBuffer(pixelY, pixelX) = depth;
                if (!rgbBuffer.empty()) {
                  rgbBuffer(pixelY, pixelX, 0) = color.x();
                  rgbBuffer(pixelY, pixelX, 1) = color.y();
                  rgbBuffer(pixelY, pixelX, 2) = color.z();
                }
              }
            }
          }
        }
      }
    }
  }
}

void renderCharacter2D(
    char c,
    int imageX,
    int imageY,
    int textScale,
    const Eigen::Vector3f& color,
    Span3f rgbBuffer,
    Span2f zBuffer) {
  const uint32_t charCode = static_cast<uint8_t>(c);
  const uint32_t charRow = charCode / kNumCharsWidth;
  const uint32_t charCol = charCode - (charRow * kNumCharsWidth);

  const uint32_t minTextureX = charCol * kCharWidthInImage + kPadding;
  const uint32_t minTextureY = charRow * kCharHeightInImage + kPadding;

  const int imageHeight = static_cast<int>(rgbBuffer.extent(0));
  const int imageWidth = static_cast<int>(rgbBuffer.extent(1));

  for (uint32_t charY = 0; charY < kCharHeight; ++charY) {
    for (uint32_t charX = 0; charX < kCharWidth; ++charX) {
      const uint32_t texX = minTextureX + charX;
      const uint32_t texY = minTextureY + charY;

      if (getPixelFromBitmap(texX, texY)) {
        for (int sy = 0; sy < textScale; ++sy) {
          for (int sx = 0; sx < textScale; ++sx) {
            const int pixelX = imageX + static_cast<int>(charX) * textScale + sx;
            const int pixelY = imageY + static_cast<int>(charY) * textScale + sy;

            if (pixelX >= 0 && pixelX < imageWidth && pixelY >= 0 && pixelY < imageHeight) {
              rgbBuffer(pixelY, pixelX, 0) = color.x();
              rgbBuffer(pixelY, pixelX, 1) = color.y();
              rgbBuffer(pixelY, pixelX, 2) = color.z();

              if (!zBuffer.empty()) {
                zBuffer(pixelY, pixelX) = 0.0f;
              }
            }
          }
        }
      }
    }
  }
}

} // namespace

void rasterizeText(
    gsl::span<const Eigen::Vector3f> positionsWorld,
    gsl::span<const std::string> texts,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    int textScale,
    Span2f zBuffer,
    Span3f rgbBuffer,
    float depthOffset,
    const Eigen::Vector2f& imageOffset,
    HorizontalAlignment horizontalAlignment,
    VerticalAlignment verticalAlignment) {
  MT_THROW_IF(positionsWorld.size() != texts.size(), "Number of positions and texts must be equal");

  const int scaledCharWidth = static_cast<int>(kCharWidth) * textScale;
  const int scaledCharHeight = static_cast<int>(kCharHeight) * textScale;

  const Eigen::Affine3f worldFromEye = camera.worldFromEye();
  const Eigen::Affine3f eyeFromWorld = worldFromEye.inverse();
  const auto& intrinsicsModel = camera.intrinsicsModel();

  for (size_t i = 0; i < positionsWorld.size(); ++i) {
    const Eigen::Vector3f worldPos = (modelMatrix * positionsWorld[i].homogeneous()).head<3>();
    const Eigen::Vector3f eyePos = eyeFromWorld * worldPos;

    if (eyePos.z() <= nearClip) {
      continue;
    }

    auto [imagePos, valid] = intrinsicsModel->project(eyePos);
    if (!valid) {
      continue;
    }

    imagePos.z() += depthOffset;

    const std::string& text = texts[i];
    const int textWidth = static_cast<int>(text.length()) * scaledCharWidth;

    int offsetX = 0;
    switch (horizontalAlignment) {
      case HorizontalAlignment::Left:
        offsetX = 0;
        break;
      case HorizontalAlignment::Center:
        offsetX = -textWidth / 2;
        break;
      case HorizontalAlignment::Right:
        offsetX = -textWidth;
        break;
    }

    int offsetY = 0;
    switch (verticalAlignment) {
      case VerticalAlignment::Top:
        offsetY = 0;
        break;
      case VerticalAlignment::Center:
        offsetY = -scaledCharHeight / 2;
        break;
      case VerticalAlignment::Bottom:
        offsetY = -scaledCharHeight;
        break;
    }

    int currentX = static_cast<int>(imagePos.x() + imageOffset.x()) + offsetX;
    int currentY = static_cast<int>(imagePos.y() + imageOffset.y()) + offsetY;

    for (char c : text) {
      renderCharacter(c, currentX, currentY, textScale, imagePos.z(), color, zBuffer, rgbBuffer);
      currentX += scaledCharWidth;
    }
  }
}

void rasterizeText2D(
    gsl::span<const Eigen::Vector2f> positionsImage,
    gsl::span<const std::string> texts,
    const Eigen::Vector3f& color,
    int textScale,
    Span3f rgbBuffer,
    Span2f zBuffer,
    const Eigen::Vector2f& imageOffset,
    HorizontalAlignment horizontalAlignment,
    VerticalAlignment verticalAlignment) {
  if (positionsImage.size() != texts.size()) {
    return;
  }

  const int scaledCharWidth = static_cast<int>(kCharWidth) * textScale;
  const int scaledCharHeight = static_cast<int>(kCharHeight) * textScale;

  for (size_t i = 0; i < positionsImage.size(); ++i) {
    const std::string& text = texts[i];
    const int textWidth = static_cast<int>(text.length()) * scaledCharWidth;

    int offsetX = 0;
    switch (horizontalAlignment) {
      case HorizontalAlignment::Left:
        offsetX = 0;
        break;
      case HorizontalAlignment::Center:
        offsetX = -textWidth / 2;
        break;
      case HorizontalAlignment::Right:
        offsetX = -textWidth;
        break;
    }

    int offsetY = 0;
    switch (verticalAlignment) {
      case VerticalAlignment::Top:
        offsetY = 0;
        break;
      case VerticalAlignment::Center:
        offsetY = -scaledCharHeight / 2;
        break;
      case VerticalAlignment::Bottom:
        offsetY = -scaledCharHeight;
        break;
    }

    int currentX = static_cast<int>(positionsImage[i].x() + imageOffset.x()) + offsetX;
    int currentY = static_cast<int>(positionsImage[i].y() + imageOffset.y()) + offsetY;

    for (char c : text) {
      renderCharacter2D(c, currentX, currentY, textScale, color, rgbBuffer, zBuffer);
      currentX += scaledCharWidth;
    }
  }
}

} // namespace momentum::rasterizer
