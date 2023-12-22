# JPEG Encoder-Decoder
## General Description
Basic PoC JPEG module, with support for compression of varying quality, based on a compression factor, and for searching an appropriate compression factor which yields a desired MSE. Images must be provided in the RGB color space.
Encoding stages:
- convert the provided image from RGB to YCrCb
- add padding where necessary
- center values around 0, by subtracting 128
- take each 8x8 pixel block from the image and:
  - extract the luminance and color channels
  - perform DCTN on each of the extracted channels
  - quantize the DCTN using the appropriate quantization matrix
  - insert the channels back into an 8x8 pixel block, which will be stored in an _encoded_ matrix

Decoding stages:
- take each 8x8 pixel block from the encoded matrix and:
  - extract the luminance and color channels
  - perform IDCTN on each of the extracted channels
  - insert the channels back into an 8x8 pixel block, which will be stored in the _decoded_ matrix
- remove the padding where necessary
- add 128
- clip the pixel values between 0 and 255
- convert the decoded image from YCrCb to RGB

There is minimal support for video compression, which works by applying the JPEG compression over each frame.

## Dependencies
Both development and testing were performed using python v3.11.2, and the packages specified in `requirements.txt`.
