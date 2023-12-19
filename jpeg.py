import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt


luminance_quantization_matrix = np.array([
    [16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0],
    [12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0, 55.0],
    [14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0],
    [14.0, 17.0, 22.0, 29.0, 51.0, 87.0, 80.0, 62.0],
    [18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0],
    [24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0, 92.0],
    [49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0],
    [72.0, 92.0, 95.0, 98.0, 112.0, 100.0, 103.0, 99.0]
])


chrominance_quantization_matrix = np.array([
    [17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0],
    [18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0, 99.0],
    [24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0],
    [47.0, 66.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
    [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
    [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
    [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
    [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]
])


def jpeg_image_encoding(image, compression_amount=1.0):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    image = image.astype(np.float64)

    # by subtracting (image axis size % period size) from the period size,
    # the necessary number of pixels to obtain said period size is obtained;
    # modulo that period size must be performed, in order to deal with the
    # case in which the image axis size is a multiple of the period size;
    # this way, 0 is considered the final size of the padding on that
    # particular axis, and not the period size;
    padding_size = ((8 - image.shape[0] % 8) % 8, (8 - image.shape[1] % 8) % 8)
    # take the last row, and repeat that whole axis for a 'padding' number
    # of times, maintaining the desired shape
    height_pad = np.tile(image[-1], (padding_size[0], 1, 1))
    image = np.append(image, height_pad, axis=0)
    # take the last column, reshape it, in order to preserve the axis each
    # set of pixel values is initially on, and repeat the set of pixel values
    # on the second axis for a 'padding' number of times
    width_pad = np.tile(image[:, -1].reshape((768, 1, 3)), (1, padding_size[1], 1))
    image = np.append(image, width_pad, axis=1)

    # center pixel values around 0
    image -= 128

    encoded_image = np.empty(shape=image.shape)
    for i in range(image.shape[0] // 8):
        for j in range(image.shape[1] // 8):
            block = image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8]

            luminance = block[:, :, 0]
            dctn_of_luminance = sp.fft.dctn(luminance)
            dctn_of_luminance = luminance_quantization_matrix * compression_amount * np.round(dctn_of_luminance / luminance_quantization_matrix / compression_amount)
            quantized_luminance = sp.fft.idctn(dctn_of_luminance)

            red_chrominance = block[:, :, 1]
            dctn_of_red_chrominance = sp.fft.dctn(red_chrominance)
            dctn_of_red_chrominance = chrominance_quantization_matrix * compression_amount * np.round(dctn_of_red_chrominance / chrominance_quantization_matrix / compression_amount)
            quantized_red_chrominance = sp.fft.idctn(dctn_of_red_chrominance)

            blue_chrominance = block[:, :, 2]
            dctn_of_blue_chrominance = sp.fft.dctn(blue_chrominance)
            dctn_of_blue_chrominance = chrominance_quantization_matrix * compression_amount * np.round(dctn_of_blue_chrominance / chrominance_quantization_matrix / compression_amount)
            quantized_blue_chrominance = sp.fft.idctn(dctn_of_blue_chrominance)

            encoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 0] = quantized_luminance
            encoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 1] = quantized_red_chrominance
            encoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 2] = quantized_blue_chrominance

    encoded_image += 128
    encoded_image = np.clip(encoded_image, 0, 255)
    # unpad image
    encoded_image = encoded_image[:-padding_size[0], :-padding_size[1]]
    encoded_image = cv2.cvtColor(encoded_image.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    plt.imshow(encoded_image)
    plt.savefig('encoded_image.png')
    plt.show()


# def jpeg_image_decoding(image):



def main():
    image = sp.datasets.face()
    image = image[:-7, :-7]
    plt.imshow(image)
    plt.savefig('original_image.png')
    plt.show()
    jpeg_image_encoding(image, 100)


if __name__ == '__main__':
    main()
