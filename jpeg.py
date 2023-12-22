import numpy as np
import scipy as sp
import cv2


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


def image_encoding(image, compression_factor=1.0):
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
    width_pad = np.tile(image[:, -1].reshape((image.shape[0], 1, 3)), (1, padding_size[1], 1))
    image = np.append(image, width_pad, axis=1)

    # center pixel values around 0
    image -= 128

    encoded_image = np.ndarray(image.shape)
    for i in range(image.shape[0] // 8):
        for j in range(image.shape[1] // 8):
            block = image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8]

            luminance = block[:, :, 0]
            dctn_of_luminance = sp.fft.dctn(luminance)
            dctn_of_luminance = luminance_quantization_matrix * compression_factor * np.round(dctn_of_luminance / luminance_quantization_matrix / compression_factor)

            red_chrominance = block[:, :, 1]
            dctn_of_red_chrominance = sp.fft.dctn(red_chrominance)
            dctn_of_red_chrominance = chrominance_quantization_matrix * compression_factor * np.round(dctn_of_red_chrominance / chrominance_quantization_matrix / compression_factor)

            blue_chrominance = block[:, :, 2]
            dctn_of_blue_chrominance = sp.fft.dctn(blue_chrominance)
            dctn_of_blue_chrominance = chrominance_quantization_matrix * compression_factor * np.round(dctn_of_blue_chrominance / chrominance_quantization_matrix / compression_factor)

            encoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 0] = dctn_of_luminance
            encoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 1] = dctn_of_red_chrominance
            encoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 2] = dctn_of_blue_chrominance

    return encoded_image, padding_size


def image_decoding(encoded_image, padding_size):
    decoded_image = np.ndarray(encoded_image.shape)
    for i in range(encoded_image.shape[0] // 8):
        for j in range(encoded_image.shape[1] // 8):
            block = encoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8]

            dctn_of_luminance = block[:, :, 0]
            quantized_luminance = sp.fft.idctn(dctn_of_luminance)

            dctn_of_red_chrominance = block[:, :, 1]
            quantized_red_chrominance = sp.fft.idctn(dctn_of_red_chrominance)

            dctn_of_blue_chrominance = block[:, :, 2]
            quantized_blue_chrominance = sp.fft.idctn(dctn_of_blue_chrominance)

            decoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 0] = quantized_luminance
            decoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 1] = quantized_red_chrominance
            decoded_image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 2] = quantized_blue_chrominance

    # unpad image
    if padding_size[0]:
        decoded_image = decoded_image[:-padding_size[0]]

    if padding_size[1]:
        decoded_image = decoded_image[:, :-padding_size[1]]

    decoded_image += 128
    decoded_image = np.clip(decoded_image, 0, 255)
    decoded_image = decoded_image.astype(np.uint8)
    decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_YCrCb2RGB)

    return decoded_image


def get_mean_squared_error(first_image, second_image):
    return np.sum(np.square(first_image - second_image)) / first_image.size


def image_compression(image, desired_mean_squared_error, initial_compression_factor=1.0, compression_factor_step=0.9):
    compression_factor = initial_compression_factor
    
    encoded_image, padding_size = image_encoding(image, compression_factor)
    decoded_image = image_decoding(encoded_image, padding_size)
    mean_squared_error = get_mean_squared_error(image, decoded_image)
    
    if mean_squared_error < desired_mean_squared_error:
        while mean_squared_error < desired_mean_squared_error:
            compression_factor /= compression_factor_step
            encoded_image, padding_size = image_encoding(image, compression_factor)
            decoded_image = image_decoding(encoded_image, padding_size)
            mean_squared_error = get_mean_squared_error(image, decoded_image)
    elif mean_squared_error > desired_mean_squared_error:
        while mean_squared_error > desired_mean_squared_error:
            compression_factor *= compression_factor_step
            encoded_image, padding_size = image_encoding(image, compression_factor)
            decoded_image = image_decoding(encoded_image, padding_size)
            mean_squared_error = get_mean_squared_error(image, decoded_image)

    return decoded_image, compression_factor, mean_squared_error


def video_compression(source_path, destination_path, compression_factor=1.0):
    video_capture = cv2.VideoCapture(source_path)

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(destination_path, codec, frame_rate, (frame_width, frame_height))

    # once all the frames have been read successfully, the
    # return code will be non-zero and the loop will halt
    while True:
        return_code, frame = video_capture.read()
        if not return_code:
            break

        # opencv loads frames in the BGR color space by default
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encoded_frame, padding_size = image_encoding(frame, compression_factor)
        decoded_frame = image_decoding(encoded_frame, padding_size)
        # the BGR color space is also necessary when writing to a file
        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_RGB2BGR)

        video_writer.write(decoded_frame)

    video_capture.release()
    video_writer.release()
