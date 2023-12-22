import jpeg
import logging
import scipy as sp
import matplotlib.pyplot as plt


def main():
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    image = sp.datasets.face()
    # both the width and the height of the image are multiples of 8,
    # so the last 7 rows and the last 7 columns are removed, in order
    # to demonstrate that the built-in padding feature works
    image = image[:-7, :-7]

    plt.imshow(image)
    plt.savefig("src/original-sample-image.png")
    logging.info("Original image saved.")

    logging.info("Compression initialized with a compression factor of 1.")
    logging.info("Encoding image...")
    encoded_image, padding_size = jpeg.image_encoding(image)

    logging.info("Decoding image...")
    decoded_image = jpeg.image_decoding(encoded_image, padding_size)

    plt.imshow(decoded_image)
    plt.savefig("dst/compressed-sample-image.png")
    logging.info("Compressed image saved.")

    logging.info("Heavy compression initialized with a compression factor of 150.")
    logging.info("Encoding image...")
    encoded_image, padding_size = jpeg.image_encoding(image, compression_factor=150.0)

    logging.info("Decoding image...")
    decoded_image = jpeg.image_decoding(encoded_image, padding_size)

    plt.imshow(decoded_image)
    plt.savefig("dst/heavily-compressed-sample-image.png")
    logging.info("Heavily compressed image saved.")

    logging.info("Searching for an appropriate compression factor, based on the desired MSE of 1.9. This may take a while...")
    compressed_image, compression_factor, mean_squared_error = jpeg.image_compression(image, desired_mean_squared_error=1.9,
                                                                                      initial_compression_factor=1, compression_factor_step=0.9)
    logging.info(f"Achieved mean squared error: {mean_squared_error}")
    logging.info(f"Associated compression factor: {compression_factor}")
    plt.imshow(compressed_image)
    plt.show()

    logging.info("Applying jpeg compression over every frame in the sample video, with a compression factor of 1. This will take a while...")
    jpeg.video_compression("src/original-sample-video.mp4", "dst/compressed-sample-video.avi")
    logging.info("Compressed video saved.")

    logging.info("Applying jpeg compression over every frame in the sample video, with a compression factor of 150. This will take a while...")
    jpeg.video_compression("src/original-sample-video.mp4", "dst/heavily-compressed-sample-video.avi", compression_factor=150.0)
    logging.info("Heavily compressed video saved.")


if __name__ == '__main__':
    main()
