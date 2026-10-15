import sys
import cv2
import numpy as np

# =====================================================================
# USER CONFIGURATION
# =====================================================================

IMAGE_PATH = "cactus_heart.jpg"
OUTPUT_PATH = "convolution_result.jpg"

CUSTOM_KERNEL = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# =====================================================================

def main():
    """Loads the image, applies the custom kernel, saves and displays the result."""
    # 1. Load the image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Unable to load image at '{IMAGE_PATH}'. Check the path.")
        sys.exit(1)

    # 2. Apply the 2D convolution
    filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=CUSTOM_KERNEL)

    # 3. Save the output image
    cv2.imwrite(OUTPUT_PATH, filtered_image)
    print(f"Image successfully saved to '{OUTPUT_PATH}'")

    # 4. Display the images
    cv2.imshow("Original Image", image)
    cv2.imshow("Convolution Result", filtered_image)

    # 5. Wait for any key press to close windows and exit
    print("Press any key on the image windows to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
