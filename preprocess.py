# preprocess.py
import cv2
import numpy as np
import argparse
import os

def deskew(image):
    """
    Corrects the skew of an image. This is part of the preprocessing
    [cite_start]phase mentioned in the paper[cite: 310].
    """
    # Convert image to grayscale and invert colors
    gray = cv2.bitwise_not(image)
    
    # Threshold the image, setting all foreground pixels to 255 and background to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Grab the coordinates of all foreground pixels
    coords = np.column_stack(np.where(thresh > 0))
    
    # Find the minimum area rotated rectangle that encloses all foreground pixels
    angle = cv2.minAreaRect(coords)[-1]
    
    # The cv2.minAreaRect function returns values in the range [-90, 0);
    # we need to adjust the angle to be upright.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    print(f"[INFO] Deskewing image by {angle:.2f} degrees")
    return rotated

def remove_noise(image):
    """
    Removes noise from the image using a median blur filter.
    [cite_start]This corresponds to the Noise Removal step in the paper[cite: 304, 305].
    """
    print("[INFO] Applying noise removal...")
    return cv2.medianBlur(image, 5)

def binarize(image):
    """
    Converts a grayscale image to a binary (black and white) image.
    [cite_start]This is the Binarization step discussed in the paper[cite: 306].
    We use adaptive thresholding for better results with uneven lighting.
    """
    print("[INFO] Applying binarization...")
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def thin_image(image):
    """
    Applies thinning to the binary image to reduce characters to a single-pixel width.
    [cite_start]This is the Thinning step from the paper, useful for character recognition[cite: 311].
    The input image must be black text on a white background.
    """
    print("[INFO] Applying thinning...")
    # The thinning function requires a black-on-white image, so we invert it
    inverted_image = cv2.bitwise_not(image)
    thinned = cv2.ximgproc.thinning(inverted_image)
    # Invert back to white-on-black for consistency
    return cv2.bitwise_not(thinned)

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Preprocesses a handwriting image for dataset creation.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input handwriting image.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the final processed image.")
    parser.add_argument("-s", "--steps", action="store_true", help="Save images from intermediate processing steps.")
    args = vars(parser.parse_args())

    input_path = args["input"]
    output_path = args["output"]
    save_intermediate = args["steps"]
    
    # Create a directory for intermediate steps if requested
    if save_intermediate:
        if not os.path.exists("intermediate_steps"):
            os.makedirs("intermediate_steps")
        print("[INFO] Intermediate steps will be saved in 'intermediate_steps/' directory.")

    # --- Preprocessing Pipeline ---
    # [cite_start]The methodology involves several stages of cleaning and standardization[cite: 289].
    
    # [cite_start]1. Image Acquisition and Digitization (reading the image file) [cite: 291]
    image = cv2.imread(input_path)
    if image is None:
        print(f"[ERROR] Could not read image from path: {input_path}")
        return

    # 2. Convert to Grayscale (most preprocessing works on single-channel images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if save_intermediate:
        cv2.imwrite("intermediate_steps/1_grayscale.png", gray_image)

    # [cite_start]3. Skew Correction [cite: 310]
    deskewed_image = deskew(gray_image)
    if save_intermediate:
        cv2.imwrite("intermediate_steps/2_deskewed.png", deskewed_image)

    # [cite_start]4. Noise Removal [cite: 304, 305]
    denoised_image = remove_noise(deskewed_image)
    if save_intermediate:
        cv2.imwrite("intermediate_steps/3_denoised.png", denoised_image)

    # [cite_start]5. Binarization (Thresholding) [cite: 306]
    binary_image = binarize(denoised_image)
    if save_intermediate:
        cv2.imwrite("intermediate_steps/4_binarized.png", binary_image)

    # [cite_start]6. Thinning [cite: 311]
    final_image = thin_image(binary_image)
    if save_intermediate:
        cv2.imwrite("intermediate_steps/5_thinned.png", final_image)

    # Save the final processed image
    cv2.imwrite(output_path, final_image)
    print(f"[SUCCESS] Final processed image saved to: {output_path}")

if __name__ == "__main__":
    main()