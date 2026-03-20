# MFAD_dev/scripts/compress_image.py
import cv2, sys

def compress(input_path: str, quality: int = 50):
    img         = cv2.imread(input_path)
    output_path = input_path.replace(".jpg", f"_q{quality}.jpg")
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    compress(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
