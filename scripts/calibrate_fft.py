import sys, numpy as np, cv2
sys.path.insert(0, ".")

def radial_bands(path, label):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    fshift  = np.fft.fftshift(np.fft.fft2(img))
    log_mag = 20.0 * np.log10(np.abs(fshift) + 1e-8)
    cy, cx  = 112, 112
    Y, X    = np.ogrid[:224, :224]
    R       = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(int)
    max_r   = 112
    rad = np.array([log_mag[R==r].mean() if (R==r).any() else 0 for r in range(max_r)])
    low  = rad[:int(0.15*max_r)].mean()
    mid  = rad[int(0.15*max_r):int(0.50*max_r)].mean()
    high = rad[int(0.50*max_r):int(0.75*max_r)].mean()
    print(f"{label}:")
    print(f"  low={low:.2f}  mid={mid:.2f}  high={high:.2f}")
    print(f"  mid_offset_from_low  = {mid-low:.2f}")
    print(f"  high_offset_from_low = {high-low:.2f}")

radial_bands("test_images/test_fake_stylegan2.jpg", "StyleGAN2 FAKE")
radial_bands("test_images/test_real_ffhq.jpg",      "Real FFHQ")
