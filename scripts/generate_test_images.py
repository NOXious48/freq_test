import cv2
import numpy as np

np.random.seed(42)

# 1. test_real_ffhq.jpg - smooth natural face-like image
img = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    for j in range(256):
        img[i,j] = [int(180 + 30*np.sin(i/30)*np.cos(j/40)),
                     int(150 + 20*np.sin(i/25)*np.cos(j/35)),
                     int(140 + 25*np.sin(i/20)*np.cos(j/30))]
img = cv2.GaussianBlur(img, (15,15), 5)
cv2.imwrite('test_images/test_real_ffhq.jpg', img)
print('1. test_real_ffhq.jpg')

# 2. test_fake_stylegan2.jpg - GAN periodic artifacts
fake = img.copy().astype(np.float32)
for c in range(3):
    for freq in [16, 32, 48, 64, 80]:
        fake[:,:,c] += 15 * np.sin(np.arange(256)[None,:] * 2 * np.pi * freq / 256)
        fake[:,:,c] += 10 * np.cos(np.arange(256)[:,None] * 2 * np.pi * freq / 256)
fake = np.clip(fake, 0, 255).astype(np.uint8)
cv2.imwrite('test_images/test_fake_stylegan2.jpg', fake)
print('2. test_fake_stylegan2.jpg')

# 3. test_fake_faceswap.jpg - blending boundary artifacts
swap = img.copy().astype(np.float32)
mask = np.zeros((256,256), dtype=np.float32)
cv2.circle(mask, (128,128), 80, 1.0, -1)
mask = cv2.GaussianBlur(mask, (3,3), 1)
inner = np.random.randint(100, 200, (256,256,3)).astype(np.float32)
for c in range(3):
    swap[:,:,c] = swap[:,:,c] * (1-mask) + inner[:,:,c] * mask
swap = np.clip(swap, 0, 255).astype(np.uint8)
cv2.imwrite('test_images/test_fake_faceswap.jpg', swap)
print('3. test_fake_faceswap.jpg')

# 4. test_fake_diffusion.jpg - diffusion model artifacts
diff = img.copy().astype(np.float32)
diff += np.random.randn(256, 256, 3).astype(np.float32) * 20
for freq in [40, 60, 90]:
    diff[:,:,0] += 8 * np.sin(np.arange(256)[None,:] * 2 * np.pi * freq / 256)
diff = np.clip(diff, 0, 255).astype(np.uint8)
cv2.imwrite('test_images/test_fake_diffusion.jpg', diff)
print('4. test_fake_diffusion.jpg')

# 5. test_no_face.jpg - landscape
landscape = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(128):
    landscape[i,:] = [200 + i//3, 180 - i//4, 100]
for i in range(128, 256):
    landscape[i,:] = [80, 120 + (i-128)//4, 60]
landscape = cv2.GaussianBlur(landscape, (11,11), 3)
cv2.imwrite('test_images/test_no_face.jpg', landscape)
print('5. test_no_face.jpg')

print('All base images created')
