from aicspylibczi import CziFile
import matplotlib.pyplot as plt
import numpy as np

# Gán màu cho từng channel
DEFAULT_COLORS = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']

# Mở file CZI
czi = CziFile("/home/nguyen/Biofilms_git/BioFilmsCZI/Biofilms/image/Bi-Bag-P-P1-3h-4.czi")
root = czi.meta

# Lấy tên channel không trùng
seen = set()
channels = []
for ch in root.findall(".//Channels/Channel"):
    name = ch.attrib.get("Name")
    if name not in seen:
        channels.append(name)
        seen.add(name)

print("Available unique channel names:", channels)

# Khởi tạo ảnh RGB 8-bit như ImageJ
mosaic_shape = czi.read_mosaic(C=0, Z=0, T=0).shape[-2:]
composite_img = np.zeros((mosaic_shape[0], mosaic_shape[1], 3), dtype=np.uint8)

for ch_index, ch_name in enumerate(channels):
    try:
        # Đọc mosaic ảnh thô
        mosaic = np.squeeze(czi.read_mosaic(C=ch_index, Z=0, T=0))

        # Stretch về 8-bit giống ImageJ (0–255)
        norm = ((mosaic - mosaic.min()) / (mosaic.ptp() + 1e-6) * 255).astype(np.uint8)

        # Gán vào kênh màu
        color = DEFAULT_COLORS[ch_index % len(DEFAULT_COLORS)]
        rgb = np.zeros_like(composite_img)
        if color == 'red':
            rgb[..., 0] = norm
        elif color == 'green':
            rgb[..., 1] = norm
        elif color == 'blue':
            rgb[..., 2] = norm
        elif color == 'magenta':
            rgb[..., 0] = norm
            rgb[..., 2] = norm
        elif color == 'cyan':
            rgb[..., 1] = norm
            rgb[..., 2] = norm
        elif color == 'yellow':
            rgb[..., 0] = norm
            rgb[..., 1] = norm

        # Cộng overlay (clipped ở 255)
        composite_img = np.clip(composite_img + rgb, 0, 255)

        # Cũng hiển thị từng channel riêng
        plt.figure()
        plt.imshow(norm, cmap="gray")
        plt.title(f"Channel: {ch_name} ({color})")
        plt.axis("off")

    except Exception as e:
        print(f"⚠️ Lỗi khi load {ch_name} (index {ch_index}): {e}")

# Hiển thị ảnh RGB composite như ImageJ
plt.figure()
plt.imshow(composite_img)
plt.title("Composite (ImageJ-style)")
plt.axis("off")
plt.show()
