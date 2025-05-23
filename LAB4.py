import os
import glob
import random
import numpy as np
import cv2
from Anisotropic import anisodiff, anisodiff3
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -----------------------------
# Configuración de parámetros
# -----------------------------
TRAIN_DIR = 'train'
PATCH_SIZE = 32
OVERLAP = PATCH_SIZE // 2
NUM_PATCHES = 200_000
SPLIT = (0.8, 0.1, 0.1)
MODEL_FILENAME = 'unet_anisotropic.h5'
FULL_IMAGE_PATH = 'train/309004.jpg'
DO_TRAIN = False


ANISO_PARAMS = {
    'niter': 50,
    'kappa': 20,
    'gamma': 0.2,
    'step': (1.,1.), 
    'option': 1,
    'ploton': False
}
ANISO3_PARAMS = {
    'niter': 50,
    'kappa': 20,
    'gamma': 0.2,
    'step': (1.,1.,1.),
    'option': 1,
    'ploton': False
}

# -----------------------------
# Funciones auxiliares
# -----------------------------
def load_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"No se pudo cargar imagen: {path}")
    return img.astype(np.float32) / 255.0


def load_image_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"No se pudo cargar imagen: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def preprocess_image(img):
    """
    Aplica filtro anisotrópico usando las funciones del archivo Anisotropic.py.
    Soporta imágenes en 2D (grayscale) y 3D (RGB).
    """
    img_uint = (img * 255).astype(np.float32)
    if img_uint.ndim == 2:
        diff = anisodiff(img_uint, **ANISO_PARAMS)
    elif img_uint.ndim == 3:
        diff = anisodiff3(img_uint, **ANISO3_PARAMS)
    else:
        raise ValueError(f"Formato de imagen no soportado (dim={img_uint.ndim})")
    diff_scaled = np.clip(diff / 255.0, 0.0, 1.0)
    return diff_scaled.astype(np.float32)


def generate_patches(images, patch_size, num_patches):
    X, Y = [], []
    n = len(images)
    while len(X) < num_patches:
        idx = random.randrange(n)
        orig = images[idx]['orig']
        filt = images[idx]['filt']
        h, w = orig.shape
        if h < patch_size or w < patch_size:
            continue
        y = random.randint(0, h - patch_size)
        x = random.randint(0, w - patch_size)
        xi = orig[y:y+patch_size, x:x+patch_size]
        yi = filt[y:y+patch_size, x:x+patch_size]
        if xi.shape != (patch_size, patch_size):
            continue
        X.append(xi[..., np.newaxis])
        Y.append(yi[..., np.newaxis])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def extract_patches(full_img, patch_size, overlap):
    stride = patch_size - overlap
    h, w = full_img.shape
    ys = list(range(0, h - patch_size, stride)) + [h - patch_size]
    xs = list(range(0, w - patch_size, stride)) + [w - patch_size]
    patches, coords = [], []
    for y in ys:
        for x in xs:
            patch = full_img[y:y+patch_size, x:x+patch_size]
            patches.append(patch[np.newaxis, ..., np.newaxis])
            coords.append((y, x))
    return np.vstack(patches), coords, (h, w)


def reconstruct_from_patches(patches, coords, full_shape, patch_size, overlap):
    h, w = full_shape
    accum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for p, (y, x) in zip(patches, coords):
        patch = p.squeeze()
        accum[y:y+patch_size, x:x+patch_size] += patch
        count[y:y+patch_size, x:x+patch_size] += 1
    return (accum / count).astype(np.float32)


def build_unet(input_shape, depth=4, start_filters=32):
    inputs = layers.Input(input_shape)
    skips, x = [], inputs
    for d in range(depth):
        f = start_filters * (2**d)
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    f = start_filters * (2**depth)
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    for d in reversed(range(depth)):
        f = start_filters * (2**d)
        x = layers.Conv2DTranspose(f, 2, strides=2, padding='same')(x)
        x = layers.concatenate([x, skips[d]])
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    return models.Model(inputs, outputs)


def visualize_full(model_path, image_path, patch_size, overlap):
    model = tf.keras.models.load_model(model_path, compile=False)
    full = load_image_gray(image_path)
    aniso_full = preprocess_image(full)
    patches, coords, shape = extract_patches(full, patch_size, overlap)
    preds = model.predict(patches)
    recon = reconstruct_from_patches(preds, coords, shape, patch_size, overlap)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(full, cmap='gray'); axs[0].set_title('Original')
    axs[1].imshow(aniso_full, cmap='gray'); axs[1].set_title('Anisotrópico')
    axs[2].imshow(recon, cmap='gray'); axs[2].set_title('Reconstruido')
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig('visualization.png')
    print('Visualización guardada en visualization.png')

# -----------------------------
# Ejecución principal
# -----------------------------
if DO_TRAIN:
    paths = glob.glob(os.path.join(TRAIN_DIR, '*.*'))
    images = []
    for p in paths:
        orig = load_image_gray(p)
        filt = preprocess_image(orig)
        if orig.shape[0] >= PATCH_SIZE and orig.shape[1] >= PATCH_SIZE:
            images.append({'orig': orig, 'filt': filt})
    print(f"Imágenes válidas: {len(images)}")
    X, Y = generate_patches(images, PATCH_SIZE, NUM_PATCHES)
    n = X.shape[0]
    idx = np.random.permutation(n)
    X, Y = X[idx], Y[idx]
    t, v, _ = SPLIT
    i1 = int(n * t); i2 = int(n * (t + v))
    X_train, Y_train = X[:i1], Y[:i1]
    X_val, Y_val     = X[i1:i2], Y[i1:i2]
    X_test, Y_test   = X[i2:], Y[i2:]
    X_train, Y_train = X_train.astype(np.float32), Y_train.astype(np.float32)
    X_val,   Y_val   = X_val.astype(np.float32),   Y_val.astype(np.float32)
    X_test,  Y_test  = X_test.astype(np.float32),  Y_test.astype(np.float32)
    model = build_unet((PATCH_SIZE, PATCH_SIZE, 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, Y_train, epochs=2, batch_size=64,
              validation_data=(X_val, Y_val))
    loss, mae = model.evaluate(X_test, Y_test)
    print(f"Test MAE: {mae:.4f}")
    model.save(MODEL_FILENAME)
    print(f"Modelo guardado en {MODEL_FILENAME}")

if FULL_IMAGE_PATH:
    visualize_full(MODEL_FILENAME, FULL_IMAGE_PATH, PATCH_SIZE, OVERLAP)
