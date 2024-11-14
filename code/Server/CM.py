import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter
import tools
import matplotlib.pyplot as plt


class FaceDepthVerification:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_weight = 0.6
        self.depth_weight = 0.4

    def detect_face(self, image):
        if image is None:
            return None
        faces = self.face_cascade.detectMultiScale(image, 1.3, 5)
        if len(faces) == 0:
            return None

        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        return (x, y, w, h)

    def preprocess_depth(self, depth_map):
        if depth_map is None:
            return None
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_smoothed = gaussian_filter(depth_normalized, sigma=1)
        return depth_smoothed.astype(np.uint8)

    def extract_face_features(self, face_image):
        if face_image is None:
            return None

        features = []

        lbp = self._compute_lbp(face_image)
        lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 256]).flatten()
        features.extend(lbp_hist)

        gx = cv2.Sobel(face_image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(face_image, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = cv2.phase(gx, gy, angleInDegrees=True)

        grad_hist, _ = np.histogram(mag.flatten(), bins=32)
        features.extend(grad_hist)

        h, w = face_image.shape
        regions = [
            face_image[: h // 2, : w // 2],
            face_image[: h // 2, w // 2 :],
            face_image[h // 2 :, : w // 2],
            face_image[h // 2 :, w // 2 :],
            face_image[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3],  # Center
        ]

        for region in regions:
            region_hist = cv2.calcHist([region], [0], None, [16], [0, 256]).flatten()
            features.extend(region_hist)

        return np.array(features)

    def extract_depth_features(self, depth_map):
        if depth_map is None:
            return None

        features = []

        hist = cv2.calcHist([depth_map], [0], None, [32], [0, 255])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

        gx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

        z = np.ones_like(gx) * 255
        normals = np.dstack((-gx, -gy, z))
        normals /= np.linalg.norm(normals, axis=2, keepdims=True)

        normal_features = normals.reshape(-1, 3)
        normal_hist, _ = np.histogramdd(normal_features, bins=(4, 4, 4))
        features.extend(normal_hist.flatten())

        h, w = depth_map.shape
        regions = [
            depth_map[: h // 2, : w // 2],
            depth_map[: h // 2, w // 2 :],
            depth_map[h // 2 :, : w // 2],
            depth_map[h // 2 :, w // 2 :],
            depth_map[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3],
        ]

        for region in regions:
            region_stats = [
                np.mean(region),
                np.std(region),
                np.max(region) - np.min(region),
            ]
            features.extend(region_stats)

        return np.array(features)

    def _compute_lbp(self, image):
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = image[i, j]
                code = (
                    (image[i - 1, j - 1] >= center) << 7
                    | (image[i - 1, j] >= center) << 6
                    | (image[i - 1, j + 1] >= center) << 5
                    | (image[i, j + 1] >= center) << 4
                    | (image[i + 1, j + 1] >= center) << 3
                    | (image[i + 1, j] >= center) << 2
                    | (image[i + 1, j - 1] >= center) << 1
                    | (image[i, j - 1] >= center)
                )
                lbp[i, j] = code
        return lbp

    def verify(
        self, ref_facePath, ref_depthPath, test_facePath, test_depthPath, threshold=0.6
    ):
        ref_face = self.readImage(ref_facePath)
        ref_depth = self.readImage(ref_depthPath)
        test_face = self.readImage(test_facePath)
        test_depth = self.readImage(test_depthPath)

        ref_face_proc = ref_face[..., ::-1]
        test_face_proc = test_face[..., ::-1]
        ref_depth_proc = self.preprocess_depth(ref_depth)
        test_depth_proc = self.preprocess_depth(test_depth)

        ref_face_roi = self.detect_face(ref_face_proc)
        test_face_roi = self.detect_face(test_face_proc)

        if ref_face_roi is None or test_face_roi is None:
            return False, 0.0, {}

        x1, y1, w1, h1 = ref_face_roi
        x2, y2, w2, h2 = test_face_roi

        ref_face_crop = cv2.resize(
            ref_face_proc[y1 : y1 + h1, x1 : x1 + w1], (480, 480)
        )
        test_face_crop = cv2.resize(
            test_face_proc[y2 : y2 + h2, x2 : x2 + w2], (480, 480)
        )
        ref_depth_crop = cv2.resize(
            ref_depth_proc[y1 : y1 + h1, x1 : x1 + w1], (480, 480)
        )
        test_depth_crop = cv2.resize(
            test_depth_proc[y2 : y2 + h2, x2 : x2 + w2], (480, 480)
        )

        ref_face_features = self.extract_face_features(ref_face_crop)
        test_face_features = self.extract_face_features(test_face_crop)
        ref_depth_features = self.extract_depth_features(ref_depth_crop)
        test_depth_features = self.extract_depth_features(test_depth_crop)

        face_similarity = 1 - cosine(ref_face_features, test_face_features)
        depth_similarity = 1 - cosine(ref_depth_features, test_depth_features)

        combined_score = (
            self.face_weight * face_similarity + self.depth_weight * depth_similarity
        )

        return combined_score

    def readImage(self, imagePath):
        return cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
