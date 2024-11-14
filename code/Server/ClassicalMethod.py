import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter


class FaceDepthVerification:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Weights for combining face and depth scores
        self.face_weight = 0.6
        self.depth_weight = 0.4

    def preprocess_face(self, face_image):
        """Preprocess face image"""
        if face_image is None:
            return None
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_image
        # Normalize lighting
        face_normalized = cv2.equalizeHist(face_gray)
        return face_normalized

    def preprocess_depth(self, depth_map):
        """Preprocess depth map"""
        if depth_map is None:
            return None
        # Normalize depth values
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        # Reduce noise
        depth_smoothed = gaussian_filter(depth_normalized, sigma=1)
        return depth_smoothed.astype(np.uint8)

    def detect_face(self, image):
        """Detect and extract face region"""
        if image is None:
            return None
        faces = self.face_cascade.detectMultiScale(image, 1.3, 5)
        if len(faces) == 0:
            return None

        # Get largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        return (x, y, w, h)

    def extract_face_features(self, face_image):
        """Extract features from face image"""
        if face_image is None:
            return None

        features = []

        # 1. LBP-like features
        lbp = self._compute_lbp(face_image)
        lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 256]).flatten()
        features.extend(lbp_hist)

        # 2. Gradient features
        gx = cv2.Sobel(face_image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(face_image, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = cv2.phase(gx, gy, angleInDegrees=True)

        # Gradient histogram
        grad_hist, _ = np.histogram(mag.flatten(), bins=32)
        features.extend(grad_hist)

        # 3. Regional features
        h, w = face_image.shape
        regions = [
            face_image[: h // 2, : w // 2],  # Top-left
            face_image[: h // 2, w // 2 :],  # Top-right
            face_image[h // 2 :, : w // 2],  # Bottom-left
            face_image[h // 2 :, w // 2 :],  # Bottom-right
            face_image[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3],  # Center
        ]

        for region in regions:
            region_hist = cv2.calcHist([region], [0], None, [16], [0, 256]).flatten()
            features.extend(region_hist)

        return np.array(features)

    def extract_depth_features(self, depth_map):
        """Extract features from depth map"""
        if depth_map is None:
            return None

        features = []

        # 1. Depth histogram
        hist = cv2.calcHist([depth_map], [0], None, [32], [0, 255])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

        # 2. Depth gradients and surface normals
        gx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

        # Surface normals
        z = np.ones_like(gx) * 255
        normals = np.dstack((-gx, -gy, z))
        normals /= np.linalg.norm(normals, axis=2, keepdims=True)

        # Normal vector histogram
        normal_features = normals.reshape(-1, 3)
        normal_hist, _ = np.histogramdd(normal_features, bins=(4, 4, 4))
        features.extend(normal_hist.flatten())

        # 3. Regional depth analysis
        h, w = depth_map.shape
        regions = [
            depth_map[: h // 2, : w // 2],  # Top-left
            depth_map[: h // 2, w // 2 :],  # Top-right
            depth_map[h // 2 :, : w // 2],  # Bottom-left
            depth_map[h // 2 :, w // 2 :],  # Bottom-right
            depth_map[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3],  # Center (nose region)
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
        """Compute simplified LBP features"""
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

    def verify(self, ref_face, ref_depth, test_face, test_depth, threshold=0.6):
        """Verify identity using both face and depth information"""
        # Preprocess images
        ref_face_proc = self.preprocess_face(ref_face)
        test_face_proc = self.preprocess_face(test_face)
        ref_depth_proc = self.preprocess_depth(ref_depth)
        test_depth_proc = self.preprocess_depth(test_depth)

        # Detect faces and crop regions
        ref_face_roi = self.detect_face(ref_face_proc)
        test_face_roi = self.detect_face(test_face_proc)

        if ref_face_roi is None or test_face_roi is None:
            return False, 0.0, {}

        # Extract ROIs
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

        # Extract features
        ref_face_features = self.extract_face_features(ref_face_crop)
        test_face_features = self.extract_face_features(test_face_crop)
        ref_depth_features = self.extract_depth_features(ref_depth_crop)
        test_depth_features = self.extract_depth_features(test_depth_crop)

        # Calculate similarities
        face_similarity = 1 - cosine(ref_face_features, test_face_features)
        depth_similarity = 1 - cosine(ref_depth_features, test_depth_features)

        # Combined score
        combined_score = (
            self.face_weight * face_similarity + self.depth_weight * depth_similarity
        )

        # Prepare detailed results
        details = {
            "face_similarity": face_similarity,
            "depth_similarity": depth_similarity,
            "combined_score": combined_score,
            "face_roi": ref_face_roi,
            "threshold_used": threshold,
        }

        return combined_score > threshold, combined_score, details

    def visualize_results(self, ref_face, ref_depth, test_face, test_depth, details):
        """Visualize verification results"""
        # Create visualization
        h1, w1 = ref_face.shape[:2]
        h2, w2 = test_face.shape[:2]

        # Draw ROI on faces
        ref_face_viz = cv2.cvtColor(ref_face.copy(), cv2.COLOR_GRAY2BGR)
        test_face_viz = cv2.cvtColor(test_face.copy(), cv2.COLOR_GRAY2BGR)

        if "face_roi" in details:
            x, y, w, h = details["face_roi"]
            cv2.rectangle(ref_face_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(test_face_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Create depth visualizations
        ref_depth_viz = cv2.applyColorMap(ref_depth, cv2.COLORMAP_JET)
        test_depth_viz = cv2.applyColorMap(test_depth, cv2.COLORMAP_JET)

        # Combine visualizations
        viz_row1 = np.hstack([ref_face_viz, test_face_viz])
        viz_row2 = np.hstack([ref_depth_viz, test_depth_viz])
        visualization = np.vstack([viz_row1, viz_row2])

        # Add text with scores
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            visualization,
            f"Face Similarity: {details['face_similarity']:.2f}",
            (10, h1 + 30),
            font,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            visualization,
            f"Depth Similarity: {details['depth_similarity']:.2f}",
            (10, h1 + 60),
            font,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            visualization,
            f"Combined Score: {details['combined_score']:.2f}",
            (10, h1 + 90),
            font,
            0.6,
            (255, 255, 255),
            2,
        )

        return visualization


def main():
    """Example usage"""
    verifier = FaceDepthVerification()

    # Load images
    ref_face = cv2.imread(
        "./Server/data/FaceImage20241112204422.jpg", cv2.IMREAD_GRAYSCALE
    )
    ref_depth = cv2.imread(
        "./Server/data/DepthImage20241112204422.png", cv2.IMREAD_GRAYSCALE
    )
    test_face = cv2.imread(
        "./Server/data/tmp/FaceImage20241112204140.jpg", cv2.IMREAD_GRAYSCALE
    )
    test_depth = cv2.imread(
        "./Server/data/tmp/DepthImage20241112204140.png", cv2.IMREAD_GRAYSCALE
    )

    # Display results
    print(f"Match: {is_match}")
    print(f"Overall Score: {score:.2f}")
    print("\nDetailed Results:")
    for key, value in details.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")

    cv2.imshow("Verification Results", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
