import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from hw_kernels import KernelizedRidgeRegression, SVR, RBF
from tqdm import tqdm


def load_utkface_data(path, target_size=(64, 64), max_samples=23708):
    images = []
    ages = []
    count = 0

    filenames = sorted(os.listdir(path))

    # First, take every 7th image
    step_filenames = filenames[::7]
    # Then, take the remaining images to fill up to max_samples ... opting for more balanced dataset
    remaining_filenames = [f for f in filenames if f not in step_filenames]
    selected_filenames = step_filenames + remaining_filenames[:max_samples - len(step_filenames)]

    for filename in tqdm(selected_filenames, desc="Processing images"):
        if filename.endswith('.jpg'):
            # [age]_[gender]_[race]_[date&time].jpg
            try:
                age = int(filename.split('_')[0])
                img = cv2.imread(os.path.join(path, filename))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    ages.append(age)
                    count += 1
            except:
                continue

    return np.array(images), np.array(ages)


class NaiveImageKernel:
    """Treats images as flattened vectors and uses standard kernels"""

    def __init__(self, base_kernel):
        self.base_kernel = base_kernel

    def __call__(self, A, B=None):
        A_flat = np.array([a.flatten() for a in A])
        B_flat = A_flat if B is None else np.array([b.flatten() for b in B])

        with tqdm(total=1, desc="Computing kernel") as pbar:
            result = self.base_kernel(A_flat, B_flat)
            pbar.update(1)

        return result


class LBPKernel:
    """Local Binary Pattern kernel for facial age estimation"""

    def __init__(self, sigma=1.0, radius=2, n_points=8):
        self.sigma = sigma
        self.gamma = 1.0 / (2 * sigma ** 2)
        self.radius = radius
        self.n_points = n_points

    def _extract_lbp(self, image):
        lbp = local_binary_pattern(image, self.n_points, self.radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, self.n_points + 3),
                               range=(0, self.n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    def __call__(self, A, B=None):
        if B is None:
            B = A

        batch_size = 32
        hist_A = []
        for i in tqdm(range(0, len(A), batch_size), desc="Processing A"):
            batch = A[i:i+batch_size]
            hist_A.extend([self._extract_lbp(a) for a in batch])
        hist_A = np.array(hist_A)

        hist_B = []
        for i in tqdm(range(0, len(B), batch_size), desc="Processing B"):
            batch = B[i:i+batch_size]
            hist_B.extend([self._extract_lbp(b) for b in batch])
        hist_B = np.array(hist_B)

        distances = np.zeros((len(A), len(B)))
        for i in tqdm(range(len(A)), desc="Computing distances"):
            diff = np.linalg.norm(hist_A[i] - hist_B, axis=1)
            distances[i] = np.exp(-self.gamma * diff)

        return distances

def plot_all_predictions(results, X_test, y_test):
    plt.figure(figsize=(15, 10))
    for i, (name, model) in enumerate(results.items(), 1):
        preds = model.predict(X_test)
        mae = np.mean(np.abs(preds - y_test))

        plt.subplot(2, 2, i)
        plt.scatter(y_test, preds, alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        plt.title(f'{name}\nMAE: {mae:.2f} years')
        plt.grid()

    plt.tight_layout()
    plt.savefig('visualizations/age_predictions.png')
    plt.show()


def plot_error_distribution(results, X_test, y_test):
    plt.figure(figsize=(12, 6))
    for name, model in results.items():
        preds = model.predict(X_test)
        errors = preds - y_test
        plt.hist(errors, bins=30, alpha=0.5, label=name)

    plt.xlabel('Prediction Error (years)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Across Models')
    plt.legend()
    plt.grid()
    plt.savefig('visualizations/age_error_distribution.png')
    plt.show()


if __name__ == "__main__":
    n = 7000
    print(f'Loading ${n} samples of UTKFace data...')
    X, y = load_utkface_data('UTKFace', max_samples=n)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    kernels = {
        'Naive RBF': NaiveImageKernel(RBF(sigma=0.1)),
        'LBP': LBPKernel(sigma=0.1, radius=2, n_points=8)
    }

    models = {
        'Naive RBF SVR': SVR(kernels['Naive RBF'], lambda_=0.1, epsilon=1.0),
        'LBP SVR': SVR(kernels['LBP'], lambda_=0.1, epsilon=1.0)
    }

    print("\nTraining models...")
    trained_models = {}
    for name, model in models.items():
        print(f"- Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    print("\nGenerating plots...")
    plot_all_predictions(trained_models, X_test, y_test)
    plot_error_distribution(trained_models, X_test, y_test)
