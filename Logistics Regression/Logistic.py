import numpy as np
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

LABELS = ["airplane","automobile","bird","cat",
          "deer","dog","frog","horse","ship","truck"]

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    X_train = trainset.data.reshape(len(trainset), -1).astype('float32') / 255.0
    y_train = np.array(trainset.targets)
    X_test  = testset.data.reshape(len(testset),  -1).astype('float32') / 255.0
    y_test  = np.array(testset.targets)
    return X_train, y_train, X_test, y_test

def train_logistic(X_train, y_train):
    clf = LogisticRegression(
        solver='saga',
        max_iter=5000,
        tol=1e-3,
        n_jobs=1,
        verbose=0
    )
    print(f"Training on {len(X_train)} samples...")
    clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X, y, name=None):
    if name:
        print(f"\n=== Evaluation on {name} ({len(y)} samples) ===")
    y_pred = clf.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred, target_names=LABELS))
    return y_pred

def show_examples(X, y_true, y_pred, num=8):
    idxs = np.random.choice(len(y_true), size=num, replace=False)
    plt.figure(figsize=(num*2, 3))
    for i, idx in enumerate(idxs):
        img = (X[idx].reshape(32,32,3) * 255).astype('uint8')
        plt.subplot(1, num, i+1)
        plt.imshow(img)
        plt.title(f"T:{LABELS[y_true[idx]]}\nP:{LABELS[y_pred[idx]]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=50000,
                        help='Number of training samples (e.g. 10000, 30000, 50000)')
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data()
    
    if args.subset < len(X_train):
        X_sub, y_sub = X_train[:args.subset], y_train[:args.subset]
    else:
        X_sub, y_sub = X_train, y_train

    
    scaler = StandardScaler()
    X_sub_scaled = scaler.fit_transform(X_sub)
    X_test_scaled = scaler.transform(X_test)

    clf = train_logistic(X_sub_scaled, y_sub)
    evaluate(clf, X_sub_scaled, y_sub, name=f"train (first {args.subset})")
    evaluate(clf, X_test_scaled, y_test, name="test")

    y_pred_test = clf.predict(X_test_scaled)
    show_examples(X_test, y_test, y_pred_test, num=8)

if __name__ == "__main__":
    main()
