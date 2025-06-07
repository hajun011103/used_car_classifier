import os
import matplotlib.pyplot as plt
from collections import Counter
import config

def count_images_per_class(train_dir):
    class_counts = {}
    for class_name in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.lower().endswith('.jpg')])
            class_counts[class_name] = num_images
    return class_counts

def plot_class_distribution(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(12, 6))
    plt.barh(classes, counts)
    plt.xlabel('Number of Images')
    plt.ylabel('Class')
    plt.title('Number of Images per Class')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    class_counts = count_images_per_class(config.TRAIN_DIR)
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")
    plot_class_distribution(class_counts)