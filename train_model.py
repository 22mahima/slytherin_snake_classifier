import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_sample_dataset():
    """Create a placeholder dataset with sample images"""
    print("Creating sample dataset...")
    
    # Create directories
    os.makedirs('sample_dataset/venomous', exist_ok=True)
    os.makedirs('sample_dataset/non-venomous', exist_ok=True)
    
    # Create sample images (simple patterns to simulate different snake types)
    def create_sample_image(filename, pattern_type):
        img = Image.new('RGB', (200, 150), color='black')
        pixels = img.load()
        
        for i in range(200):
            for j in range(150):
                if pattern_type == 'venomous':
                    # Create a diamond/zigzag pattern (simulating viper patterns)
                    if (i + j) % 20 < 10:
                        pixels[i, j] = (139, 69, 19)  # Brown
                    else:
                        pixels[i, j] = (255, 140, 0)  # Orange
                else:
                    # Create solid/striped pattern (simulating non-venomous snakes)
                    if i % 15 < 7:
                        pixels[i, j] = (34, 139, 34)  # Green
                    else:
                        pixels[i, j] = (85, 107, 47)  # Dark olive green
        
        img.save(filename)
    
    # Create venomous samples
    for i in range(10):
        create_sample_image(f'sample_dataset/venomous/venomous_{i+1}.jpg', 'venomous')
    
    # Create non-venomous samples
    for i in range(10):
        create_sample_image(f'sample_dataset/non-venomous/non_venomous_{i+1}.jpg', 'non-venomous')
    
    print("Sample dataset created successfully!")

def load_dataset(dataset_path='sample_dataset'):
    """Load and preprocess the dataset"""
    images = []
    labels = []
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Creating sample dataset...")
        create_sample_dataset()
    
    # Load venomous images
    venomous_path = os.path.join(dataset_path, 'venomous')
    if os.path.exists(venomous_path):
        for filename in os.listdir(venomous_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(venomous_path, filename)
                try:
                    image = Image.open(img_path)
                    processed = preprocess_image(image)
                    images.append(processed)
                    labels.append('venomous')
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    # Load non-venomous images
    non_venomous_path = os.path.join(dataset_path, 'non-venomous')
    if os.path.exists(non_venomous_path):
        for filename in os.listdir(non_venomous_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(non_venomous_path, filename)
                try:
                    image = Image.open(img_path)
                    processed = preprocess_image(image)
                    images.append(processed)
                    labels.append('non-venomous')
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return np.array(images), np.array(labels)

def preprocess_image(image):
    """Preprocess image: resize to 128x128, convert to grayscale, flatten"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 128x128
    image = image.resize((128, 128))
    
    # Convert to numpy array and flatten
    image_array = np.array(image)
    flattened = image_array.flatten()
    
    # Normalize pixel values to 0-1 range
    normalized = flattened / 255.0
    
    return normalized

def train_model():
    """Train the Random Forest classifier"""
    print("Loading dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("No images found in dataset!")
        return
    
    print(f"Dataset loaded: {len(X)} images")
    print(f"Classes: {np.unique(y)}")
    
    # Split dataset (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Training Complete!")
    print(f"Test Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_data = {
        'model': model,
        'accuracy': accuracy
    }
    
    with open('snake_classifier_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'snake_classifier_model.pkl'")
    
    return model, accuracy

if __name__ == "__main__":
    train_model()