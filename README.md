# Live Cell Segmentation Project

This repository contains the code for segmenting live cell images using a U-Net model.

## Project Structure

- `live_cell_segmentation.py`: The main script that loads data, defines the U-Net model, trains it, and evaluates the results.

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.x
- NumPy
- TensorFlow
- Matplotlib

You can install the required Python packages using:

pip install numpy tensorflow matplotlib

## Dataset
The dataset should be images of live cells. Implement the load_data function to load and preprocess the images. Update the data directory path in the script accordingly.

## Usage
1. Load Data:

The load_data function loads and preprocesses the image data.

data_dir = 'path/to/data'
train_data, val_data = load_data(data_dir)

2. Define U-Net Model:

The U-Net model is defined using TensorFlow's Keras API.

model = unet_model()

3. Train Model:

The model is trained on the training data.

train_model(model, train_data, val_data)

4. Save Model:

The trained model is saved.

model.save('live_cell_segmentation_model.h5')

5. Evaluate and Visualize Results

# Evaluate and visualize results

def evaluate_and_visualize(model, val_data, val_labels):
    # Evaluate the model
    loss, accuracy = model.evaluate(val_data, val_labels)
    print(f'Validation loss: {loss}')
    print(f'Validation accuracy: {accuracy}')

    # Make predictions on a few validation images
    predictions = model.predict(val_data[:5])

    # Plot the results
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        # Original image
        axes[0, i].imshow(val_data[i].reshape(256, 256), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        # Ground truth
        axes[1, i].imshow(val_labels[i].reshape(256, 256), cmap='gray')
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')

        # Predicted mask
        axes[2, i].imshow(predictions[i].reshape(256, 256), cmap='gray')
        axes[2, i].set_title('Predicted')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = 'path/to/data'
    train_data, val_data, val_labels = load_data(data_dir)
    
    model = unet_model()
    train_model(model, train_data, val_data)

    # Save the model
    model.save('live_cell_segmentation_model.h5')

    # Evaluate and visualize results
    evaluate_and_visualize(model, val_data, val_labels)
    
## Contributing
If you have any suggestions or improvements, feel free to open an issue or create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/JaCar-868/Disease-Progression/blob/main/LICENSE) file for details.
