# LLM Testing Frontend

This web-based interface allows you to test the trained language models in this project. You can select from different models, customize your prompts, and generate text.

## Features

- Select from any of the trained models in the `models/` directory
- Customize your prompts
- Adjust the maximum output length
- Copy generated text to clipboard

## Getting Started

### Prerequisites

Make sure you have all the necessary packages installed:

```bash
pip install -r requirements.txt
```

### Running the Frontend

You can start the frontend using the main script:

```bash
python main.py --launch-frontend
```

Or you can run it directly:

```bash
python app/app.py
```

The web interface will be available at `http://localhost:5000`.

## Using the Interface

1. **Select a Model**: Choose one of the trained models from the list on the left.
2. **Configure Settings**: Adjust the maximum length slider to control how long the generated text will be.
3. **Enter a Prompt**: Type your prompt in the text area.
4. **Generate Text**: Click the "Generate Text" button (or press Ctrl+Enter) to generate text.
5. **Copy Output**: Use the "Copy" button to copy the generated text to your clipboard.

## Available Models

The interface automatically detects and lists all available models in the `models/` directory. Different models have different characteristics:

- **fine_tuned**: Standard trained models with balanced performance and quality
- **fast_trained**: Models trained with speed optimizations, slightly lower quality
- **ultra_fast_trained**: Models optimized for maximum speed, may have lower quality
- **super_fast_trained**: Extremely fast models with basic capabilities
- **collected_data_trained**: Models trained specifically on the collected dataset

## Troubleshooting

- If no models appear, make sure you have trained at least one model.
- If loading a model takes a long time, consider using a smaller model or one trained with fewer parameters.
- If you encounter memory errors, try closing other applications or using a smaller model.

## Contributing

Feel free to enhance this frontend by adding features such as:

- Batch processing of prompts
- Visualization of model performance metrics
- Comparison between different models
- Support for more advanced generation parameters