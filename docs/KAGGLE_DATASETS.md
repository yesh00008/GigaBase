# Using Kaggle Datasets

This project supports collecting and using datasets from Kaggle for training your language models.

## Prerequisites

Before using Kaggle datasets, you need to:

1. Create a Kaggle account at https://www.kaggle.com
2. Generate an API key:
   - Go to your Kaggle account settings
   - Scroll down to "API" section
   - Click "Create New API Token"
   - This will download a `kaggle.json` file
3. Install the Kaggle CLI tool:
   ```bash
   pip install kaggle
   ```
4. Configure Kaggle authentication:
   - Place the `kaggle.json` file in the appropriate location:
     - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
     - macOS/Linux: `~/.kaggle/kaggle.json`
   - Set appropriate permissions (600) on the file

## Collecting Kaggle Datasets

### Using the Main Script

```bash
# Collect Kaggle datasets only
python main.py --collect-data --data-source kaggle

# Collect all datasets including Kaggle
python main.py --collect-data --data-source all
```

### Using the Data Collection Script Directly

```bash
# Collect Kaggle datasets
python scripts/data_collection.py --source kaggle

# Collect specific number of datasets
python scripts/data_collection.py --source kaggle --num-articles 20
```

## Available Kaggle Dataset Sources

The system automatically searches for relevant datasets on Kaggle based on these keywords:
- "text generation"
- "natural language processing"
- "nlp"
- "language model"
- "llm training"
- "machine learning text"
- "ai text dataset"

## Dataset Processing

Collected Kaggle datasets are processed as follows:

1. **Download**: Datasets are downloaded using the Kaggle API
2. **Extraction**: Compressed files are automatically extracted
3. **Text Extraction**: Text content is extracted from various file formats:
   - CSV/TSV files (text columns)
   - JSON files
   - Text files (.txt)
   - Markdown files (.md)
4. **Cleaning**: Text is cleaned and normalized
5. **Integration**: Processed text is integrated with other collected data

## Using Kaggle Datasets for Training

Once collected, Kaggle datasets are automatically used in the training pipeline:

```bash
# Collect data (including Kaggle datasets)
python main.py --collect-data --data-source kaggle

# Preprocess all collected data
python main.py --preprocess-data

# Train model with Kaggle data
python main.py --train-with-collected-data

# Or use fast training
python main.py --fast-train
```

## Example Workflow

```bash
# 1. Collect Kaggle datasets
python main.py --collect-data --data-source kaggle --num-articles 15

# 2. Preprocess the data
python main.py --preprocess-data

# 3. Train a model with the collected data
python main.py --fast-train --model gpt2

# 4. Test the trained model
python main.py --launch-frontend
```

## Supported File Formats

The data collection script can process these file formats from Kaggle datasets:
- `.csv` and `.tsv` (CSV/Tab-separated values)
- `.json` and `.jsonl` (JSON files)
- `.txt` (Plain text files)
- `.md` (Markdown files)
- `.xml` (XML files with text content)

## Customizing Kaggle Dataset Collection

You can customize the Kaggle dataset collection by modifying the search parameters in `scripts/data_collection.py`.

## Troubleshooting

### Authentication Issues
If you encounter authentication errors:
1. Verify that `kaggle.json` is in the correct location
2. Check that the file has the correct permissions (600)
3. Ensure your Kaggle account is active

### Download Issues
If datasets fail to download:
1. Check your internet connection
2. Verify you have sufficient disk space
3. Some datasets may require agreement to terms before download

### Processing Issues
If text extraction fails:
1. Check that the dataset contains text-based files
2. Some datasets may require custom processing scripts
3. Verify file formats are supported

## Best Practices

1. **Select Quality Datasets**: Choose datasets with high-quality, relevant text
2. **Check Dataset Size**: Large datasets may take significant time to download and process
3. **Verify Content**: Review dataset descriptions to ensure they match your training goals
4. **Respect Licenses**: Ensure compliance with dataset licenses and usage terms
5. **Combine Sources**: Use Kaggle datasets in combination with other data sources for better results

## Example Kaggle Datasets

Some popular datasets for language model training on Kaggle include:
- "The Pile" datasets
- "OpenWebText" collections
- "BookCorpus" datasets
- "WikiText" datasets
- Domain-specific text collections

## Next Steps

After collecting and training with Kaggle datasets:
1. Test your model using the web interface
2. Evaluate performance compared to models trained on other data sources
3. Fine-tune hyperparameters for optimal results
4. Combine with other data sources for comprehensive training