# Multi-Dataset Training Guide

## 📥 Downloaded Datasets

You now have access to **9 high-quality datasets** from Hugging Face:

### 1. **FineWeb** (HuggingFaceFW/fineweb)
- **Type**: Web text
- **Quality**: High quality, filtered web content
- **Samples**: 10,000
- **Use**: General knowledge, diverse topics

### 2. **C4** (allenai/c4)
- **Type**: Colossal Clean Crawled Corpus
- **Quality**: Cleaned and filtered web text
- **Samples**: 10,000
- **Use**: General language understanding

### 3. **DeepMath** (zwhe99/DeepMath-103K)
- **Type**: Mathematical reasoning
- **Quality**: Math problems and solutions
- **Samples**: Full dataset (~103K)
- **Use**: Mathematical and logical reasoning

### 4. **Nemotron Personas** (nvidia/Nemotron-Personas)
- **Type**: Conversational data with personas
- **Quality**: High-quality dialogue
- **Samples**: 5,000
- **Use**: Conversational AI, persona modeling

### 5. **TinyStories** (roneneldan/TinyStories)
- **Type**: Simple stories
- **Quality**: Synthetic but coherent narratives
- **Samples**: 50,000
- **Use**: Story generation, basic reasoning

### 6. **Wikipedia** (wikimedia/wikipedia)
- **Type**: Encyclopedia articles
- **Quality**: Factual, well-structured
- **Samples**: 20,000
- **Use**: Factual knowledge, formal writing

### 7. **Medical O1 Reasoning** (FreedomIntelligence/medical-o1-reasoning-SFT)
- **Type**: Medical reasoning
- **Quality**: Domain-specific medical content
- **Samples**: 5,000
- **Use**: Medical knowledge, specialized reasoning

### 8. **WildChat** (allenai/WildChat-1M)
- **Type**: Natural conversations
- **Quality**: Real-world chat data
- **Samples**: 10,000
- **Use**: Conversational AI, natural dialogue

### 9. **FineWeb-Edu** (HuggingFaceFW/fineweb-edu)
- **Type**: Educational content
- **Quality**: High-quality educational text
- **Samples**: 15,000
- **Use**: Educational responses, tutoring

---

## 📊 Total Training Data

- **Total Datasets**: 9
- **Total Samples**: ~135,000+
- **Storage Location**: `data/downloaded_datasets/`
- **Format**: JSONL (JSON Lines)

---

## 🚀 How to Use for Training

### Option 1: Quick Training (Recommended First)
Already downloaded! Use the quick dataset:

```bash
python train_with_downloaded_data.py --data data/quick_training/quick_training_data.jsonl
```

### Option 2: Train with All Datasets
Wait for the full download to complete, then:

```bash
python train_with_all_datasets.py --all
```

### Option 3: Select Specific Datasets
Train on specific types:

```bash
# Conversational only
python train_with_all_datasets.py --datasets wildchat nemotron

# Educational + Wikipedia
python train_with_all_datasets.py --datasets fineweb-edu wikipedia

# Math reasoning
python train_with_all_datasets.py --datasets deepmath
```

---

## 📁 File Structure

```
data/
├── quick_training/
│   └── quick_training_data.jsonl          (20,000 samples - READY NOW)
│
└── downloaded_datasets/
    ├── HuggingFaceFW_fineweb_default.jsonl
    ├── allenai_c4_en.jsonl
    ├── zwhe99_DeepMath-103K_default.jsonl
    ├── nvidia_Nemotron-Personas_default.jsonl
    ├── roneneldan_TinyStories_default.jsonl
    ├── wikimedia_wikipedia_20231101.en.jsonl
    ├── FreedomIntelligence_medical-o1-reasoning-SFT_en.jsonl
    ├── allenai_WildChat-1M_default.jsonl
    ├── HuggingFaceFW_fineweb-edu_default.jsonl
    └── download_metadata.json              (Download statistics)
```

---

## ⏱️ Download Status

### ✅ Already Downloaded (FAST):
- **Quick Training Dataset**: 20,000 samples ready to use NOW

### 🔄 Currently Downloading (Background):
- All 9 datasets downloading in parallel
- Check progress: `data/downloaded_datasets/download_metadata.json`
- Estimated time: 10-30 minutes depending on connection

---

## 🎯 Training Recommendations

### For Quick Testing:
1. ✅ Use `data/quick_training/quick_training_data.jsonl` (ready now)
2. Fast training: 1-2 hours on CPU
3. Good for validating your training pipeline

### For Best Results:
1. Wait for all datasets to download
2. Use combined data from all sources
3. Train for longer (4-8 hours)
4. Better model quality with diverse data

---

## 💡 Next Steps

1. **Check download progress**:
   ```bash
   python check_download_progress.py
   ```

2. **Start quick training NOW**:
   ```bash
   python train_with_quick_data.py
   ```

3. **Or wait for full download and train on everything**:
   ```bash
   python train_with_all_datasets.py --all --epochs 3
   ```

---

## 🔍 Dataset Quality

**High Quality** (Best for training):
- ✅ FineWeb-Edu
- ✅ Wikipedia  
- ✅ DeepMath
- ✅ Medical O1 Reasoning

**Good Quality** (Conversational):
- ✅ WildChat
- ✅ Nemotron Personas

**Large Scale** (Diverse):
- ✅ C4
- ✅ FineWeb
- ✅ TinyStories

---

## 📝 Training Script (Coming Next)

I'll create a training script that:
1. Loads all downloaded datasets
2. Combines them intelligently
3. Preprocesses for your model
4. Trains with optimal settings
5. Saves checkpoints regularly

**Ready to train?** The quick dataset is ready now, and the full datasets are downloading in the background! 🚀
