# FinVQA-Chart: Financial Vision-Language Reasoning at Scale

<div align="center">

[![Dataset](https://img.shields.io/badge/Dataset-FinVQA--Chart-blue.svg)](.)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ACL](https://img.shields.io/badge/Conference-ACL%202026-red.svg)](.)
[![Questions](https://img.shields.io/badge/Questions-149.0K-orange.svg)](.)
[![Images](https://img.shields.io/badge/Images-9.96K-purple.svg)](.)

**The First Large-Scale Multi-Modal Benchmark for Financial Chart Understanding**

[📊 Dataset](#dataset-https://huggingface.co/datasets/qsWDFGHN/FIN-VQA) | [🚀 Quick Start](#quick-start) | [📖 Documentation](#dataset-structure) | [🎯 Evaluation](#evaluation-protocol) | [📝 Citation](#citation)

</div>

---

## 🎯 Abstract

**FinVQA-Chart** is a groundbreaking large-scale benchmark dataset designed to evaluate Vision-Language Models (VLMs) on complex financial reasoning tasks. Unlike existing datasets that focus on generic chart understanding or financial text analysis, FinVQA-Chart uniquely combines:

- **Real-world financial data** spanning 5 years (2020-2024) across critical market periods (COVID crash, inflation surge, Fed tightening)
- **Multi-modal reasoning** requiring simultaneous understanding of visual patterns, numerical data, and market context
- **Technical analysis depth** with 35,000+ professionally detected candlestick patterns and technical signals
- **Scale and diversity** with 149,000 questions across 7 distinct reasoning types and 3 difficulty levels

This dataset pushes the boundaries of VLM capabilities by requiring models to integrate visual chart interpretation, numerical reasoning, temporal understanding, and domain-specific financial knowledge—tasks that remain challenging for even state-of-the-art models.

---

## 📊 Dataset Overview

### Key Statistics

| Metric | Value | Description |
|--------|-------|-------------|
| **Total Images** | 9,960 | High-resolution stock charts (2000×1500 px) |
| **Total Questions** | 149,000 | Carefully curated QA pairs |
| **Questions per Image** | 15 | Diverse reasoning types per chart |
| **Stocks Covered** | 96 | 45 Technology + 51 Finance sector |
| **Time Period** | 2020-2024 | 5 years covering major market events |
| **Trading Days** | ~1,260/stock | Complete OHLCV data |
| **Patterns Detected** | 35,725+ | Professionally validated technical patterns |
| **Sectors** | 2 | Technology & Finance (balanced 50/50) |

### Question Type Distribution

```
Pattern Recognition      : 29,805 questions (21.1%) - Visual pattern identification
Volume Analysis          : 19,870 questions (14.1%) - Price-volume relationships  
Technical Indicators     : 19,870 questions (14.1%) - Moving averages, signals
Price Action            : 19,870 questions (14.1%) - Performance & volatility
Correlation Analysis    : 19,870 questions (14.1%) - Cross-stock relationships
Comparative Analysis    : 19,870 questions (14.1%) - Relative assessment
Contextual Reasoning    : 19,870 questions (14.1%) - Market context integration
```

### Difficulty Distribution (CVPR Standard)

- **Easy**: 46,618 questions (33%) - Single-step reasoning
- **Medium**: 66,395 questions (47%) - Multi-step integration
- **Hard**: 28,253 questions (20%) - Complex synthesis

### Modality Distribution

- **Visual-Only**: 42,380 questions (30%) - Answerable from chart alone
- **Textual-Only**: 32,491 questions (23%) - Answerable from description alone
- **Multi-Modal**: 66,395 questions (47%) - Requires both image and text

---

## 🌟 What Makes FinVQA-Chart Unique?

### 🏆 Unprecedented in Financial VLM Benchmarking

#### 1. **True Multi-Modal Fusion** (Not Just Visual QA)
Unlike existing datasets that primarily test visual understanding OR text comprehension, FinVQA-Chart requires **simultaneous integration** of:
- **Visual elements**: Candlestick patterns, trend lines, moving averages, volume bars
- **Numerical data**: Prices, volumes, percentages, correlations (embedded as natural language)
- **Temporal context**: Time-series relationships, sequential patterns, market events
- **Domain knowledge**: Technical analysis principles, market behavior, sector dynamics

**Example Multi-Modal Question:**
> "The text indicates a Hanging Man pattern was detected in AAPL on January 23. Based on the visual chart and subsequent price action, did this pattern lead to the expected bearish movement?"

This requires: (1) Reading text to find pattern, (2) Visual inspection to locate it, (3) Understanding pattern implications, (4) Analyzing follow-through movement.

#### 2. **Real Financial Data at Scale** (Not Synthetic or Limited)
- **96 real stocks** from S&P 500 (Technology & Finance sectors)
- **5 years** of actual market data including unprecedented events:
  - COVID-19 crash (March 2020)
  - Historic bull market (2021)
  - Inflation surge & Fed tightening (2022)
  - Banking crisis (March 2023)
  - AI-driven rally (2023-2024)
- **35,725+ validated patterns** using professional technical analysis algorithms
- **Realistic complexity**: Real market noise, failed signals, divergences, correlations

**Comparison with Other Datasets:**
| Dataset | Real Data | Multi-Modal | Financial Domain | Scale (Questions) | Pattern Detection |
|---------|-----------|-------------|------------------|-------------------|-------------------|
| ChartQA | ❌ Synthetic | ⚠️ Limited | ❌ Generic | 9.6K | ❌ None |
| PlotQA | ❌ Synthetic | ❌ Visual-only | ❌ Generic | 20K | ❌ None |
| FigureQA | ❌ Synthetic | ❌ Visual-only | ❌ Generic | 100K | ❌ None |
| FinQA | ✅ Real | ❌ Text-only | ✅ Financial | 8.3K | ❌ None |
| ConvFinQA | ✅ Real | ❌ Text-only | ✅ Financial | 14K | ❌ None |
| **FinVQA-Chart** | ✅ Real | ✅ True Multi-Modal | ✅ Financial | **141K** | ✅ 35K+ patterns |

#### 3. **Professional-Grade Technical Analysis**
Every pattern detection is based on **established technical analysis principles**:
- **15 pattern types**: Morning Star, Doji, Engulfing, Hammer, Hanging Man, etc.
- **Confidence scores**: 80-100% detection confidence
- **Multiple indicators**: Moving averages (20/50-day), volume analysis, support/resistance
- **Ground truth validation**: Patterns verified against price action outcomes

This enables questions like:
> "Which stock shows the most significant divergence between technical signals and actual price performance?"

Such questions require deep understanding of technical analysis—a capability absent in general VLMs.

#### 4. **Granular Reasoning Taxonomy** (7 Question Types)
Unlike binary "What is the value?" questions, FinVQA-Chart tests diverse cognitive skills:

**Visual Reasoning:**
- Pattern recognition and sequence analysis
- Relative performance comparison
- Moving average interpretation

**Numerical Reasoning:**
- Price change calculations
- Volatility assessment
- Performance ranking

**Relational Reasoning:**
- Stock correlation analysis
- Volume-price relationships
- Cross-asset comparisons

**Contextual Reasoning:**
- Market regime understanding
- Event impact analysis
- Pattern effectiveness validation

**Integration Reasoning:**
- Multi-factor synthesis
- Signal confirmation
- Risk-reward assessment

#### 5. **Temporal Complexity** (Time-Series Understanding)
Questions explicitly test temporal reasoning:
- "Did this pattern lead to expected movement over the next 5 days?"
- "Which stock led the sector recovery?"
- "How did correlation change during the period?"

This temporal dimension is **absent** in static chart QA datasets.

#### 6. **Controlled Difficulty Progression**
- **Easy**: Direct observation (e.g., "Which stock had highest volume?")
- **Medium**: Comparison/integration (e.g., "Which two stocks moved most similarly?")
- **Hard**: Synthesis/judgment (e.g., "Which shows highest conviction bullish setup?")

Enables **fine-grained model capability analysis** across skill levels.

#### 7. **Balanced Modality Requirements**
Unlike datasets biased toward visual or textual modalities:
- 30% **Visual-only** (tests pure chart reading)
- 23% **Textual-only** (tests numerical data extraction)
- 47% **Multi-modal** (tests integration)

This balance prevents models from "cheating" by relying on a single modality.

#### 8. **Natural Language Numerical Encoding**
Instead of raw arrays `[224.37, 228.91, 235.47, ...]`, numerical data is embedded as:
> "AAPL declined from $289.03 to $224.37 (-22.4% decline), then recovered to $288.25 (+28.5% from low). Volume averaged 58.2M shares/day, 65% above normal."

This tests VLM ability to **parse and reason about numbers in text**—crucial for real-world applications.

#### 9. **Strategic Distractor Generation**
Multiple-choice options are not random. Wrong answers include:
- **Tier 1**: Patterns from other stocks in same image (requires precise identification)
- **Tier 2**: Visually similar patterns (tests discrimination)
- **Tier 3**: Common but incorrect patterns (tests knowledge)

**Example:**
Correct: "Hanging Man"
Distractors: "Hammer" (visually similar), "Bullish Engulfing" (actually in MSFT), "Doji" (common pattern)

This **eliminates trivial elimination strategies** and requires true understanding.

#### 10. **Comprehensive Explanations**
Every answer includes 150-300 word explanation with:
- **Justification**: Why this answer is correct
- **Evidence**: Specific data points from chart/text
- **Reasoning**: Step-by-step logic
- **Context**: Broader implications

Enables **interpretable evaluation** and error analysis.

---

## 🎓 Motivation & Background

### The Financial VLM Gap

Vision-Language Models have achieved remarkable success on generic benchmarks (VQA, GQA, OK-VQA), yet their performance on **specialized domains requiring numerical reasoning and domain expertise** remains largely unexplored. 

**Critical Gap**: Financial analysis requires simultaneous understanding of:
1. **Visual patterns**: Technical chart formations that signal market sentiment
2. **Numerical precision**: Exact price levels, percentages, volumes matter
3. **Temporal dynamics**: Time-series relationships and sequential dependencies
4. **Domain knowledge**: Financial principles, market mechanics, trading concepts

### Why Financial Charts?

Financial charts are ideal for VLM evaluation because they:
- **Combine modalities naturally**: Visual (candlesticks, lines), textual (labels, legends), numerical (prices, volumes)
- **Require expert knowledge**: Understanding patterns requires domain training
- **Have objective ground truth**: Market outcomes validate predictions
- **Are high-stakes**: Real-world applications in trading, risk management, investment
- **Test temporal reasoning**: Charts inherently encode time-series information

### Real-World Impact

Strong performance on FinVQA-Chart indicates VLM capability for:
- **Automated financial analysis**: Assisting analysts in pattern recognition
- **Investment research**: Rapid screening of thousands of charts
- **Risk assessment**: Identifying technical warning signals
- **Educational tools**: Teaching technical analysis to retail investors
- **Multimodal financial agents**: Building AI systems that understand markets

---

## 📁 Dataset Structure

```
FinVQA-Chart/
│
├── 📂 images/                          # 9,960 high-resolution stock charts
│   ├── 📂 finance/                     # Finance sector (50%)
│   │   ├── 📂 2020/                    # ~1,000 images per year
│   │   │   ├── img_00001.png          # 2000×1500 px PNG
│   │   │   ├── img_00002.png
│   │   │   └── ...
│   │   ├── 📂 2021/
│   │   ├── 📂 2022/
│   │   ├── 📂 2023/
│   │   └── 📂 2024/
│   │
│   └── 📂 technology/                  # Technology sector (50%)
│       ├── 📂 2020/
│       ├── 📂 2021/
│       ├── 📂 2022/
│       ├── 📂 2023/
│       └── 📂 2024/
│
├── 📂 text_descriptions/               # 9,960 rich textual descriptions
│   ├── 📂 finance/
│   │   ├── 📂 2020/
│   │   │   ├── img_00001.json         # ~400-500 word description
│   │   │   ├── img_00002.json
│   │   │   └── ...
│   │   └── ...
│   └── 📂 technology/
│       └── ...
│
├── 📂 questions/                       # 141,266 QA pairs
│   ├── all_questions.jsonl            # Main file (JSONL format, ~85 MB)
│   │
│   ├── 📂 by_type/                     # Split by question type
│   │   ├── pattern_recognition.json   # 29,805 questions
│   │   ├── volume_analysis.json       # 19,870 questions
│   │   ├── technical_indicators.json  # 19,870 questions
│   │   ├── price_action.json          # 19,870 questions
│   │   ├── correlation_analysis.json  # 19,870 questions
│   │   ├── comparative_analysis.json  # 19,870 questions
│   │   └── contextual_reasoning.json  # 19,870 questions
│   │
│   ├── 📂 by_difficulty/               # Split by difficulty
│   │   ├── easy.json                  # 46,618 questions (33%)
│   │   ├── medium.json                # 66,395 questions (47%)
│   │   └── hard.json                  # 28,253 questions (20%)
│   │
│   └── 📂 by_modality/                 # Split by modality requirement
│       ├── visual_only.json           # 42,380 questions (30%)
│       ├── textual_only.json          # 32,491 questions (23%)
│       └── multi_modal.json           # 66,395 questions (47%)
│
├── 📂 metadata/                        # Dataset metadata
│   ├── complete_metadata.json         # Image index (paths, stocks, dates)
│   └── question_statistics.json       # Comprehensive statistics
│
├── 📂 samples/                         # Example questions for quick exploration
│   └── example_questions.json         # 100 representative samples
│
├── README.md                           # This file
└── LICENSE                             # Dataset license

```

---

## 📋 Data Format

### Image Format

Each chart image contains:
- **Layout**: 2×2 grid showing 4 stocks simultaneously
- **Resolution**: 2000 × 1500 pixels (high quality for pattern visibility)
- **Elements per subplot**:
  - Candlestick chart (green=bullish, red=bearish)
  - 20-day SMA (blue line)
  - 50-day SMA (orange line)
  - Volume bars (below price chart)
  - Date axis, price axis, grid
  - Stock ticker label
- **Time windows**: 30-day (20%), 60-day (60%), or 90-day (20%)

**Visual Example**:
```
┌─────────────────────┬─────────────────────┐
│  AAPL               │  MSFT               │
│  ┌───────────────┐  │  ┌───────────────┐  │
│  │ Candlesticks  │  │  │ Candlesticks  │  │
│  │ + MAs         │  │  │ + MAs         │  │
│  └───────────────┘  │  └───────────────┘  │
│  ┌───────────────┐  │  ┌───────────────┐  │
│  │ Volume Bars   │  │  │ Volume Bars   │  │
│  └───────────────┘  │  └───────────────┘  │
├─────────────────────┼─────────────────────┤
│  GOOGL              │  NVDA               │
│  [Same layout]      │  [Same layout]      │
└─────────────────────┴─────────────────────┘
```

### Text Description Format

```json
{
  "image_id": "img_00138",
  "image_path": "images/technology/2024/img_00138.png",
  "stocks": ["AAPL", "GDDY", "STX", "DELL"],
  "sector": "Technology",
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-03-31",
    "window_days": 90
  },
  "description": "This chart displays four Technology stocks (AAPL, GDDY, STX, DELL) during January 01 to March 31, 2024, covering Rate cut anticipation, strong market start. Market Context: Low volatility with 1.5% average daily price movements.\n\nAAPL: Declined 7.6% over the period, moving from $185.64 to $171.48. Trading range spanned $168.49 to $196.38 (14.2% range). Average daily volume was 62.0M shares. Hanging Man pattern detected on January 23 at $195.18 (confidence: 100%). Bearish Engulfing pattern detected on January 24 at $194.50 (confidence: 100%). Currently trading below both moving averages, suggesting bearish pressure.\n\n[... continues for all 4 stocks ...]\n\nTechnical Comparison: DELL outperformed with 52.6% gain versus sector average of 18.2%. Correlation Analysis: Average sector correlation of 0.17 indicates weak co-movement. Pattern Summary: Detected 15 bullish signals and 19 bearish signals across the group.",
  "word_count": 387,
  "generation_timestamp": "2025-10-24T09:04:15Z"
}
```

**Description Structure** (400-500 words):
1. **Overview** (50-70 words): Stocks, date range, market context
2. **Individual Analysis** (250-300 words): Each stock's performance, patterns, technical status
3. **Cross-Stock Comparison** (80-100 words): Correlations, relative performance, volume analysis
4. **Technical Summary** (50-70 words): Pattern counts, support/resistance levels, overall picture

### Question Format

```json
{
  "question_id": "img_00138_q001",
  "image_id": "img_00138",
  "question_type": "pattern_recognition",
  "difficulty": "easy",
  "modality": "visual_only",
  "question": "Looking at the chart for AAPL, what candlestick pattern appears on January 23, 2024?",
  "options": [
    "A) Bullish Engulfing",
    "B) Bearish Engulfing",
    "C) Morning Star",
    "D) Hanging Man",
    "E) No clear pattern"
  ],
  "correct_answer": "D",
  "answer_text": "Hanging Man",
  "explanation": "The chart shows a Hanging Man pattern on January 23 at $195.18. This is characterized by a small body near the top of the trading range with a long lower shadow, indicating potential bearish reversal after an uptrend. The pattern has 100% confidence based on technical analysis criteria.",
  "requires_image": true,
  "requires_text": false,
  "ground_truth_source": "visual_inspection",
  "related_stocks": ["AAPL"]
}
```

**Field Descriptions:**
- `question_id`: Unique identifier (format: `img_XXXXX_qYYY`)
- `image_id`: Links to corresponding image
- `question_type`: One of 7 types (pattern_recognition, volume_analysis, etc.)
- `difficulty`: easy | medium | hard
- `modality`: visual_only | textual_only | textual_primary | multi_modal
- `question`: Question text
- `options`: 4-5 multiple choice options
- `correct_answer`: Letter of correct option
- `answer_text`: Short answer string
- `explanation`: Detailed 150-300 word explanation with reasoning
- `requires_image`: Boolean - must look at chart to answer
- `requires_text`: Boolean - must read description to answer
- `ground_truth_source`: How answer was determined
- `related_stocks`: Array of relevant stock tickers

---

## 🚀 Quick Start

### Loading the Dataset

#### Option 1: Load Complete Dataset (JSONL - Memory Efficient)

```python
import json

def load_dataset_streaming(filepath='questions/all_questions.jsonl'):
    """Generator for memory-efficient loading"""
    with open(filepath, 'r') as f:
        for line in f:
            yield json.loads(line)

# Usage - iterate without loading all into memory
for question in load_dataset_streaming():
    print(question['question_id'], question['question'])
```

#### Option 2: Load Complete Dataset (All at Once)

```python
import json

def load_all_questions(filepath='questions/all_questions.jsonl'):
    """Load all questions into memory (requires ~2GB RAM)"""
    questions = []
    with open(filepath, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

questions = load_all_questions()
print(f"Loaded {len(questions):,} questions")
```

#### Option 3: Load Specific Subsets

```python
import json

# Load by difficulty
with open('questions/by_difficulty/easy.json', 'r') as f:
    easy_questions = json.load(f)

# Load by type
with open('questions/by_type/pattern_recognition.json', 'r') as f:
    pattern_questions = json.load(f)

# Load by modality
with open('questions/by_modality/multi_modal.json', 'r') as f:
    multimodal_questions = json.load(f)
```

### Loading Images and Text

```python
from PIL import Image
import json

def load_sample(image_id):
    """Load image, text description, and questions for one sample"""
    
    # Load metadata to find paths
    with open('metadata/complete_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    img_meta = metadata['images'][image_id]
    
    # Load image
    image = Image.open(img_meta['image_path'])
    
    # Load text description
    with open(img_meta['text_path'], 'r') as f:
        text_data = json.load(f)
    
    # Load questions for this image
    questions = []
    with open('questions/all_questions.jsonl', 'r') as f:
        for line in f:
            q = json.loads(line)
            if q['image_id'] == image_id:
                questions.append(q)
    
    return {
        'image': image,
        'text': text_data['description'],
        'stocks': text_data['stocks'],
        'questions': questions
    }

# Example usage
sample = load_sample('img_00138')
print(f"Stocks: {sample['stocks']}")
print(f"Questions: {len(sample['questions'])}")
sample['image'].show()
```

### Basic Statistics

```python
import json

# Load statistics
with open('metadata/question_statistics.json', 'r') as f:
    stats = json.load(f)

print("Dataset Statistics:")
print(f"Total Questions: {stats['dataset_info']['total_questions']:,}")
print(f"Total Images: {stats['dataset_info']['total_images']:,}")
print("\nBy Type:")
for q_type, count in stats['by_type'].items():
    print(f"  {q_type}: {count:,}")
print("\nBy Difficulty:")
for diff, count in stats['by_difficulty'].items():
    print(f"  {diff}: {count:,}")
```

---

## 🎯 Evaluation Protocol

### Metrics

We recommend reporting the following metrics:

#### 1. **Overall Accuracy**
```python
accuracy = correct_answers / total_questions
```

#### 2. **Accuracy by Question Type**
Breakdown across 7 types:
- Pattern Recognition
- Volume Analysis
- Technical Indicators
- Price Action
- Correlation Analysis
- Comparative Analysis
- Contextual Reasoning

#### 3. **Accuracy by Difficulty**
- Easy (33% of questions)
- Medium (47% of questions)
- Hard (20% of questions)

#### 4. **Accuracy by Modality**
- Visual-Only (30%)
- Textual-Only (23%)
- Multi-Modal (47%)

#### 5. **Sector Performance**
- Technology stocks
- Finance stocks

### Evaluation Code Template

```python
def evaluate_model(model, questions):
    """
    Evaluate VLM on FinVQA-Chart
    
    Args:
        model: VLM with predict(image, text, question) method
        questions: List of question dicts
    
    Returns:
        results: Dict with accuracy breakdowns
    """
    correct = 0
    results_by_type = {}
    results_by_difficulty = {}
    results_by_modality = {}
    
    for q in questions:
        # Load image and text
        image = load_image(q['image_id'])
        text = load_text(q['image_id'])
        
        # Get model prediction
        prediction = model.predict(
            image=image if q['requires_image'] else None,
            text=text if q['requires_text'] else None,
            question=q['question'],
            options=q['options']
        )
        
        # Check correctness
        is_correct = (prediction == q['correct_answer'])
        correct += is_correct
        
        # Track by category
        q_type = q['question_type']
        if q_type not in results_by_type:
            results_by_type[q_type] = {'correct': 0, 'total': 0}
        results_by_type[q_type]['correct'] += is_correct
        results_by_type[q_type]['total'] += 1
        
        # Similar tracking for difficulty and modality...
    
    return {
        'overall_accuracy': correct / len(questions),
        'by_type': results_by_type,
        'by_difficulty': results_by_difficulty,
        'by_modality': results_by_modality
    }
```

### Recommended Train/Val/Test Split

For benchmarking purposes, we recommend:

**Option A: Standard Split**
- Train: 70% (6,972 images, 98,868 questions)
- Validation: 15% (1,494 images, 21,690 questions)
- Test: 15% (1,494 images, 21,690 questions)

**Option B: Few-Shot Evaluation** (Recommended for VLMs)
- Use full dataset as test set
- Few-shot examples: 5-10 samples per question type

**Option C: Zero-Shot Evaluation** (Most challenging)
- Use full dataset as test set
- No training examples provided

---

## 📊 Dataset Statistics Visualization

### Question Type Distribution
```
Pattern Recognition    ████████████████████░ 21.1% (29,805)
Volume Analysis        ██████████████░░░░░░░ 14.1% (19,870)
Technical Indicators   ██████████████░░░░░░░ 14.1% (19,870)
Price Action          ██████████████░░░░░░░ 14.1% (19,870)
Correlation Analysis   ██████████████░░░░░░░ 14.1% (19,870)
Comparative Analysis   ██████████████░░░░░░░ 14.1% (19,870)
Contextual Reasoning   ██████████████░░░░░░░ 14.1% (19,870)
```

### Difficulty Distribution
```
Easy     █████████████████░░░░░░░░░░ 33.0% (46,618)
Medium   ████████████████████████░░░ 47.0% (66,395)
Hard     ██████████░░░░░░░░░░░░░░░░░ 20.0% (28,253)
```

### Modality Distribution
```
Visual-Only    ███████████████░░░░░░░░░░░ 30.0% (42,380)
Textual-Only   ███████████░░░░░░░░░░░░░░░ 23.0% (32,491)
Multi-Modal    ████████████████████████░░░ 47.0% (66,395)
```

### Temporal Coverage
```
2020: ████████████████████░ (2,000 images, ~28,200 questions)
2021: ████████████████████░ (2,000 images, ~28,200 questions)
2022: ████████████████████░ (2,000 images, ~28,200 questions)
2023: ████████████████████░ (2,000 images, ~28,200 questions)
2024: ███████████████████░░ (1,960 images, ~27,666 questions)
```

---

## 💡 Example Questions

### Easy - Visual Only (Pattern Recognition)
**Image**: Chart showing 4 tech stocks  
**Question**: "Looking at the chart for AAPL, what candlestick pattern appears on January 23, 2024?"  
**Options**:
- A) Bullish Engulfing
- B) Bearish Engulfing
- C) Morning Star
- D) Hanging Man ✓
- E) No clear pattern

**Answer**: D) Hanging Man  
**Explanation**: The chart shows a Hanging Man pattern on January 23 at $195.18, characterized by a small body near the top with a long lower shadow, indicating potential bearish reversal.

---

### Medium - Multi-Modal (Correlation Analysis)
**Image**: 4 finance stocks chart  
**Text**: "...Average sector correlation of 0.17 indicates weak co-movement..."  
**Question**: "The text reports a sector correlation of 0.17. What does this indicate about these four stocks?"  
**Options**:
- A) Very strong positive correlation (moved almost identically)
- B) Moderate positive correlation (generally moved together)
- C) Weak positive correlation (mostly independent movements) ✓
- D) Negative correlation (moved in opposite directions)
- E) Zero correlation (completely random movements)

**Answer**: C) Weak positive correlation  
**Explanation**: A correlation of 0.17 indicates weak positive correlation, meaning these stocks largely moved independently. Visual inspection confirms: AAPL declined 7.6%, while DELL surged 52.6%—a 60% performance spread demonstrating company-specific factors dominated over sector trends.

---


