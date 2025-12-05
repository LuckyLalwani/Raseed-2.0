<div align="center">

# 🧾 Raseed 2.0

### AI-Powered Expense Intelligence Platform

*Transform your receipts into actionable financial insights with the power of AI*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## 🎯 What is Raseed?

**Raseed** (رَصِيد - Arabic for "balance") is an intelligent expense tracking system that uses OCR and AI to automatically extract, categorize, and analyze your spending from receipt images. No more manual data entry—just snap a photo and let AI do the heavy lifting!

### 💡 The Problem It Solves

- 📱 **Manual Entry Hell**: Tired of typing every expense manually?
- 🧾 **Receipt Chaos**: Shoebox full of crumpled receipts?
- 💸 **Budget Mystery**: Wonder where your money actually goes?
- 📊 **Analysis Paralysis**: Need insights but drowning in data?

**Raseed fixes all of this—automatically.**

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔍 Intelligent OCR
- Extract text from receipts (images & PDFs)
- Multi-page PDF support
- Handles poor quality images
- Powered by Tesseract OCR

</td>
<td width="50%">

### 🤖 AI-Powered Parsing
- Automatic vendor detection
- Smart item categorization
- Date & amount extraction
- Uses local LLM (Llama 3.2)

</td>
</tr>
<tr>
<td width="50%">

### 📊 Visual Analytics
- Interactive spending charts
- Category breakdowns
- Timeline visualizations
- Vendor analysis

</td>
<td width="50%">

### 💰 Budget Management
- Set category budgets
- Real-time alerts
- Monthly progress tracking
- Overspending warnings

</td>
</tr>
<tr>
<td width="50%">

### 🔎 Advanced Search
- Filter by date range
- Category & vendor filters
- Amount range queries
- Keyword search

</td>
<td width="50%">

### 🧠 AI Insights
- Ask questions in natural language
- Spending pattern analysis
- Personalized recommendations
- Trend detection

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites Checklist

- [x] Python 3.8 or higher
- [x] Tesseract OCR installed
- [x] Poppler (for PDF support)
- [x] Ollama with Llama 3.2 model

### 📦 Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/raseed-expense-tracker.git
cd raseed-expense-tracker
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Install External Tools

<details>
<summary><b>🔧 Install Tesseract OCR</b></summary>

**Windows:**
1. Download from: [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run installer (default location: `C:\Program Files\Tesseract-OCR\`)
3. Note the installation path

**Mac:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

</details>

<details>
<summary><b>📄 Install Poppler (PDF Support)</b></summary>

**Windows:**
1. Download from: [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
2. Extract to a folder (e.g., `C:\poppler\`)
3. Note the `bin` folder path

**Mac:**
```bash
brew install poppler
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

</details>

<details>
<summary><b>🤖 Install Ollama & Model</b></summary>

1. Download Ollama: [ollama.ai](https://ollama.ai)
2. Install and run Ollama
3. Pull the Llama model:
```bash
ollama pull llama3.2
```

</details>

#### Step 4: Configure Paths

Open `app_streamlit.py` and update these lines:
```python
# Line 18
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Line 41
POPPLER_PATH = r"C:\path\to\poppler\bin"
```

#### Step 5: Add Your Logo

Place your logo image as `Asset 1.png` in the project root directory.

---

## 🎮 Usage

### Launch the App
```bash
streamlit run app_streamlit.py
```

The app opens automatically at `http://localhost:8501`

### Workflow

1. **📸 Add Receipt**: Upload image or PDF
2. **📊 Dashboard**: View spending analytics
3. **🔎 Search**: Filter and find receipts
4. **💰 Budgets**: Set limits and track progress
5. **🧠 Insights**: Ask AI about your spending

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) |
| **OCR** | ![Tesseract](https://img.shields.io/badge/Tesseract-3C78D8?style=flat) ![PIL](https://img.shields.io/badge/Pillow-FFD43B?style=flat) |
| **AI/LLM** | ![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat) ![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat&logo=ollama&logoColor=white) |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white) |
| **PDF** | ![pdf2image](https://img.shields.io/badge/pdf2image-red?style=flat) ![Poppler](https://img.shields.io/badge/Poppler-orange?style=flat) |

</div>

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│                    (Streamlit App)                       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐      ┌─────────▼────────┐
│  OCR Engine    │      │   AI Parser      │
│  (Tesseract)   │      │  (Llama 3.2)     │
└───────┬────────┘      └─────────┬────────┘
        │                         │
        └────────────┬────────────┘
                     │
           ┌─────────▼──────────┐
           │   Data Storage     │
           │  (JSONL + JSON)    │
           └────────────────────┘
```

---

## 📁 Project Structure
```
raseed-expense-tracker/
│
├── 📱 app_streamlit.py              # Main application
├── 🔄 receipt_agent_graph.py       # LangGraph pipeline
├── 🤖 receipt_parser_ai.py         # AI parser
├── 📝 receipt_parser_basic.py      # Rule-based parser
├── 💬 receipt_query_ai.py          # AI query interface
│
├── 🛠️ Utilities
│   ├── clean_text.py               # Text preprocessing
│   ├── parse_basic.py              # Basic parsing
│   └── structured_output.py        # Data structures
│
├── 🧪 Testing
│   ├── test_ocr.py                 # Tesseract tests
│   └── test_easyocr.py             # EasyOCR tests
│
├── 📄 Configuration
│   ├── .env                        # API keys (not in repo)
│   ├── .gitignore                  # Git ignore rules
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # You are here!
│
└── 💾 Data (auto-created)
    ├── receipts_store.jsonl        # Receipt database
    └── budget.json                 # Budget settings
```

---

## 🎨 Different Tabs

<details>
<summary><b>📊 Dashboard View</b></summary>

- Total spending metrics
- Category breakdown (pie chart)
- Daily spending timeline
- Recent receipts cards

</details>

<details>
<summary><b>🔎 Search & Filter</b></summary>

- Advanced filters (date, amount, category, vendor)
- Keyword search
- Export to CSV

</details>

<details>
<summary><b>💰 Budget Tracking</b></summary>

- Set monthly budgets per category
- Visual progress bars
- Overspending alerts

</details>

<details>
<summary><b>🧠 AI Insights</b></summary>

- Ask questions in natural language
- Get personalized recommendations
- Spending pattern analysis

</details>

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file for optional OpenAI integration:
```env
OPENAI_API_KEY=your_api_key_here
```

### Data Storage

| File | Purpose | Format |
|------|---------|--------|
| `data/receipts_store.jsonl` | Parsed receipts | JSON Lines |
| `data/budget.json` | Budget settings | JSON |

---

## 🤝 Contributing

We love contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

### Ideas for Contribution

- 🌐 Multi-language OCR support
- 📱 Mobile app version
- ☁️ Cloud storage integration
- 🔔 Email/SMS budget alerts
- 📈 Advanced ML-based predictions
- 🎨 Custom themes

---

## 🐛 Known Issues & Limitations

- OCR accuracy depends on image quality
- LLM parsing may occasionally misclassify items
- Currently supports English receipts only
- Requires local Ollama installation (no cloud option yet)

---

## 🗺️ Roadmap

- [x] Basic OCR & parsing
- [x] Streamlit dashboard
- [x] AI insights
- [x] Budget tracking
- [ ] Mobile app (React Native)
- [ ] Cloud sync
- [ ] Multi-user support
- [ ] Receipt templates
- [ ] Export to accounting software
- [ ] Subscription tracking
- [ ] Tax deduction suggestions

---

## ❓ FAQ

<details>
<summary><b>Does this work with handwritten receipts?</b></summary>

OCR works best with printed text. Handwritten receipts may have lower accuracy.

</details>

<details>
<summary><b>Is my data secure?</b></summary>

All data is stored locally on your machine. Nothing is sent to external servers (except when using OpenAI API if configured).

</details>

<details>
<summary><b>Can I use this commercially?</b></summary>

Yes! This project is MIT licensed. Use it however you want.

</details>

<details>
<summary><b>What currencies are supported?</b></summary>

Currently optimized for Indian Rupees (₹), but you can modify the code for any currency.

</details>

<details>
<summary><b>Can I use OpenAI instead of Ollama?</b></summary>

Yes! The code supports both. Just set your `OPENAI_API_KEY` in the `.env` file and modify the LLM initialization.

</details>

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2024 Raseed Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so.
```

---

## 🙏 Acknowledgments

Built with ❤️ using these amazing technologies:

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - The OCR engine
- [Ollama](https://ollama.ai) - Local LLM platform
- [Streamlit](https://streamlit.io) - Beautiful Python apps
- [LangChain](https://www.langchain.com) - LLM application framework
- [Plotly](https://plotly.com) - Interactive visualizations
- [Meta's Llama](https://llama.meta.com) - Open-source LLM

Special thanks to the open-source community! 🌟

---

## 📞 Contact & Support

- 🐛 **Bug Reports**: [Open an issue](https://github.com/YOUR_USERNAME/raseed-expense-tracker/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/YOUR_USERNAME/raseed-expense-tracker/discussions)
- 📧 **Email**: lalwanilucky333@gmail.com
- 🐦 **LinkedIn**: [@Lucky Lalwani](https://www.linkedin.com/in/lucky-lalwani-410b42256/)

---

<div align="center">

### ⭐ Star this repo if you find it useful!

Made with 💚 and ☕ by Lucky Lalwani

[⬆ Back to Top](#-raseed-20)

</div>
