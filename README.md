# Indian Flag Image Validator

**A championship-grade computer vision system for validating the Indian National Flag** against the official **Bureau of Indian Standards (BIS)** specifications.  
This tool performs pixel-level analysis to check compliance with **color accuracy, aspect ratio, stripe proportions, and Ashoka Chakra design**.

---

## ✨ Features

### 📏 Aspect Ratio Check
- Validates the official **3:2 aspect ratio** with **sub-pixel precision**.

### 🎨 Color Analysis
- Ensures correct **saffron, white, green**, and **navy blue (Chakra)** colors.
- Uses **dual algorithms**:
  - **K-Means Clustering** for color segmentation.
  - **Robust Average Color** with outlier removal.
- Matches colors against official **BIS color codes**.

### 📚 Stripe Proportion Validation
- Confirms **each horizontal stripe** occupies **exactly one-third** of the flag's height.

### ⚙ Ashoka Chakra Validation
- **Position:** Detects if the Chakra is **perfectly centered** in the white stripe.
- **Spoke Count:** Verifies **exactly 24 spokes** using a **radial sampling technique**.
- **Dual detection system:**
  - **Hough Circle Transform**.
  - **Contour-based analysis** for cross-verification.

### 🧠 Dual-Algorithm Validation
- **Color detection** and **Chakra detection** both run on **two independent methods**, selecting the **more confident** result.

### 🖼 Pre-processing Pipeline
- Advanced enhancement pipeline for **noise reduction, contrast improvement, and color normalization** before analysis.

### 📊 Confidence Scores
- Each validation metric comes with a **confidence score**.

### 🐞 Debug Mode
- Generates **visual debug images** showing detected stripes, Chakra position, and analysis overlays.

### 🌐 Web Interface
- Built with **Streamlit**.
- Upload a flag image → Get a **comprehensive validation report**.

---

## 🚀 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Indian-Flag-Image-Validator.git
cd Indian-Flag-Image-Validator
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
# Activate venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application
```bash
streamlit run app.py
```

---

## 🖥 Usage
1. **Open** the Streamlit app in your browser.
2. **Upload** an image of the Indian flag.
3. Click **"Validate Flag"**.
4. View a **detailed validation report** with:
   - Pass/Fail for each criterion.
   - Confidence scores.
   - Optional debug visualization.

---

## 📂 Project Structure
```
📁 Indian-Flag-Image-Validator
│
├── app.py                # Streamlit web app
├── flag_validator.py     # Orchestrates validation process
├── color_analysis.py     # Color analysis logic
├── chakra_detector.py    # Chakra detection logic
├── image_processor.py    # Image preprocessing pipeline
│
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Ignore unnecessary files
```

---

## 🛠 Tech Stack
- **Python 3.9+**
- **OpenCV** – Image processing & computer vision
- **NumPy** – Numerical operations
- **scikit-learn** – K-means clustering
- **Streamlit** – Web interface
- **Matplotlib** – Debug visualization

---

## 📜 BIS Specifications Reference
- **Aspect Ratio:** 3:2
- **Colors:**
  - **Saffron:** IS: 1-1968
  - **White:** IS: 5-1968
  - **India Green:** IS: 33-1968
  - **Navy Blue (Chakra):** IS: 50-1968
- **Stripe Proportion:** Equal thirds
- **Ashoka Chakra:** 24 equally spaced spokes, centered in white stripe

---

## 📸 Example Output

| Metric | Status | Confidence |
|--------|--------|------------|
| Aspect Ratio | ✅ Pass | 99.2% |
| Saffron Color | ✅ Pass | 97.8% |
| Stripe Proportion | ⚠ Borderline | 92.4% |
| Chakra Spoke Count | ❌ Fail | 85.0% |

*Debug Mode Preview:*  
![Debug Output Example](docs/debug_example.png)

---

## 🏆 Why This Project Stands Out
- **Dual-algorithm verification** for robustness.
- **Confidence scoring** for transparency.
- **Professional-grade preprocessing pipeline**.
- **Streamlit-powered web interface** for usability.

---

## 📄 License
This project is released under the **MIT License**.

---

## 🤝 Contributing
Contributions are welcome!  
Please:
1. **Fork** the repo.
2. Create a new branch (`feature/amazing-feature`).
3. **Commit** your changes.
4. Submit a **pull request**.

---

## ⭐ Support the Project
If you found this useful, **give it a star** ⭐ on GitHub — it helps the project grow!
