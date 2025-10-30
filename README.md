# RCS Target Detection

פרויקט זה עוסק בזיהוי סוגי מטרות (Small, Medium, Large) על סמך נתוני **Radar Cross Section (RCS)** סינתטיים.  
הפרויקט כולל יצירת דאטה, גרפים של התפלגות SNR, ואימון מודלים של **Random Forest** ו-**MLP** להשוואת ביצועים.

---

## מבנה הפרויקט

rcs-target-detection/
│
├── main.py # סקריפט ראשי: יוצר דאטה, גרפים ומאמן מודלים
├── data/ # קבצי CSV עם נתוני RCS
├── graphs/ # גרפים שנוצרים (SNR distribution, confusion matrices)
└── src/
├── generate_rcs_data.m # פונקציה ליצירת דאטה סינתטי ב-Octave
├── train_mlp.py # אימון MLP עם Optuna
└── train_random_forest.py # אימון Random Forest עם Optuna


---

## דרישות

- **Python 3.8+**  
- **Octave**  
- ספריות Python:  
  `numpy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib`, `seaborn`, `oct2py`

---

## הפעלה

```bash
python main.py
