# RCS Target Detection

פרויקט זה עוסק בזיהוי סוגי מטרות (Small, Medium, Large) על סמך נתוני **Radar Cross Section (RCS)** סינתטיים.  
הפרויקט כולל יצירת דאטה ואימון מודלים של **Random Forest** ו-**MLP** בעזרת Optuna.

---
## מבנה הפרויקט
```
rcs-target-detection/
│
├── main.py                 # סקריפט ראשי, מריץ את שאר הסקריפטים
├── data/                   # דאטה שנוצרת שמורה כאן
├── graphs/                 # גרפים שנוצרים שמורים כאן
└── src/
    ├── generate_rcs_data.m 
    ├── train_mlp.py      
    └── train_random_forest.py 
```
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
