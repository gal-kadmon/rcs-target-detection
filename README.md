# RCS Target Detection

פרויקט זה עוסק בזיהוי סוגי מטרות (ציפור, כטב"ם, מטוס) על סמך נתוני **Radar Cross Section (RCS)** סינתטיים.  
הפרויקט כולל יצירת דאטה ואימון מודלים של **Random Forest** ו-**MLP** בעזרת Optuna.

---
## מבנה הפרויקט
```
rcs-target-detection/
│
├── main.py                
├── data/                   
├── graphs/                 
└── src/
    ├── generate_rcs_data.m 
    ├── train_mlp.py      
    └── train_random_forest.py 
```
---
## דרישות

- **Python 3.8+**  
- **Octave**  
- Python Libraries:  
  `numpy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib`, `seaborn`, `oct2py`

---
## הפעלה

```bash
python main.py
