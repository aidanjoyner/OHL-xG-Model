<img width="1805" height="708" alt="image" src="https://github.com/user-attachments/assets/98fcd509-40dc-4c80-be41-4016d5b6ba45" />

# Ontario Hockey League (OHL) Expected Goals (xG) Model 🏒

An end-to-end machine learning pipeline designed to quantify shot quality in the Ontario Hockey League (OHL). By analyzing over **200,000 distinct shot events**, this model assigns a probabilistic value to every shot becoming a goal, providing another baseline stat to base player & team performance off of


---

## 🥅 The Problem & Motivation
In junior hockey, raw goal totals can be deceptive due to high variance  and varying categorical factors such age/competition. Standard "shots on goal" metrics treat a point-shot through a screen the same as a breakaway or rebound into an opening net, failing to provide enough insight on what's going on over a full 60 minute game. 

This project was built to:

* **Provide publicly available expected goal data & Bridge the Analytics Gap**: Make commonly privated data available for public view, giving fans an accessible way to learn more about player & team performance in the OHL through an objective model. 
* **Isolate Finishing & Goaltending Ability**: Identify which players consistently outperform their expected totals, in terms of both scoring goals on the skater side, and saving on the goaltender side. Shooting can also be largely luck based, so xG can help identify if a player is simply in a shooting slump, or if strong/poor shooting performance is a common occurence based on season-to-season results
* **Evaluate Team Structures**: The model quantifies how well teams limit high-danger scoring chances, and produce them themsleves. It can also depict situational strengths, such as powerplay/penalty kill generation/suppression, to show team strenghts/weaknesses. 

## 🛠️ The Process

### 1. Data Pipeline & Scraping
* **Source**: Scraped 200,000+ distinct shot events from OHL JSON feeds using **Python**.
* **ETL**: Cleaned and structured raw nested data into a tabular format using **Pandas**, with **Excel** used for intermediate data auditing and validation.

### 2. Feature Engineering
Created spatial and contextual metrics to capture the "danger" of a shot:
* **Spatial**: Euclidean distance from the net, shot angle
* **Context**: Rebound shots (<3 seconds since last shot), game state (Even Strength, PP, PK, EN), Game score, Clock Time. 

### 3. Modeling & Validation
* **Algorithm**: **XGBoost Classifier**. This gradient-boosted tree approach was selected for its ability to capture non-linear relationships between coordinates and goal probability, outperforming the Logistic Regression baseline I originally tested.
* **Validation Strategy**: Employed **Leave-One-Group-Out (LOGO) Cross-Validation** based on the `season` column. This ensures the model generalizes to new seasons and prevents temporal data leakage.
* **Calibration**: Applied **Isotonic Regression** to the raw model outputs. This ensures that the predicted probabilities are "reliable". i.e., if a group of shots has a 15% xG, they should result in a goal 15% of the time in reality.
* **Feature Importance**: The models big 3 features are distance from goal, angle from goal and if the shot was a rebound (in order). From a logical standpoint, these results make tons of sense as the generally the closer the shot is to being right in front of the next, the highe the chance of it going in. 
* **Results**: The model achieves an ROC-AUC of 0.73 and Brier of 0.087. There's clearly room for improvment, but with not having full shot attempt data and missing key context variables such as shot type in recent OHL seasons, the model performs as strongly as possible given its data limitations. 
* **Next Steps**: If more data surrounding each shot on goal or full shot attempt data is added, I can continue to develop new features accordingly. A few things that could be addressed as well are time since last event (somewhat addressed with rebound shots), as well as +/- attribution on goals to calculate player goals for & goal against rate & relative to team data.


---

## 📈 Visualizations & Insights
The model's output is visualized through interactive dashboards to make the data accessible for all.

* **Interactive Dashboard**: https://public.tableau.com/app/profile/aidan.joyner

---

## 📂 Repository Structure
* `src/`: Core Python scripts for scraping, cleaning, model training & stat building.
* `models/`: Saved XGBoost weights and the Isotonic Calibrator.
* `data/`: Data schema and a `sample_data.csv` (Full dataset excluded due to size).
* `requirements.txt`: List of libraries used
