# ML-Lab-brevin07

---

## Requirements & Tasks

### Task 1: A Lesson on Classification (Extra Credit)
- **Objective:** Complete the classification section in [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/).
- **Deliverables:**
  - A complete Jupyter Notebook (saved as `ML_Crash_Course_Extra_Credit.ipynb` in the `notebooks/` folder) including all exercises.
  - A screenshot (saved in the `assets/` folder as `quiz_screenshot.png`) showing the final quiz results.
- **Note:** This material is supplementary to the main Dash app.

### Task 2: Decide on a Classification Problem
- **Chosen Problem:** Classifying tumors as malignant or benign using the Breast Cancer Wisconsin dataset.
- **Implementation:**  
  - The dataset is pre-labeled and is loaded from scikit-learn.
  - PCA is applied to reduce the multi-dimensional data to 2 dimensions for visualization purposes.
- **UI:** A new option is available in the dataset dropdown labeled “Breast Cancer”.

### Task 3: Apply an Appropriate Classification Algorithm
- **Algorithm:** The project uses an SVM classifier.
- **Evaluation:**  
  - The SVM model is trained using the selected dataset.
  - Model performance is visualized with the decision boundary, ROC curve, and confusion matrix.
- **Outcome:** The model’s performance (accuracy, AUC, etc.) is computed and displayed to verify performance above random chance.

### Task 4: Put Together a Visual Dashboard to Explore Parameter-Tweaking
- **Dashboard Features:**
  - **Interactive Controls:**
    - Dataset selection (Moons, Circles, Linearly Separable, Breast Cancer)
    - Sample size and noise level for dataset generation
    - SVM hyperparameters: kernel type, cost (C), gamma, degree, and threshold
    - A button to reset the decision threshold
  - **Output Visuals:**
    - Decision boundary visualized as a contour plot
    - ROC curve for performance evaluation
    - Confusion matrix (displayed as a pie chart in the sample, with room for enhancement using a table)
- **Implementation:**  
  The interactive components, callbacks, and visual outputs are defined in `app.py` (with layout elements using custom components defined in `dash_reusable_components.py` and figure-generation functions in `figures.py`).

---

## Installation

### Dependencies

Ensure you have Python 3.x installed. The required packages are:
- dash
- numpy
- scikit-learn
- plotly
- colorlover

Install these using pip:
```bash
pip install dash numpy scikit-learn plotly colorlover




&#9744;
&#9745;


- &#9744; Multi-file structure we practiced in Project C and learned again in this TBoD chapter.
- &#9744; Reusable component definitions and multiples of actual components that are built using the reusable component definitions.
- &#9744; Title that states the classification problem and the subtitle that provides class and author (your) information.
- &#9745; Components that allow interaction in the following manner:
  - &#9745; Input components that allow changing at least two different parameters of your algorithm (e.g., sample size, test-train split, threshold, etc.)
  - &#9745; Output components that visualize the result of your classification (e.g., contour plot in the SVM demo app)
  - &#9745; ROC curve similar (or identical) to the one that was shown in the SVM demo.
  - &#9745; Confusion matrix, but an actual table (similar to the one from Google's crash course), not a Pie Chart.
  - &#9745; A button that allows the user to set the parameters back to the defaults (of your choice; e.g., the parameters for best accuracy).