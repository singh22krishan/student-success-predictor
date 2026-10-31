# Model Experiment Results

Recorded from the original executed TensorFlow/Keras training notebook.

- UCI dataset 697: Predict Students' Dropout and Academic Success
- Rows: 4,424
- Enrollment-only features: 24
- Classes: Dropout, Enrolled, Graduate
- Split: 80% initial train / 20% test, with 10% of initial train used for validation
- Test accuracy: **0.5819**
- Macro F1: **0.5439**

Class report from the recorded run:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Dropout | 0.67 | 0.52 | 0.59 | 284 |
| Enrolled | 0.29 | 0.46 | 0.36 | 159 |
| Graduate | 0.70 | 0.66 | 0.68 | 442 |

The portfolio UI uses these recorded values instead of the unrelated hard-coded demo percentages that existed in the classroom frontend.
