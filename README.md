1. Dataset

Lyrics for the training data set was taken here:
    http://www.oldielyrics.com/p/paul_mccartney_wings.html
    http://www.oldielyrics.com/j/john_lennon.html

All the texts were processed then: lowered, some characters and non meaningful words were removed.  Features were extracted from text
using CountVectorizer of sklearn library ( http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html )

Lyrics for the test dataset was taken here:
    http://ultimateclassicrock.com/paul-mccartney-beatles-songs/
    http://ultimateclassicrock.com/top-10-john-lennon-beatles-songs/

2. Classification Model:

To classify Paul (0 - class) vs John (1 - class) SVM algorythm was chosen. Have tried both linear and gaussian kernels. And gaussian kerner gave a better score (0.95 vs 0.84).
Gaussin kernel requires feature parameters to be scaled so sklearn.preprocessing module was used. To choose coefficients C and gamma model was trained several times.

|C = 1     | C = 1    | C = 1   | C = 1 |C = 5     | C = 5    | C = 5   | C = 5 |C = 15    | C = 15   | C = 15  | C = 15 |
|g = 0.001 | g = 0.01 | g = 0.1 | g = 1 |g = 0.001 | g = 0.01 | g = 0.1 | g = 1 |g = 0.001 | g = 0.01 | g = 0.1 | g = 1  |
|----------|----------|---------|-------|----------|----------|---------|-------|----------|----------|---------|--------|
| 0.52631  | 0.47368  | 0.47368 | 0.4737|0.947368  | 0.47368  | 0.47368 | 0.4737|0.947368  | 0.47368  | 0.47368 | 0.47368|

|C = 20    | C = 20   | C = 20  | C = 20|C = 30    | C = 30   | C = 30  | C = 30 |C = 100   | C = 100  | C = 100 | C = 100|
|g = 0.001 | g = 0.01 | g = 0.1 | g = 1 |g = 0.001 | g = 0.01 | g = 0.1 | g = 1  |g = 0.001 | g = 0.01 | g = 0.1 | g = 1  |
|----------|----------|---------|-------|----------|----------|---------|--------|----------|----------|---------|--------|
| 0.947368 | 0.47368  | 0.47368 | 0.4737|0.947368  | 0.47368  | 0.47368 | 0.4737 | 0.947368 | 0.47368  | 0.47368 | 0.47368|

So final C = 5, gamma = 0.001

Model Scores:

F1        =  0.947368421053
Recall    =  0.9
Precision =  1.0


3. Learning Curves