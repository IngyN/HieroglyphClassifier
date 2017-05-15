**********************************************************
#Hieroglyph classifier
**********************************************************
##Ingy Nazif and Alia Hassan
**********************************************************

For this project, we trained two classifiers, one for handwritten glyphs (colored/grayscale), and one for handwritten glyphs taken from handwritten Egyptology books (Raymond Faulkner's *Concise Dictionary of Middle Egyptian* and DeBuck's *Egyptian Reading Book*).

Results for our training can be found in the `poster.pdf`

### The data
~1000-1200 training images
~300-400 validation images
  - heiro_train/heiro_val: images for training/validation of only textbook glyphs
  - heiro_train_handwriting/heiro_val_handwriting: images for training/validation of handwriting and some textbook handwriting
  - heiro_train_textbook/heiro_val_textbook: images for training/validation for textbook and around 4 handwritten glyphs.

### Weights for handwritten glyphs
https://drive.google.com/file/d/0B1fGlMcJ9Fa2ZTdDV0ZIUjRPMzA/view?usp=sharing
(The model architecture is Keras Xception)

### Weights for textbook glyphs
https://drive.google.com/file/d/0B1fGlMcJ9Fa2Y3RPYUdtUkNRbTA/view?usp=sharing
(The model architecture can be loaded from the `augmented_model_architecture.json` file)
This model was taken from another repository that was used for the Kanji character set:
  -https://github.com/KyotoSunshine/CNN-for-handwritten-kanji 
