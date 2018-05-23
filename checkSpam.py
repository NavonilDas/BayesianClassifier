import sys
import Classify

classfier = Classify.SpamClassification()
classfier.readData()
if classfier.Predict(sys.argv[1]):
    print("SPAM")
else:
    print("NOTSPAM")