import math     # import math module for log-sum-exp method
import pickle

class SpamClassification:
    def __init__(self):
        self.noOFSpamMessages = 0.0      # Total number of Spam Messages
        self.totalMessages = 0.0         # Total Messages in the Data
        self.pSpam = 0.0                 # Probability of Spam
        self.pNotSpam = 0.0              # Probability of Not a Spam
        self.spamWords = {}              # Dictonary of Spam Words
        self.notSpamWords = {}           # Dictonary of Not a Spam Word
        self.totalWords = 0.0            # Total no of Non repeated Words
        self.nSpamWords = 0.0            # Total number of Spam Words
        self.nNotSpamWords = 0.0         # Total number of not a Spam word

        self.trainingData = list()       # Data for the Training

        self.threshold = 0.49            # Select three

        self.alpha = 1.0                 # Laplace Smoothing Factor
    #function to set laplace smoothing Factor
    def setAlpha(self,val):
        self.alpha = val
    # Function to set Threshold if needed
    def setThreshold(self,val):
        self.threshold = val
    # Function return the natural log ln
    def ln(self,x):
        return math.log(x)
    # This function Prepares the words dictonary of spam Words and non Spam Words
    # an it Calculates the total no of Spam Words (nSpamWords)
    # and Calculates the total no of non Spam Words (nNotSpamWords)
    def ProcessText(self,text,label):
        for word in text:
            if label == 1:
                if word in self.spamWords:
                    self.spamWords[word] += 1.0
                else:
                    self.spamWords[word] = 1.0
                self.nSpamWords += 1.0
            else:
                if word in self.notSpamWords:
                        self.notSpamWords[word] += 1.0
                else:
                    self.notSpamWords[word] = 1.0
                self.nNotSpamWords += 1
    # This function Finds the Conditional Probability of a Word given that it is spam or not
    def conditionalWord(self,word,isSpam):
        # global spamWords
        # global notSpamWords
        # global totalWords
        self.totalWords = self.nSpamWords+self.nNotSpamWords
        if isSpam:
            if word not in self.spamWords:
                self.spamWords[word] = 1.0
            return (self.ln(self.spamWords[word]+self.alpha) - self.ln(self.nSpamWords+self.alpha*self.totalWords))
        else:
            if word not in self.notSpamWords:
                self.notSpamWords[word] = 1.0
            return (self.ln(self.notSpamWords[word]+self.alpha) - self.ln(self.nNotSpamWords+self.alpha*self.totalWords))
    # This function computes the Conditional probality of text given that it is spam or not
    def conditionalText(self,text,isSpam):
        result = 0
        for word in text:
            result += self.conditionalWord(word,isSpam)
        return result

    def Train(self,trainingData):
        self.trainingData = trainingData
        self.totalMessages = len(trainingData)
        for data in trainingData:
            if data['label'] == 1:
                self.noOFSpamMessages += 1
            self.ProcessText(data['text'],data['label'])
        self.pSpam = self.noOFSpamMessages/self.totalMessages
        self.pNotSpam = (1-self.pSpam)
        
    def Predict(self,text): # function Predict wether the text is Spam or Not
        ps = self.ln(self.pSpam)
        pns = self.ln(self.pNotSpam)
        ps += self.conditionalText(text,True)
        pns += self.conditionalText(text,False)
        # Compute Log-Sum-Exp
        # Sum = pns - ps
        # p = math.exp(Sum)
        # p = 1/(1+p)
        return ps > pns
    # Function to Save the Data to a File 
    def saveData(self):
        f = open("saved.bin","w")
        f.write(str(self.threshold))
        f.write('\n')
        f.write(str(self.pSpam))
        f.write('\n')
        f.write(str(self.pNotSpam))
        f.close()
        with open('spamWords.bin','wb') as f:
            pickle.dump(self.spamWords,f,pickle.HIGHEST_PROTOCOL)
        with open('notSpamWords.bin','wb') as f:
            pickle.dump(self.notSpamWords,f,pickle.HIGHEST_PROTOCOL)
    # Function to Read the Data From the File
    def readData(self):
        try:
            f = open("saved.bin","r")
            self.threshold = float(f.readline())
            self.pSpam = float(f.readline())
            self.pNotSpam = float(f.readline())
            f.close()
            with open('spamWords.bin','rb') as f:
                self.spamWords = pickle.load(f)
            with open('notSpamWords.bin','rb') as f:
                self.notSpamWords = pickle.load(f)
            self.nSpamWords = len(self.spamWords)
            self.nNotSpamWords = len(self.notSpamWords)
        except:
            print("Error: Classifier is Not Trained!")