import fastText
import re

def load_data(path):
    file = open(path, "r",encoding="utf-8")
    data = file.readlines() 
    return [line.split("\t") for line in data]     

def save_data(path,data):
    with open(path, 'w',encoding="utf-8") as f:        
        f.write("\n".join(data))

def train():
    traning_parameters = {'input': 'fasttext.train', 'epoch': 60, 'lr': 0.01, 'wordNgrams': 1, 'verbose': 2, 'minCount': 1, 'loss': "ns",
                        'lrUpdateRate': 100, 'thread': 1, 'ws':5, 'dim': 300}  
    model = fastText.train_supervised(**traning_parameters)
    model.save_model("model.bin")            
    return model

def replaceNumbers(text):
        text = text.replace("0"," null ")
        text = text.replace("1"," eins ")
        text = text.replace("2"," zwei ")
        text = text.replace("3"," drei ")
        text = text.replace("4"," vier ")
        text = text.replace("5"," fünf ")
        text = text.replace("6"," sechs ")
        text = text.replace("7"," sieben ")
        text = text.replace("8"," acht ")
        text = text.replace("9"," neun ")
        return text

def loadSmileyData(path):
        file = open(path, "r",encoding="utf-8")
        data = file.readlines()
        data = [line.replace("\n","") for line in data]
        return [line.split("\t") for line in data]   

simleys = loadSmileyData("data/custom-emo.txt")

def replaceSmiley(text):
        for simley in simleys:
                text = text.replace(simley[0],' '+simley[1].lower()+' ')
        return text

cleanChars = re.compile(r'[^a-züöäÖÜÄß ]', re.MULTILINE)
def preprocess(line):
        line = replaceSmiley(line)
        line = line.strip().lower()        
        line = " ".join([word for word in line.split() if word[0] is not "@"])        
        line = line.replace("."," ").replace("#","").replace("@"," ").replace(":","").replace(",","").replace("|","").replace("("," ").replace(")"," ").replace("-"," ").replace("/"," ").replace("!","").replace(";","").replace("\"","").replace("="," ")
        line = replaceNumbers(line)        
        #line = line.replace(" u "," und ")
        #line = cleanChars.sub('', line)
        return line

def test(model):
    f1_score = lambda precision, recall: 2 * ((precision * recall) / (precision + recall))
    nexamples, recall, precision = model.test('fasttext.test')     
    print (f'recall: {recall}' )
    print (f'precision: {precision}')
    print (f'f1 score: {f1_score(precision,recall)}')
    print (f'Number of examples: {nexamples}')

def transform(input_file, output_file):
    # load data
    data = load_data(input_file)
    # transform it into fasttext format __label__other have a nice day
    data = [f"__label__{line[1]}\t{preprocess(line[0])}" for line in data]
    # and save the data
    save_data(output_file,data)

if __name__ == "__main__":
    transform("data/germeval2018.training.txt","fasttext.train")
    transform("data/germeval2018.test.txt","fasttext.test")

    # train the model
    model = train() 
    test(model)
        
