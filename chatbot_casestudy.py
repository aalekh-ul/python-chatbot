import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json
import os
from tkinter import *

stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tk_words = nltk.word_tokenize(pattern)
        words.extend(tk_words)
        docs_x.append(tk_words)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])


words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    tk_words = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in tk_words:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)



tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

if os.path.exists("model.tflearn.meta"):
	model.load("model.tflearn")
else:
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")


def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]
	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1
  		      
	return numpy.array(bag)


root = Tk()
root.geometry('500x900')
root.title('ChatBot')

img = PhotoImage(file="bot.png")
photol = Label(root,image=img,borderwidth=7, relief="raised")
photol.pack(pady=5)

l1 = Label(root,text="AI CHATBOT",font=("Verdana",20),bg="orange",borderwidth=7, relief="raised")
l1.pack()

frame = Frame(root)
sc = Scrollbar(frame)

msgs=Listbox(frame,width=80,height=20)
sc.pack(side=RIGHT ,fill=Y)

msgs.pack(side=LEFT,fill=BOTH,pady=10)
frame.pack()

textf = Entry(root,font=("Verdana",10))
textf.pack(fill=X,pady=15)


def chat(event):
    inp = textf.get()
    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    ans = random.choice(responses)
    
    msgs.insert(END,"You: "+inp)
    msgs.insert(END,"Bot :"+ans)
    textf.delete(0,END)




btn=Button(root,text="Ask from bot",font=("Verdana",20),command=chat)
root.bind("<Return>",chat)

btn.pack()
root.mainloop()







