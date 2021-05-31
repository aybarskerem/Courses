import sys
import importlib
import time
import threading

# for gui2
from tkinter import *  
import tkinter
from tkinter.ttk import *

currentFunction = "None"
preprocessed=False
NLP_executed=False
DL_executed=False
training_data_number = 0 # the rest is test_data_number
number_of_epochs=10 # default is 10
total_word2vec_reading_limit = 0
test_sample_filename="Sheet_1"
chosen_test_sample_index=-1 # use a checkbox to test if one_test_sample exists, if so use only this sample instead of whole test instances(rerun run_NLP)
use_chosen_test_sample = False # if we use one_test_sample_index, then use the sample at that index. By default we do not use it
show_wordcloud = False

NLP_Accuracy_Text = "" # emtpy string by default, run_DLP will fill it
Preprocessing_Accuracy_Text = ""
Selected_Test_Sentence_Text=""
DL_GroundTruth_Classes_Text = ""
DL_Predicted_Classes_Text = ""
DL_Accuracy_Text=""

Manual_Test_Sentence_Text = ""
Manual_Test_Sentence_Exist = False # if exists override chosen test sample index etc.
Manual_Test_Sentence_Class="not_flagged" # False=not_flagged=0

DL_Method_Choice = ""

# To be able to run the script from anywhere, set the current working directory to where this script resides
import os
this_script_path = os.path.abspath(__file__)
this_script_dirname = os.path.dirname(this_script_path)
os.chdir(this_script_dirname)

def gui2():

  root = Tk()
  root.title("DEEP LEARNING TERM PROJECT")

  # on change dropdown value
  def change_dropdown(*args):
    global DL_Method_Choice
    print("Dropdown has become",  DL_Choices_Combobox.get()  )
    DL_Method_Choice = DL_Choices_Combobox.get() 

  def preprocess_data():
    global currentFunction
    currentFunction = "preprocess"

  def run_NLP():
    global currentFunction
    currentFunction = "run_NLP"

  def run_DL():
    global currentFunction
    currentFunction = "run_DL"

  def read_whether_to_show_wordcloud():
    global show_wordcloud
    print("Whether to show word cloud has become ", True if wordcloud_checkbox_var.get() else False  )
    show_wordcloud = True if wordcloud_checkbox_var.get() else False 

  def read_word2vec_limit():
    global total_word2vec_reading_limit
    print("word limit has become ", int(word2vec_limit_entry.get() ))
    total_word2vec_reading_limit = int(word2vec_limit_entry.get() )

  def read_training_data_number():
    global training_data_number
    print("training data number has become ", int(training_data_number_entry.get() ))
    training_data_number = int(training_data_number_entry.get() )

  def read_number_of_epochs():
    global number_of_epochs
    print("#of epochs is chosen as ", int(number_of_epochs_entry.get() ))
    number_of_epochs = int(number_of_epochs_entry.get() )
#### test sample related functions start ####
  def read_test_sample_filename():
    global test_sample_filename
    print("Test sample file name has become ", test_sample_filename_entry.get() )
    test_sample_filename = test_sample_filename_entry.get()
  def read_chosen_test_sample_index():
    global chosen_test_sample_index
    print("Chosen test sample index has become ", int(chosen_test_sample_index_entry.get() ))
    chosen_test_sample_index = int(chosen_test_sample_index_entry.get() )

  def read_whether_use_test_sample():
    global use_chosen_test_sample
    print("Should we use test sample chosen become ", True if chosen_test_sample_checkbox_var.get() else False  )
    use_chosen_test_sample = True if chosen_test_sample_checkbox_var.get() else False 
#### test sample related functions end ####

  def read_Manual_Test_Sentence_Text():
    global Manual_Test_Sentence_Text
    print("Manual Test sentence has become ", Manual_Test_Sentence_Entry.get() )
    Manual_Test_Sentence_Text = Manual_Test_Sentence_Entry.get() 

  def read_whether_use_Manual_Test_Sentence_Text():
    global Manual_Test_Sentence_Exist
    print("Should we use manual test sentence has become ", True if Manual_Test_Sentence_Entry_checkbox_var.get() else False  )
    Manual_Test_Sentence_Exist = True if Manual_Test_Sentence_Entry_checkbox_var.get() else False 

  def read_manuel_test_sentence_class():
    global Manual_Test_Sentence_Class
    print("Class of Manual test sentence chosen as ", "flagged" if Manual_Test_Sentence_Entry_Class_checkbox_var.get() else "not_flagged"  )
    Manual_Test_Sentence_Class = "flagged" if Manual_Test_Sentence_Entry_Class_checkbox_var.get() else "not_flagged"

#### 
  def update_preprocessing_result_label():
    Preprocessing_AccuracyLabel.configure(text=Preprocessing_Accuracy_Text)
    Preprocessing_AccuracyLabel.after(1000, update_preprocessing_result_label)

  def update_NLP_result_label():
    NLP_AccuracyLabel.configure(text=NLP_Accuracy_Text)
    NLP_AccuracyLabel.after(1000, update_NLP_result_label)

  def update_DL_Predicted_class_result_label():
    DL_Predicted_Class_Result_Label.configure(text=DL_Predicted_Classes_Text)
    DL_Predicted_Class_Result_Label.after(1000, update_DL_Predicted_class_result_label)

  def update_DL_Ground_Truth_class_result_label():
    DL_GroundTruth_Class_Result_Label.configure(text=DL_GroundTruth_Classes_Text)
    DL_GroundTruth_Class_Result_Label.after(1000, update_DL_Ground_Truth_class_result_label)

    

  def update_DL_Accuracy_result_label():
    DL_Accuracy_Result_Label.configure(text=DL_Accuracy_Text)
    DL_Accuracy_Result_Label.after(1000, update_DL_Accuracy_result_label)

  def update_Selected_Test_Sentence_Label():
    Selected_Test_Sentence_Label.configure(text=Selected_Test_Sentence_Text)
    Selected_Test_Sentence_Label.after(1000, update_Selected_Test_Sentence_Label)

  mainframe = Frame(root)
  mainframe.rowconfigure(0, weight = 1)
  mainframe.columnconfigure(0, weight = 1)
  mainframe.pack()


  choices = [ 'Choose the DL method to apply','Vanilla Fully Connected Network','Vanilla CNN','CNN','Vanilla RNN','LSTM','LSTM with CNN']

  DL_Choices_Combobox = Combobox(mainframe, values = choices,state="readonly",justify=CENTER,width=len(max(choices, key=len)))
  DL_Choices_Combobox.grid(row=0,column=2, sticky=(N,W,E,S) ) 
  DL_Choices_Combobox.bind("<<ComboboxSelected>>", change_dropdown)
   
  word2vec_limit_entry = Entry(mainframe, width=10)
  word2vec_limit_entry.grid(row=0,column=0,sticky=(N,W,E,S))
  word2vec_button = Button(mainframe,text="Save word2vec reading limit for GoogleNews-vectors-negative300.bin", command=read_word2vec_limit)
  word2vec_button.grid(row=0,column=1,sticky=(N,W,E,S))

  number_of_epochs_entry = Entry(mainframe, width=10)
  number_of_epochs_entry.grid(row=1,column=0,sticky=(N,W,E,S))
  number_of_epochs_button = Button(mainframe,text="Save #of epochs while fitting the selected DL method",command=read_number_of_epochs)
  number_of_epochs_button.grid(row=1,column=1,sticky=(N,W,E,S))

  csv_label=tkinter.Label(mainframe, text="***** BELOW ARE THE THREE INPUT .csv FILE PARAMETERS: *****")
  csv_label.grid(row=2,column=0, sticky=(N,W,E,S))

  test_sample_filename_entry = Entry(mainframe, width=10)
  test_sample_filename_entry.grid(row=3,column=0, sticky=(N,W,E,S))
  test_sample_filename_button = Button(mainframe,text="[OPTIONAL] Click to set the input .csv file name without the .csv part (the default is 'Sheet_1').",command=read_chosen_test_sample_index)
  test_sample_filename_button.grid(row=3,column=1, sticky=(N,W,E,S))

  training_data_number_entry = Entry(mainframe, width=10)
  training_data_number_entry.grid(row=4,column=0,sticky=(N,W,E,S))
  training_data_number_button = Button(mainframe,text="Click to set #of training data (use data for training up until this number [row index] in the input .csv file)",command=read_training_data_number)
  training_data_number_button.grid(row=4,column=1,sticky=(N,W,E,S))

  chosen_test_sample_index_entry = Entry(mainframe, width=10)
  chosen_test_sample_index_entry.grid(row=5,column=0, sticky=(N,W,E,S))
  chosen_test_sample_index_button = Button(mainframe,text="[OPTIONAL] Click to set 'test sample index (row index)' for the input .csv file to opt to use one test sentence only",command=read_chosen_test_sample_index)
  chosen_test_sample_index_button.grid(row=5,column=1, sticky=(N,W,E,S))

  chosen_test_sample_checkbox_var = IntVar()
  chosen_test_sample_checkbox = Checkbutton(mainframe,text="Check to use selected test sample (as determined by the 'test sample index')", command=read_whether_use_test_sample,variable=chosen_test_sample_checkbox_var)
  chosen_test_sample_checkbox.grid(row=6,column=0, sticky=(N,W,E,S))

  #########
  # button
  preprocess_button = Button(mainframe,text="Preprocess Data", command=preprocess_data)
  preprocess_button.grid(row=0,column=3, sticky=(N,W,E,S))

  NLP_button = Button(mainframe,text="Run NLP", command=run_NLP)
  NLP_button.grid(row=1,column=3, sticky=(N,W,E,S))

  DL_button = Button(mainframe,text="Run DL", command=run_DL)
  DL_button.grid(row=2,column=3, sticky=(N,W,E,S))

  wordcloud_checkbox_var = IntVar()
  wordcloud_checkbox = Checkbutton(mainframe,text="Check to show the 'WordCloud' at the end", command=read_whether_to_show_wordcloud,variable=wordcloud_checkbox_var)
  wordcloud_checkbox.grid(row=3,column=3, sticky=(N,W,E,S))

  ######################## DYNAMIC RESULT LABELS ##############3

  Manual_Test_Sentence_Entry = Entry(mainframe, width=10)
  Manual_Test_Sentence_Entry.grid(row=7,column=0,sticky=(N,W,E,S) )
  Manual_Test_Sentence_button = Button(mainframe,text="[OPTIONAL] Click to set a 'test sentence' manually instead of using the test sample(s) from the input .csv file", command=read_Manual_Test_Sentence_Text)
  Manual_Test_Sentence_button.grid(row=7,column=1,sticky=(N,W,E,S))


  Manual_Test_Sentence_Entry_checkbox_var = IntVar()
  Manual_Test_Sentence_Entry_checkbox = Checkbutton(mainframe,text="Check to use 'test sentence' entered above instead of using the input .csv file for the test samples", command=read_whether_use_Manual_Test_Sentence_Text,variable=Manual_Test_Sentence_Entry_checkbox_var)
  Manual_Test_Sentence_Entry_checkbox.grid(row=8,column=0, sticky=(N,W,E,S))

  Manual_Test_Sentence_Entry_Class_checkbox_var = IntVar()
  Manual_Test_Sentence_Entry_Class_checkbox = Checkbutton(mainframe,text="Check if the correct class for the test sentence is 'flagged' (default is 'Not Flagged')", command=read_manuel_test_sentence_class,variable=Manual_Test_Sentence_Entry_Class_checkbox_var)
  Manual_Test_Sentence_Entry_Class_checkbox.grid(row=9,column=0, sticky=(N,W,E,S))

  Selected_Test_Sentence_Label = Label(mainframe, text=Selected_Test_Sentence_Text,borderwidth=1,width=150 )
  Selected_Test_Sentence_Label.after(1000, update_Selected_Test_Sentence_Label) # call update_DL_result_label after milisecond indicated in 1st parameter
  Selected_Test_Sentence_Label.grid(row=12,column=0,sticky=(N,W,E,S), columnspan=4)

  Preprocessing_AccuracyLabel = Label(mainframe, text=Preprocessing_Accuracy_Text,borderwidth=1, width=150 )
  Preprocessing_AccuracyLabel.after(1000, update_preprocessing_result_label) # call update_preprocessing_result_label after milisecond indicated in 1st parameter
  Preprocessing_AccuracyLabel.grid(row=14,column=0,sticky=(N,W,E,S), columnspan=4)


  NLP_AccuracyLabel = Label(mainframe, text=NLP_Accuracy_Text,borderwidth=1,width=150 )
  NLP_AccuracyLabel.after(1000, update_NLP_result_label) # call update_NLP_result_label after milisecond indicated in 1st parameter
  NLP_AccuracyLabel.grid(row=16,column=0,sticky=(N,W,E,S), columnspan=4)

  DL_Accuracy_Result_Label = Label(mainframe, text=DL_Accuracy_Text,borderwidth=1,width=150 )
  DL_Accuracy_Result_Label.after(1000, update_DL_Accuracy_result_label) # call update_DL_result_label after milisecond indicated in 1st parameter
  DL_Accuracy_Result_Label.grid(row=18,column=0,sticky=(N,W,E,S), columnspan=4)

  DL_GroundTruth_Class_Result_Label = Label(mainframe, text=DL_GroundTruth_Classes_Text,borderwidth=1,width=150 )
  DL_GroundTruth_Class_Result_Label.after(1000, update_DL_Ground_Truth_class_result_label) # call update_DL_result_label after milisecond indicated in 1st parameter
  DL_GroundTruth_Class_Result_Label.grid(row=20,column=0,sticky=(N,W,E,S), columnspan=4)

  DL_Predicted_Class_Result_Label = Label(mainframe, text=DL_Predicted_Classes_Text,borderwidth=1,width=150 )
  DL_Predicted_Class_Result_Label.after(1000, update_DL_Predicted_class_result_label) # call update_DL_result_label after milisecond indicated in 1st parameter
  DL_Predicted_Class_Result_Label.grid(row=22,column=0,sticky=(N,W,E,S), columnspan=4)


  root.mainloop()
 

def main(argv):

  global Preprocessing_Accuracy_Text
  global NLP_Accuracy_Text
  global DL_GroundTruth_Classes_Text
  global DL_Predicted_Classes_Text
  global DL_Accuracy_Text
  global DL_Method_Choice

  global Selected_Test_Sentence_Text
  
  global Manual_Test_Sentence_Text
  global Manual_Test_Sentence_Class # False=not_flagged=0

  global show_wordcloud
  global currentFunction


  t1 = threading.Thread(target=gui2)
  t1.start()

  preprocessed = False
  NLP_executed = False
  while True:


    time.sleep(3)
    if(currentFunction == "preprocess"):

      try:

        print("Starting preprocessing the data...")
        import numpy as np # linear algebra
        import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

        print("The input .csv filename is: " + test_sample_filename)
        df = pd.read_csv("../input/"+test_sample_filename+".csv",usecols=['response_id','class','response_text'],encoding='latin-1')

        ## DELETE DL globals ##
        DL_GroundTruth_Classes_Text = ""
        DL_Predicted_Classes_Text = ""
        DL_Accuracy_Text = ""

        ## DELETE NLP globals ##
        NLP_Accuracy_Text = ""

        ## DELETE Preprocessing global ##
        Preprocessing_Accuracy_Text = ""

        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import nltk
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        lemma = nltk.wordnet.WordNetLemmatizer()

        from nltk.corpus import stopwords

        def is_noun(tag):
            return tag in ['NN', 'NNS', 'NNP', 'NNPS']

        def isWordNoun(param_word):
          for word, tag in nltk.pos_tag(nltk.word_tokenize(param_word)): # there is only one item
            return is_noun(tag)

        def lemmatize_word(param_word):
            retVal=""
            for word, tag in nltk.pos_tag(nltk.word_tokenize(param_word) ):
                wntag = tag[0].lower()
                wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
                if wntag is None:
                    retVal = word
                else:
                    retVal = lemma.lemmatize(word, wntag)            
                return retVal

        def isLemmatizationNoun(param_word):
            retVal=""
            for word, tag in nltk.pos_tag(nltk.word_tokenize(param_word.upper) ) :
                wntag = tag[0].lower()
                wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
                if not wntag:
                    retVal = False
                else:
                    resultingWord = lemma.lemmatize(word, wntag)
                    retVal = isWordNoun(resultingWord)
                return retVal
            

        if show_wordcloud:
          def cloud(dataframe,flagged=True):
              noun_flagged=[]
              noun_not_flagged=[]
              my_str=""
              for index,row in dataframe.iterrows():
                  noun_flagged.extend([lemmatize_word(word) for word in ( nltk.word_tokenize(row[2]) ) 
                                       if isWordNoun(lemmatize_word(word)) and row[1] == "flagged" and word!= 'i'] )
                  noun_not_flagged.extend([lemmatize_word(word) for word in ( nltk.word_tokenize(row[2]) )
                                           if isWordNoun(lemmatize_word(word)) and row[1] == "not_flagged" and word!='i' ] )
              noun_flagged_str=" ".join(noun_flagged).upper()
              noun_not_flagged_str=" ".join(noun_not_flagged).upper()
              
              if flagged:
                  wordcloud_flagged = WordCloud(background_color="orange").generate(noun_flagged_str)
                  plt.imshow(wordcloud_flagged)
              else:
                  wordcloud_not_flagged = WordCloud(background_color="green").generate(noun_not_flagged_str)
                  plt.imshow(wordcloud_not_flagged)
                  
              plt.axis("off")
              if flagged:
                plt.title("Preprocessing Cloud for Flaggeds")
              else:
                plt.title("Preprocessing Cloud for Not Flaggeds")
              plt.show()

          cloud(df,True)
          cloud(df,False)


        import operator
        from pprint import pprint
        import nltk
        from nltk.stem.snowball import SnowballStemmer
        import string

        # download necessary modoules from the nltk corpus
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

        def get_sentence_words(dataframe,flag_state=None): 
            sentence_words=[]

            if flag_state is not None:
                for index,row in dataframe.iterrows():
                    new_sentence = [lemmatize_word(word).upper().translate(str.maketrans('','',string.punctuation))
                                for word in ( nltk.word_tokenize(row[2]) ) if isWordNoun(lemmatize_word(word)) and word!='i'
                                and row[1] == flag_state]
                    if new_sentence:
                        sentence_words.append(new_sentence)
            else:
                for index,row in dataframe.iterrows():
                    new_sentence = [lemmatize_word(word).upper().translate(str.maketrans('','',string.punctuation))
                                for word in ( nltk.word_tokenize(row[2]) ) if isWordNoun(lemmatize_word(word)) and word!='i']
                    if new_sentence:
                        sentence_words.append(new_sentence)
            return sentence_words

        def my_word_freq_bag(dataframe,flag_state=None):
            counts = dict()
            bag = []
            counter = 0

            if flag_state is not None:
                for index,row in dataframe.iterrows():
                    bag.extend([lemmatize_word(word).upper().translate(str.maketrans('','',string.punctuation))
                                for word in ( nltk.word_tokenize(row[2]) ) if isWordNoun(lemmatize_word(word)) and word!='i'
                                and (row[1] == flag_state) ]) # if None directly include, act according to flag
            else:
                #it is not iterable, instead just a sentence
                bag.extend([lemmatize_word(word).upper().translate(str.maketrans('','',string.punctuation))
                            for word in ( nltk.word_tokenize(dataframe) ) if isWordNoun(lemmatize_word(word)) and word!='i']) 
                  
            for word in bag:
              if word != "":
                counts[bag[counter]] = counts.get(word,0)+1        
                counter += 1

            return counts


        TEST_SAMPLES_STARTING_INDEX=training_data_number
        train_df = df[:TEST_SAMPLES_STARTING_INDEX]

        # Test sentence typed manually > Chosen sample > Normal ( Using rest of training_data_number as test sentences)
        if Manual_Test_Sentence_Exist:
          test_df = pd.DataFrame({ 'response_id' : ['response_test_sentence'], 
                          'class' :[ Manual_Test_Sentence_Class],
                         'response_text' : [ Manual_Test_Sentence_Text ] })
          #print("test df is: \n ", test_df.head() )
        elif use_chosen_test_sample:
          print("The chosen test sentence is: ",df[chosen_test_sample_index:chosen_test_sample_index+1]['response_text'].iloc[0])
          Selected_Test_Sentence_Text = "The chosen test sentence is: {}".format( df[chosen_test_sample_index:chosen_test_sample_index+1]['response_text'].iloc[0] )
          test_df = df[chosen_test_sample_index:chosen_test_sample_index+1] # do error handlings ASAP
        else:
          test_df = df[TEST_SAMPLES_STARTING_INDEX:]


        train_freq_bag_flagged = my_word_freq_bag(train_df,"flagged")
        print(get_sentence_words(train_df,"flagged") )
        train_freq_bag_not_flagged = my_word_freq_bag(train_df,"not_flagged")

        print("********************************")
        print(train_freq_bag_not_flagged)

        import io
        import sys
        from gensim.models import KeyedVectors
        from gensim.models import Word2Vec
        import numpy as np

        global_word2vec_model = KeyedVectors.load_word2vec_format('../input/GoogleNews-vectors-negative300.bin', binary=True, 
                limit = total_word2vec_reading_limit
                )

        def score(a, b):
            #I use key upper since I expect inputs of the form lower case always
                #print(lemma.lemmatize(key).upper().translate(str.maketrans('','',string.punctuation) ) )
            mykey=[sum(value * b.get(lemmatize_word(key).upper().translate(str.maketrans('','',string.punctuation) ),0 ) 
                       for key,value in a.items() ) ]
            return(mykey)

        def classifier_accuracy(test_df,train_dataframe_flagged_bag,train_dataframe_not_flagged_bag):
            
            flagged_bag_dict = dict(sorted(train_dataframe_flagged_bag.items(), key=operator.itemgetter(1), reverse=True)[:15])
            not_flagged_bag_dict = dict(sorted(train_dataframe_not_flagged_bag.items(), key=operator.itemgetter(1), reverse=True)[:15])
            print("flagged")
            print(flagged_bag_dict)
            print("not flagged")
            print(not_flagged_bag_dict)
            df_length = len(test_df.index)
            number_of_accurate_results = 0

            for i in range(df_length):   

                text_class = test_df['class'].iloc[i]#[index]
                text = test_df['response_text'].iloc[i]#[index]
                text_bag = my_word_freq_bag(text) 

                score_flagged = score(text_bag, flagged_bag_dict)
                score_not_flagged = score(text_bag, not_flagged_bag_dict)

                #print(text_bag)
                #print("************************")          
                #print(flagged_bag_dict)
                #print("************************")
                #print(not_flagged_bag_dict)
                #print("score flagged is: ", score_flagged)
                #print("score not flagged is: ", score_not_flagged)
                #if(score_flagged>score_not_flagged):
                    #print("{} is classified as flagged whereas it was {}".format(i,text_class) )
                #elif(score_flagged<score_not_flagged):
                #    print("{} is classified as not flagged whereas it was {} iken".format(i,text_class) )
                #else:
                #    print("{} could not classify it very well :) " )

                if score_flagged>score_not_flagged and text_class == "flagged":
                    number_of_accurate_results+=1
                elif score_flagged <= score_not_flagged and text_class == "not_flagged":
                    number_of_accurate_results+=1

            global Preprocessing_Accuracy_Text
            Preprocessing_Accuracy_Text = "Word Frequency Final accuracy is: {} %".format(  (number_of_accurate_results/df_length)*100) 
            print("Word Frequency accuracy is: ",(number_of_accurate_results/df_length)*100, "%")
            return (number_of_accurate_results/df_length)*100

        classifier_accuracy(test_df, train_freq_bag_flagged, train_freq_bag_not_flagged)
        preprocessed = True
        currentFunction = "None"
        print("Completed preprocessing the data...")

      except Exception as ex:
        import traceback
        print("Encountered an error ",ex," while preprocessing the data, please preprocess again with right parameters...")
        print(traceback.format_exc())
        currentFunction = "None"
        continue

    elif preprocessed and currentFunction == "run_NLP":
    


      try:
        print("Starting running the NLP...")

        ## DELETE DL globals ##
        DL_GroundTruth_Classes_Text = ""
        DL_Predicted_Classes_Text = ""
        DL_Accuracy_Text = ""

        ## DELETE NLP globals ##
        NLP_Accuracy_Text = ""


        def get_vector_from_sentences(input_sentences):

            all_rows=[]
            inner_itr = 0
            for row in input_sentences:

                curr_word=0
                row_length=len(row)
                avg_row_vec_exist=False
                for x in row:
                    #if x in global_word2vec_model.vocab: # Gensim3.x way
                    if x in global_word2vec_model.key_to_index: # Gensim4 way
                        if not avg_row_vec_exist:
                            #avg_row_vector = np.array(global_word2vec_model[x],dtype=np.dtype('Float64'))
                            avg_row_vector = np.array(global_word2vec_model[x],dtype=np.dtype(float))
                            avg_row_vec_exist = True
                        else:
                            #avg_row_vector += np.array(global_word2vec_model[x],dtype=np.dtype('Float64'))
                            avg_row_vector += np.array(global_word2vec_model[x],dtype=np.dtype(float))
                        
                       
                    curr_word+=1
                    if curr_word == row_length:
                        break
                if avg_row_vec_exist:
                    avg_row_vector /= row_length
                else:
                    avg_row_vector = np.zeros((300,)) # if none of the word exist, assign avg vector as zero

                all_rows.append(avg_row_vector.tolist() )
                inner_itr+=1
      
            return all_rows

        def get_avg_vector_of_sentences(input_rows):

                  
            train_len = len(input_rows)
            avg = np.zeros((1,*(np.array(input_rows).shape[1:]) ))

            for row in input_rows:
                avg += np.array(row)
            avg /= train_len
            return avg


        def get_cos_similarity(v1,v2): # cosine distance formula =  x1 dot x2 / (|x1| |x2|)
            v1_loc = np.squeeze(np.array(v1) ) # shape(1,1,300) -> (300)
            v2_loc = np.squeeze(np.array(v2))
            return np.dot(v1_loc,v2_loc)/(np.linalg.norm(v1_loc )* np.linalg.norm(v2_loc) + 1e-10 ) #prevent zero division
                  
        def get_cos_similarity_of_all_test(avg_flagged,avg_not_flagged,test_df):
            test_sentences=get_sentence_words(test_df)
            all_test_rows_vectorized=get_vector_from_sentences(test_sentences)
            
            all_classification_results=[]
            accurate_results=0
            number_of_test_sentences = np.array(all_test_rows_vectorized).shape[0]
            for i in range(number_of_test_sentences):
                single_classification_result=None
    
                diff_flagged = get_cos_similarity(avg_flagged_NLP,all_test_rows_vectorized[i:i+1])
                diff_not_flagged = get_cos_similarity(avg_not_flagged_NLP,all_test_rows_vectorized[i:i+1])
                if diff_flagged > diff_not_flagged:
                    single_classification_result=1 #flagged
                else:
                    single_classification_result=0 #not_flagged
                all_classification_results.append(single_classification_result)  
                #print(i,"th diff with flagged is: ",diff_flagged)
                #print(i,"th diff with not_flagged is: ",diff_not_flagged)
                classes = ['not_flagged','flagged']
                #print("classification result is: ",single_classification_result, "whereas real result is: ",test_df['class'].apply(classes.index).iloc[i])
                if single_classification_result == test_df['class'].apply(classes.index).iloc[i]: #[i:i+1], also not iloc(80+i)
                    accurate_results+=1
         
            global NLP_Accuracy_Text
            NLP_Accuracy_Text = "NLP Final accuracy is: {} %".format(  (accurate_results/number_of_test_sentences)*100 )  
            print("NLP Final accuracy is: ",(accurate_results/number_of_test_sentences)*100, "%")
            return all_classification_results


            
        flagged_input_sentences = get_sentence_words(train_df,"flagged")
        not_flagged_input_sentences = get_sentence_words(train_df,"not_flagged")

        test_sentences=get_sentence_words(test_df)
        all_flagged_rows = get_vector_from_sentences(flagged_input_sentences)
        all_not_flagged_rows = get_vector_from_sentences(not_flagged_input_sentences)
        all_test_rows_vectorized=get_vector_from_sentences(test_sentences)
        #print("all test rows initial shape is: ",np.array(all_test_rows_vectorized).shape)

        #NLP
        avg_flagged_NLP = get_avg_vector_of_sentences(all_flagged_rows)
        avg_not_flagged_NLP = get_avg_vector_of_sentences(all_not_flagged_rows)


        all_results=get_cos_similarity_of_all_test(avg_flagged_NLP,avg_not_flagged_NLP,test_df)

        NLP_executed = True
        currentFunction = "None"
        print("Completed running the NLP...")

      except Exception as ex:
        print("Encountered an error ",ex," while running NLP on the data, please run NLP again with right parameters...")
        print(print(traceback.format_exc()))
        currentFunction = "None"
        continue

    elif preprocessed and NLP_executed and currentFunction == "run_DL":

      try:
        print("Started running the Deep Learning Methods on NLP Data...")
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SimpleRNN
        from keras.models import Model
        from keras.layers import Conv1D, MaxPooling1D,Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
        from keras.models import Sequential
        from keras.optimizers import Nadam

        try:
          from keras.utils import to_categorical
        except ImportError:
          from keras.utils.np_utils import to_categorical

        ## DELETE DL globals ##
        DL_GroundTruth_Classes_Text = ""
        DL_Predicted_Classes_Text = ""
        DL_Accuracy_Text = ""



        X1 = np.array(all_flagged_rows) #ilk 8 column
        X2 = np.array(all_not_flagged_rows)
        X = []
        X.extend(all_flagged_rows)
        X.extend(all_not_flagged_rows)
        X = np.array(X)


        #TEST
    
        X_test = np.array(all_test_rows_vectorized)


        Y1_list=np.ones((X1.shape[0],1) ).tolist()
        Y2_list=np.zeros((X2.shape[0],1) ).tolist()
        Y = []
        Y.extend(Y1_list)
        Y.extend(Y2_list)
        Y = np.array(Y)
        Y = to_categorical(Y,num_classes=2)

        classes = ['not_flagged','flagged']
        test_length = X_test.shape[0]
        Y_test = []
        for i in range(test_length):
          Y_test.append(test_df['class'].apply(classes.index).iloc[i]) #convert not_flagged and flagged to 0 and 1 respectively
        
        Y_test = np.array(Y_test)
        Y_test_categorical = to_categorical(Y_test,None)

        print("DL Method choice is: ", DL_Method_Choice)

        dropout_ratio=0.3
        filter_size = 4 # for CNN
        stride_amount=1
        no_of_words=15  #timesteps -> bu kadar sequence'i hatirlasin
        no_of_features=300 # her bir word'um bu kadar feature represent ediyor
        if DL_Method_Choice == "Vanilla Fully Connected Network":
            model = Sequential()
            model.add(Dense(300, kernel_initializer="uniform",input_shape=(300,),activation='relu'))
            model.add(Dense(100, kernel_initializer="uniform", activation='relu')) 
            model.add(Dropout(dropout_ratio))
            model.add(Dense(2,kernel_initializer="uniform",activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer=Nadam())
            model.fit(X,Y,epochs=number_of_epochs)

            
        elif DL_Method_Choice == "Vanilla CNN":
            model = Sequential()
            model.add(Conv1D(filters=16, kernel_size=filter_size,strides=stride_amount, activation='relu',  input_shape=(no_of_words,no_of_features) ))
            model.add(Conv1D(filters=32, kernel_size=filter_size,strides=stride_amount, activation='relu'))
            model.add(MaxPooling1D(2))
            model.add(Conv1D(filters=64, kernel_size=filter_size,strides=stride_amount, activation='relu'))
            model.add(GlobalMaxPooling1D()) # global pooling'ler timesteps'i yok ediyor , gl
            model.add(Dropout(dropout_ratio))
            model.add(Dense(2,kernel_initializer="uniform",activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer=Nadam())
            
            arrays = [X for i in range(no_of_words)]
            X=np.stack(arrays, axis=1)
            arrays_test = [X_test for i in range(no_of_words)]
            X_test=np.stack(arrays_test, axis=1)

            model.fit(X,Y,epochs=number_of_epochs)
                
        elif DL_Method_Choice == "CNN":
            model = Sequential()
            model.add(Conv1D(filters=16, kernel_size=filter_size,strides=stride_amount, activation='relu',  input_shape=(no_of_words,no_of_features) ))
            model.add(Dense(512, kernel_initializer="uniform",activation='relu') )
            model.add(Conv1D(filters=32, kernel_size=filter_size,strides=stride_amount, activation='relu'))
            model.add(MaxPooling1D(2))
            model.add(Conv1D(filters=64, kernel_size=filter_size,strides=stride_amount, activation='relu'))
            model.add(GlobalMaxPooling1D()) 
            model.add(Dense(64, kernel_initializer="uniform", activation='relu') )
            model.add(Dropout(dropout_ratio))
            model.add(Dense(2,kernel_initializer="uniform",activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer=Nadam() )
         
            arrays = [X for i in range(no_of_words)]
            X=np.stack(arrays, axis=1)
            arrays_test = [X_test for i in range(no_of_words)]
            X_test=np.stack(arrays_test, axis=1)

            model.fit(X,Y,epochs=number_of_epochs)  

        elif DL_Method_Choice == "Vanilla RNN":
            model = Sequential()
            model.add(SimpleRNN(64, input_shape=(no_of_words,no_of_features)))
            model.add(Dropout(dropout_ratio))
            model.add(Dense(2,kernel_initializer="uniform",activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer=Nadam())
            
            arrays = [X for i in range(no_of_words)]
            X=np.stack(arrays, axis=1)
            arrays_test = [X_test for i in range(no_of_words)]
            X_test=np.stack(arrays_test, axis=1)
            
            model.fit(X,Y,epochs=number_of_epochs)    

        elif DL_Method_Choice == "LSTM":
            model = Sequential()
            model.add(LSTM(64, input_shape=(no_of_words,no_of_features)))
            model.add(Dropout(dropout_ratio))
            model.add(Dense(2,kernel_initializer="uniform",activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer=Nadam())
            
            arrays = [X for i in range(no_of_words)]
            X=np.stack(arrays, axis=1)
            arrays_test = [X_test for i in range(no_of_words)]
            X_test=np.stack(arrays_test, axis=1)

            model.fit(X,Y,epochs=number_of_epochs) 
            
        elif DL_Method_Choice == "LSTM with CNN":
            model = Sequential()
            model.add(LSTM(64,input_shape=(no_of_words,no_of_features),return_sequences=True) )
            model.add(Conv1D(filters=32, kernel_size=filter_size,strides=stride_amount, activation='relu'))
            model.add(GlobalAveragePooling1D())
            model.add(Dropout(dropout_ratio))
            model.add(Dense(2,kernel_initializer="uniform",activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer=Nadam() )
            
            arrays = [X for i in range(no_of_words)]
            X=np.stack(arrays, axis=1)
            arrays_test = [X_test for i in range(no_of_words)]
            X_test=np.stack(arrays_test, axis=1)

            model.fit(X,Y,epochs=number_of_epochs)


        #print("X shape ", X.shape)
        #print("Y shape", Y.shape)
        #print("X_test shape is: ",X_test.shape)

        #predicted_classes = model.predict_classes(X_test) # predict_classes is deprecated now
        predicted_classes=np.argmax(model.predict(X_test), axis=-1) #multiclass classification
        #predicted_classes=(model.predict(X_test)>0.5).astype("int32")
       
        comparison_result_array = np.equal(predicted_classes,Y_test)
        print("comparison array is: ", comparison_result_array)
        no_of_True = np.count_nonzero(comparison_result_array)
        DL_Accuracy = no_of_True / comparison_result_array.size 

        print("DL Accuracy is: {} %".format(DL_Accuracy*100))
        DL_Accuracy_Text = "DL Accuracy is: {} %".format(DL_Accuracy*100) 
          

        print("Ground truth classes are:\n ", Y_test)
        DL_GroundTruth_Classes_Text = "Ground truth classes are:\n {} ".format(Y_test)

        print("Predicted classes are:\n ", predicted_classes)
        DL_Predicted_Classes_Text = "Predicted classes are:\n {} ".format(predicted_classes)


        DL_executed = True
        currentFunction = "None"
        print("Completed running the Deep Learning Methods on NLP Data...")

      except Exception as ex:
        print("Encountered an error ",ex," while running DL on the data, please run DL again with right parameters...")
        print(traceback.format_exc())
        currentFunction = "None"
        continue

  t1.join()

if __name__ == '__main__':
    main(sys.argv[1:])