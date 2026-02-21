from flask import Flask,request,render_template
import re
import nltk # nltk-3.9.2-py3-none-any.whl.metadata 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# flask App
app = Flask(__name__)
# loading files and data
cv = pickle.load(open("count_vectorizer.pkl","rb"))
feature_name = pickle.load(open("feature_names.pkl","rb"))
tfidf_transformer = pickle.load(open("tfidf_transformer.pkl","rb"))





stop_words = set(stopwords.words('english'))

## Creating  a list of custom stopwords
new_words = [
    "fig","figure","image","sample","using",
    "show","result","large","also","one","two","three","four","five",
    "six","seven","eight","nine","ten"
]

stop_words = list(stop_words.union(set(new_words)))

# Custom function

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)   
    # Convert to lowercase
    text = text.lower()    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove stop words
    text = [word for word in text if word not in stop_words]
    text = [word for word in text if len(word) > 3] # Remove short words   
    # Remove stop words and perform stemming
    stemmer = PorterStemmer()
    processed_tokens = [stemmer.stem(word) for word in text]
    
    return ' '.join(processed_tokens)


def get_keywords(docs,top_n=2 0):
      docs_words_count = tfidf_transformer.transform(cv.transform([docs]))
      # sorting sparse matrix
      docs_words_count = docs_words_count.tocoo()
      tuples = zip(docs_words_count.col, docs_words_count.data)
      # sorting tuples
      sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

      # getting top 20 keywords
      sorted_items = sorted_items[:top_n]
      
      keywords = []
      score_vals = []
      for idx, score in sorted_items:
            keywords.append(feature_name[idx])
            score_vals.append(round(score,3))

      # finally return the top n
      result = {}
      for i in range(len(keywords)):
            result[keywords[i]] = score_vals[i]
      return result




# routes
@app.route('/')
def index():
    return render_template("index.html")
# extracting keywords
@app.route('/extract_keyboards',methods=["POST","GET"])
def extract_keywords():
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html",error="No file selected")
    if file:
        file = file.read().decode('utf-8',errors='ignore')
        # cleaning the text
        cleaned_text = preprocess_text(file)
        # transforming the text
        keywords = get_keywords(cleaned_text)
        return render_template("keyword.html",keywords=keywords)
    return render_template("index.html")
        
    

# search keywords
@app.route('/search_keyboards',methods=["POST","GET"])
def search_keywords():
    search_keywords = request.form.get("search")
    if search_keywords:
        keywords = []
        for keyword in feature_name:
             if search_keywords.lower() in keyword.lower():
                    keywords.append(keyword)
                    if len(keywords) == 50: # 50 keywords only
                        break
        print(keywords)
        return render_template("keywordslist.html",keywords=keywords)
    return render_template("index.html")




# python main
if __name__ == "__main__":
    app.run(debug=True)