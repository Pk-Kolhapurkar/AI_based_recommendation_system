import os
import pickle   #It enables objects to be converted into a byte stream for storage or transmission, and then reconstructed back into their original form later.
import torch
import pickle
import matplotlib.pyplot as plt
from langchain_community.document_loaders import TextLoader
from datasets import load_dataset  #This function is typically used to load datasets for natural language processing tasks from the Hugging Face datasets library.
from sentence_transformers import SentenceTransformer, util  ##library for computing dense vector representations of sentences or text snippets.
from transformers import GPT2LMHeadModel, GPT2Tokenizer #provides pre-trained models for natural language understanding and generation tasks.
from transformers import BertModel, BertTokenizer
from langchain_core.prompts import PromptTemplate

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

class Chatbot:
    def __init__(self):
        self.load_data()
        self.load_models()
        self.load_embeddings()
        self.load_template()

    def load_data(self):
        self.data = load_dataset("ashraq/fashion-product-images-small", split="train")
        self.images = self.data["image"]  ##Extracts the image data from the loaded dataset and assigns it to the self.images attribute. This assumes that the dataset contains a column named "image" containing image data.
        self.product_frame = self.data.remove_columns("image").to_pandas()   #Removes the "image" column from the dataset and converts the modified dataset to a Pandas DataFrame
        self.product_data = self.product_frame.reset_index(drop=True).to_dict(orient='index')
        '''
         Resets the index of the Pandas DataFrame to ensure it starts from 0 and converts it to a dictionary where the keys are the indices of the DataFrame (starting from 0) 
         and the values are dictionaries representing the rows of the DataFrame. The orientation of the dictionary is set to 'index', meaning that each row of the DataFrame is represented by a key-value pair in the dictionary.
        '''

    def load_template(self):    #function is responsible for loading a template string for generating chatbot responses
        self.template = """
        You are a fashion shopping assistant that wants to convert customers based on the information given.
        Describe season and usage given in the context in your interaction with the customer.
        Use a bullet list when describing each product.
        If user ask general question then answer them accordingly, the question may be like when the store will open, where is your store located.
        Context: {context}
        User question: {question}
        Your response: {response}
        """
        self.prompt = PromptTemplate.from_template(self.template) ##This object will be used later by the chatbot to generate responses based on user queries.
    ##########Overall, this function initializes the chatbot's response template, providing guidance on how the chatbot should interact with users and structure its response
    
    def load_models(self):
        self.model = SentenceTransformer('clip-ViT-B-32')  ###Transformers are capable of converting sentences or text snippets into dense vector representations
        self.bert_model_name = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(self.bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.gpt2_model_name = "gpt2"
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(self.gpt2_model_name)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_model_name)

    def load_embeddings(self):
        if os.path.exists("embeddings_cache.pkl"):  #check file exist or not
            with open("embeddings_cache.pkl", "rb") as f:  #Opens the cache file in binary read mode if it exists.
                embeddings_cache = pickle.load(f)  ##Loads the contents of the cache file and assign to dictionary 
                #This assumes that the file contains a dictionary with keys "image_embeddings" and "text_embeddings" representing precomputed embeddings for images and text data, respectively.
                #if already calculated then loadand assign
            self.image_embeddings = embeddings_cache["image_embeddings"]
            self.text_embeddings = embeddings_cache["text_embeddings"]
        else:
            self.image_embeddings = self.model.encode([image for image in self.images])  #self.model.encode([image for image in self.images]): Computes image embeddings for all images in self.images using the encode method of the Sentence Transformer model (self.model). The resulting embeddings are assigned to the self.image_embeddings attribute.
            self.text_embeddings = self.model.encode(self.product_frame['productDisplayName'])
            # model =clip-ViT-B-32 this model used
            embeddings_cache = {"image_embeddings": self.image_embeddings, "text_embeddings": self.text_embeddings}  # created dictionry
            with open("embeddings_cache.pkl", "wb") as f: #open forwriting
                pickle.dump(embeddings_cache, f)  #dump = save

    def create_docs(self, results): ####Iterates over each result in the results list, where each result is a dictionary containing information about a product.
        docs = []
        for result in results:
            pid = result['corpus_id']
            score = result['score']
            result_string = ''
            result_string += "Product Name:" + self.product_data[pid]['productDisplayName'] + \
                             ';' + "Category:" + self.product_data[pid]['masterCategory'] + \
                             ';' + "Article Type:" + self.product_data[pid]['articleType'] + \
                             ';' + "Usage:" + self.product_data[pid]['usage'] + \
                             ';' + "Season:" + self.product_data[pid]['season'] + \
                             ';' + "Gender:" + self.product_data[pid]['gender']
            # Assuming text is imported from somewhere else######################################################################################
            doc = text(page_content=result_string)
            doc.metadata['pid'] = str(pid)###The pid metadata field stores the unique identifier of the product. This identifier allows for easy retrieval and reference to the corresponding product when needed. For example, it can be used to link the document back to the original product data, such as its name, description, or image.
                                          ###Storing the product ID as metadata ensures that the document object retains a connection to the specific product it represents, enabling further processing or analysis based on the product's identity
                                          ###The score metadata field stores the similarity score associated with the product. This score indicates the relevance or similarity of the product to the user query or search criteria used to retrieve the product.
                                          ###Storing the score as metadata allows for ranking or filtering of documents based on their relevance to the user query. It provides valuable information about the quality or suitability of the product in relation to the user's needs or preferences.
            doc.metadata['score'] = score
            docs.append(doc)
        return docs

    def get_results(self, query, embeddings, top_k=10):
        query_embedding = self.model.encode([query])  #Encodes the query into a dense vector representation 
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0] #pytorch_cos_sim computes the cosine similarity between two sets of embeddings. we get tensor 
        '''
        Computes the cosine similarity between the query embedding and the embeddings of all items in the dataset.
        
        The [0] after util.pytorch_cos_sim(query_embedding, embeddings) is used to access the first element of the output tensor, which is necessary because util.pytorch_cos_sim returns a tuple containing the cosine similarity scores and the corresponding indices.
        '''
        top_results = torch.topk(cos_scores, k=top_k)#Finds the indices and values of the top-k highest similarity scores
        indices = top_results.indices.tolist() #Extracts the indices of the top-k results from the top_results tensor and converts them to a Python list
        scores = top_results.values.tolist()
        results = [{'corpus_id': idx, 'score': score} for idx, score in zip(indices, scores)]
        '''
        Constructs a list of dictionaries representing the top-k results.
        Each dictionary contains two key-value pairs: 'corpus_id' (index of the item in the dataset) and 'score' (similarity score between the query and the item).
        The zip function pairs each index (idx) with its corresponding score, and a dictionary is created for each pair.
        '''

        return results  # list containing the top-k most similar items, along with their similarity scores.

    def display_text_and_images(self, results_text):
        for result in results_text:
            pid = result['corpus_id']
            product_info = self.product_data[pid]
            print("Product Name:", product_info['productDisplayName'])
            print("Category:", product_info['masterCategory'])
            print("Article Type:", product_info['articleType'])
            print("Usage:", product_info['usage'])
            print("Season:", product_info['season'])
            print("Gender:", product_info['gender'])
            print("Score:", result['score'])
            plt.imshow(self.images[pid]) #Displays the image of the product associated with the product ID pid
            plt.axis('off') #turn off lables
            plt.show()

    @staticmethod
    def cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm.T, b_norm)  # Reshape a_norm to (768, 1)

    def generate_response(self, query):
        # Process the user query and generate a response
        results_text = self.get_results(query, self.text_embeddings)

        # Generate chatbot response
        chatbot_response = "Here are some products for you."  

        # Display recommended products
        self.display_text_and_images(results_text)

        # Return both chatbot response and recommended products
        return chatbot_response,results_text

