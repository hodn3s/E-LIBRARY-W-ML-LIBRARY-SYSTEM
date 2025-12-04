import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{encoded}");
    background-size: cover;       /* fit the entire image inside container */
    background-position: top; /* align image to top center */
    background-repeat: no-repeat;   /* don't repeat */
    width: 100%;
    min-height: 100vh;              /* ensures container is at least viewport height */
    /* background-color: transparent; optional */
    }}
    
     label[aria-hidden="true"] div[data-testid="stMarkdownContainer"] > p {{
        color: black !important;       /* font color black */
        font-size: 16px !important;    /* increase font size */
        font-weight: bold !important;  /* make it bold */
        text-align: center;

    }}

    div[data-testid="stSelectbox"] > div > div > div > div {{
        background-color: white !important;  /* white background */
        color: black !important;             /* black text */
    }}

    [data-testid="stSidebar"] {{
        background: rgba(80, 80, 80, 0.8); !important;
        
    }}

    .stButton>button {{
        background-color: #505050;  /* solid gray */
        color: black;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.6);
        width: 150px;             /* fixed width */
        height: 50px;             /* fixed height */
    }}
    .stButton>button:hover {{
        background-color: gray !important;
        color: black !important;
        box-shadow: 0px 6px 10px rgba(0,0,0,0.8); /* stronger shadow when hovering */
        border: 2px solid white;
        width: 150px;             /* fixed width */
        height: 50px;             /* fixed height */
    }}
    .books-container {{
    display: flex;
    flex-wrap: nowrap;
    justify-content: flex-start;
    gap: 20px;
    margin-top: 20px;
    overflow-x: auto;
    padding-bottom: 10px;
    color: black !important; /* optional, can stay */
}}

/* Individual book cards */
.book-card {{
    background: rgba(0, 0, 0, 0.6);
    border-radius: 15px;
    padding: 10px;
    text-align: center;
    color: black !important; /* change from white to black */
    width: 250px;
    flex: 0 0 250px;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.5);
    transition: transform 0.2s, box-shadow 0.2s;
}}

.book-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0px 8px 12px rgba(0,0,0,0.6);
}}

.book-card img {{
    width: 100%;
    height: 350px;
    object-fit: cover;
    border-radius: 10px;
}}

.book-card h4 {{
    margin: 20px 0 0 0;
    font-size: 18px;
    font-color: black;
   
}}


    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        position: fixed;
        bottom: 200px;
        left: 50%;
        transform: translateX(-50%);
        text-align: center;
        z-index: 1000;
    ">
    <a href="https://ser-infotech.com/user" target="_blank"
        style="
            background: red;
            padding: 12px 28px;
            color: white;
            text-decoration: none;
            border: 2px solid white;
            border-radius: 999px;   /* makes it fully round */
            font-weight: bold;
            cursor: pointer;
            display: inline-block;
        ">
        Log-In to CDM E-LIBRARY
    </a>


    </div>
""", unsafe_allow_html=True)




# ------------------------------------------------------
# CALL THE BACKGROUND FUNCTION
# ------------------------------------------------------
add_bg_from_local("background1.png")




class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects,'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]:
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path,'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(
                book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6
            )

            poster_url = self.fetch_poster(suggestion)

            for i in range(len(suggestion)):
                books = book_pivot.index[suggestion[i]]
                for j in books:
                    books_list.append(j)
            return books_list, poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.text("Training Completed!")
            logging.info(f"Training pipeline executed successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_books):
        try:
            recommended_books, poster_url = self.recommend_book(selected_books)
            container = st.container()  # create a container
            with container:
                col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
            with col1:
                st.text(recommended_books[1])
                st.image(poster_url[1])
            with col2:
                st.text(recommended_books[2])
                st.image(poster_url[2])
            with col3:
                st.text(recommended_books[3])
                st.image(poster_url[3])
            with col4:
                st.text(recommended_books[4])
                st.image(poster_url[4])
            with col5:
                st.text(recommended_books[5])
                st.image(poster_url[5])
            with col6:
                st.text(recommended_books[1])
                st.image(poster_url[1])
            with col7:
                st.text(recommended_books[2])
                st.image(poster_url[2])
            with col8:
                st.text(recommended_books[3])
                st.image(poster_url[3])
            with col9:
                st.text(recommended_books[4])
                st.image(poster_url[4])
            with col10:
                st.text(recommended_books[5])
                st.image(poster_url[5])
        except Exception as e:
            raise AppException(e, sys) from e




# ---------------- Main App ---------------- #
if __name__ == "__main__":

    # --- CENTER EVERYTHING ---
    st.write("")  # optional spacing
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

   # Keep the selectbox and buttons in the center column
left, center, right = st.columns([1,2,1])

with center:
    obj = Recommendation()
    book_names = pickle.load(open(os.path.join('templates', 'book_names.pkl'), 'rb'))
    selected_books = st.selectbox(
        "What kind of books that interest you? select from dropdown below",
        book_names
    )

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button('Train'):
            obj.train_engine()
    with col2:
        if st.button('Recommend'):
            pass  # trigger recommendation

# --- Move recommendations outside the center column ---
obj.recommendations_engine(selected_books)  # now can take full page width

