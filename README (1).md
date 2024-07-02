
# MovieSel

MovieSel is sophisticated machine learning model designed to recommend movies tailored to your preferences. By analyzing ratings, votes, and population data.

# Machine learning  | notebook

Training using https://deepnote.com/


## Features

- User Preference Analysis: Analyzes user ratings, viewing history, and preferences to tailor recommendations.
- Popularity Metrics: Incorporates ratings, votes, and population data to suggest trending and highly-rated movies.
- Diverse Recommendations: Offers a mix of popular blockbusters and lesser-known indie films to suit varied tastes.
- Continuous Learning: Adapts to changing user preferences over time, improving recommendation accuracy.
- Real-Time Updates: Delivers the latest movie suggestions based on current data and trends.
- Cross-Platform Integration: Works seamlessly across different devices, ensuring consistent recommendations everywhere.









## Note from Developer
I’ve been itching to dive into machine learning since before college, and now I’ve finally built something useful—a movie recommendation model!

It's a work in progress, so bear with me as I keep tweaking it. Enjoy finding your next favorite flick!
## Data Resource

[Data Resource](https://drive.google.com/drive/folders/1GrIWAMTGoCXlaE6o2RHgGoNmusgBycQG?usp=drive_link)


## Import
    import pandas as pd

## Reading the CSV files
movies = pd.read_csv('movies.csv')

credits = pd.read_csv('credits.csv')

ratings = pd.read_csv('ratings.csv')

## Top 10 Movies based on Population Filter
### Notebook 1:
Here is where I train the model to sort movies from dataset that I provided to sort out top 10 movies that based on Population Filter.

    top_10_popular_movies = movies.sort_values(by='popularity', ascending=False).head(10)
    top_10_popular_movies[['title', 'popularity']]

![App Screenshot](<img width="881" alt="Screenshot 2024-07-02 at 5 44 29 PM" src="https://github.com/Kaydenletk/Movie-Recommendation/assets/111254859/160b099a-5622-445d-b070-af3bb21c7c97">
>
)

---

Also, I trained the machine to calculate the weighted rating from the data that I provide.

WR = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C 

    def weighted_rating(df, m=m, C=C):
    v = df['vote_count']
    R = df['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
---

    movies['weighted_rating'] =movies.apply(weighted_rating, axis=1)

---

    movies_filtered = movies.copy().loc[movies['vote_count'] >= m]
    movies_filtered
    <img width="590" alt="Screenshot 2024-07-02 at 5 56 32 PM" src="https://github.com/Kaydenletk/Movie-Recommendation/assets/111254859/09d4de29-eb68-4999-ba20-f23ed5d9fb5c">
---
    movies_filtered.sort_values(by='weighted_rating', ascending=False).head(10)




## Content-Based Filtering
### Notebook 2
---
### Import

    import pandas as pd
    movies = pd.read_csv("movies.csv", sep=",")
---
### Installation

    !pip install scikit-learn

### Converts a TF-IDF matrix into a reable DataFrame
Transfrom TextData into TF-IDF Matrix, Convert Sparse Matrix to Dense Matrix and Create a DataFrame from the Dense Matrix: 

    tfidf_matrix_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
    tfidf_matrix_df

### Find the most similar movies to a certain movies
This code generating the similar "title", "scores", sort scores and get indices of Most Similar Movies

    def similar_movies(movie_title, nr_movies):
    idx = movies.loc[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse= True)
    movies_indices = [tpl[0] for tpl in scores[1:nr_movies+1]]
    similar_title = list(movies["title"].iloc[movies_indices])
    return similar_title
--- 
---
Example:

    similar_movies("Kung Fu Panda 3", 3)
---
return

    ['Kung Fu Panda 2',
    'My Big Fat Greek Wedding 2',
    'Once Upon a Time in the West']

## Collaborative-Based Filtering
### Import
    import pandas
    ratings = pandas.read_csv('ratings.csv')[["userId", "movieId", "rating"]]
    ratings.head()
### Create the dataset
Using the Rating dataset that ranking 1 to 5

    from surprise import Dataset, Reader
    reader = Reader(rating_scale=(1.0, 5.0))
    dataset = Dataset.load_from_df(ratings, reader=Reader())

### Building the trainset
With this trainset, the model trained with the "userId", "movieId" and "rating" dataset to to predict the any userId and movieID provided

    trainset = dataset.build_full_trainset()
    list(trainset.all_ratings())

 <img width="742" alt="Screenshot 2024-07-02 at 6 37 50 PM" src="https://github.com/Kaydenletk/Movie-Recommendation/assets/111254859/cfff81ce-fd2b-4abd-b63e-0f978dcccbe1">

### Training the model
Prepare the data and train the SVD using the 'surprise' library:

    from surprise import SVD
    svd = SVD()
    svd.fit(trainset)   

Predict the userId: '14', movieId: '1956' using the [rating.cvs](https://drive.google.com/drive/u/1/folders/1GrIWAMTGoCXlaE6o2RHgGoNmusgBycQG)

    svd.predict(14, 1956)

Estimated rating about 3.465

    Prediction(uid=14, iid=1956, r_ui=None, est=3.4635761214305716, details={'was_impossible': False})

## Feedback
Embarking on this project during my free time has been an incredible journey, deepening my knowledge of machine learning and data science. I would love to hear your thoughts and feedback!