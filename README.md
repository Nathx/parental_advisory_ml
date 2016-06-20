# Idea 1: NLP Movie Classification
## High level description
Use movie scripts extracted from subtitle files to classify movies according to their movie genre or their MPAA rating. The starting point will be the 150k subtitles corpus from OpenSubtitles, crawled data from IMDB, possibly the 1M song dataset. The goal is to use NLP techniques and look for predictive power for the following:  

1. Movie features:  
    1. Movie genre  
    2. Release Decade/Year  
    3. Country of origin  
2. MPAA ratings  
    1. G/PG/PG-13/R/NC-17  
    2. Predict rating subcategory ("drug use", "explicit content", "violence", etc      
    3. Predict certification for various countries  
    http://www.imdb.com/title/tt2278871/parentalguide?ref_=tt_stry_pg#certification  
3. Predict soundtrack  
    1. Crawl http://www.soundtrack.net/ based on movie subs dataset  
    2. Intersect w/ 1 million song dataset to get song features  
    3. Cluster, train with movies with labeled soundtracks  
    4. Build suggestion interface (upload a movie, suggest soundtrack)  

## Presenting Work
* Web app - select a film or upload your subtitle
-> Returns predicted features.
* Github
* Description blog post w/ charts and detailed implementation
* Slides.

## Next step
* Crawl IMDB.
* Quickly assess feasibility of 3. Explore million song dataset, look for intersection with subtitles corpus, evaluate crawling process times.
* Push data to AWS. Build work environment/dummy website for displaying results.

## Datasets
- Open Subtitles corpus: 150k movie subtitles sorted by IMDB ID.
- IMDB metadata crawling.
- soundtrack.net
- Million song dataset.

## Pipeline

### Extracting features
* Extract text from subtitles

* Extract features from text
  * TfIdf
  * Part-of-speech tagging (syntaxnet?)
  * word2vec (gensim)

* Build target by crawling
* Explore data

### Modeling
* Movie Classification
  * Clustering
  * Softmax regression
  * CNNs? explore possibilities here
  * Bayesian Hyperparameter optimization (hyperopt, spearmint)

* Soundtrack prediction
  * Extract song features from million song dataset
  * Cluster songs
  * Extract features from lyrics (if available)
  * Cluster songs with movies?
  * Find closest songs to a movie

### Scale up
* Migrate data to S3
* Preprocess with Spark
* Text classification with MLLib

### Interface
* Build charts to explain data
* Build charts to explain data
* Build interface with Flask
