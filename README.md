# Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-brightgreen)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-red)

## Overview

This project is an advanced movie recommendation system built using the MovieLens 100K dataset. It demonstrates the application of big data tools and machine learning techniques to create a personalized movie recommendation experience.

The system implements multiple recommendation strategies:

- **Traditional collaborative filtering** using Pearson correlation on a user-rating pivot table
- **Advanced matrix factorization** using Non-negative Matrix Factorization (NMF) to extract latent factors
- **Fuzzy matching** to handle partial or approximate movie title inputs
- **Dynamic model updates** through a user feedback loop that captures ratings and updates the model
- **Interactive web dashboard** built with Streamlit for real-time recommendations and feedback
- **Enhanced logging system** with colorful, emoji-enhanced messages for better debugging

## Key Features

### 1. Data Preprocessing
- Loads and merges MovieLens data from the original file formats
- Prepares a unified dataset with userId, movieId, rating, timestamp, and movie title
- Handles missing values and data normalization

### 2. Recommendation Engines
- **Traditional Recommendation**: Uses a pivot table and Pearson correlation to find similar movies
- **Advanced Recommendation**: Implements NMF-based matrix factorization and cosine similarity for improved recommendations
- **Fuzzy Matching**: Accepts partial movie titles and finds the closest match using fuzzy logic algorithms

### 3. Dynamic Model Updates
- Captures user feedback from the dashboard (ratings on recommendations)
- Incorporates feedback as additional ratings and retrains the NMF model to adapt over time
- Saves model state for persistence between sessions

### 4. Interactive Dashboard
- Built with Streamlit for an intuitive user interface
- Allows users to select a movie, view recommendations, and submit feedback via a form
- Uses st.form to prevent unnecessary re-runs during slider adjustments
- Includes an "Exit Dashboard" button to return control to the terminal

### 5. Logging System
- A custom logger provides colorful, emoji-enhanced logging messages
- Tracks key events (e.g., dashboard launch, recommendation generation, feedback submission)
- Helps with debugging and monitoring system performance

## Project Structure

```
movie-recommendation-big-data-tools/
├── data/                       # MovieLens 100K dataset files
├── app.py                      # Streamlit dashboard for interactive recommendations
├── main.py                     # Unified main file with text-based menu
├── recommendation_engine.py    # Traditional recommendation logic
├── advanced_recommender.py     # Advanced recommendations using NMF
├── data_preprocessing.py       # Data loading and preprocessing
├── dynamic_update.py           # User feedback incorporation and model updates
├── logger.py                   # Custom logger with colorful, emoji-enhanced logging
├── README.md                   # Project documentation
└── feedback.csv                # (Generated at runtime) User feedback storage
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gamidirohan/movie-recommendation-big-data-tools.git
   cd movie-recommendation-big-data-tools
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn streamlit joblib colorlog thefuzz[speedup]
   ```

4. The MovieLens 100K dataset is included in the repository. If you need to download it again:
   ```bash
   # Download from https://grouplens.org/datasets/movielens/100k/
   # Extract the files into the 'data' folder
   ```

## Usage

### Option 1: Using the Main Menu

Run the main script to access all features through a text-based menu:

```bash
python main.py
```

This will present a menu with options for:
1. Getting traditional recommendations based on correlation
2. Viewing a sample of the merged MovieLens data
3. Launching the interactive Streamlit dashboard
4. Getting advanced recommendations using NMF-based matrix factorization
5. Updating the dynamic model to incorporate new user feedback
6. Exiting the program

### Option 2: Using the Streamlit Dashboard

Launch the interactive dashboard directly:

```bash
streamlit run app.py
```

With the dashboard, you can:
- Search for movies using partial titles (fuzzy matching)
- View personalized recommendations
- Rate recommended movies to improve future suggestions
- See your feedback history

## Future Enhancements

- Develop a hybrid recommender that combines collaborative and content-based filtering
- Deploy the system as a REST API using Flask or FastAPI
- Enhance the dashboard with additional visualizations and user analytics
- Integrate online learning algorithms to update the model continuously
- Implement CI/CD pipelines and containerize the project for production deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, suggestions, or contributions, please open an issue or submit a pull request on GitHub.

---

Thank you for exploring the Movie Recommendation System project!
